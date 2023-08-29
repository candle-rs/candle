use super::k_quants::{BlockQ4K, BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K};
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[inline(always)]
pub(crate) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
pub(crate) unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4));
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    let nb = n / qk;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {n} is not divisible by {qk}")
    }
    if nb % 2 != 0 {
        crate::bail!("vec_dot_q4_0_q8_0: {nb} is not even")
    }

    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = bytes_from_nibbles_32(x.qs.as_ptr());
            let off = _mm256_set1_epi8(8);
            let bx = _mm256_sub_epi8(bx, off);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        Ok(hsum_float_8(acc))
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> Result<f32> {
    let qk = QK8_0;
    if n % QK8_0 != 0 {
        crate::bail!("vec_dot_q8_0_q8_0: {n} is not divisible by {qk}")
    }
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        Ok(hsum_float_8(acc))
    }
}

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> Result<f32> {
    let qk = QK_K;
    if n % qk != 0 {
        crate::bail!("vec_dot_q6k_8k: {n} is not divisible by {qk}")
    }

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let m32s = _mm256_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                qh = qh.add(32);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                let q4h_1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                let q4h_2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                let q4h_3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        Ok(hsum_float_8(acc))
    }
}

#[inline(always)]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> Result<f32> {
    if n % QK_K != 0 {
        crate::bail!("vec_dot_q4k_q8k: {n} is not divisible by {QK_K}")
    }
    let mut utmp = [0u32; 4];
    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            let uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16l = _mm256_madd_epi16(scale_l, p16l);
                sumi = _mm256_add_epi32(sumi, p16l);

                let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);
                let p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi = _mm256_add_epi32(sumi, p16h);
            }

            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }

        let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

        Ok(hsum_float_8(acc) + _mm_cvtss_f32(acc_m))
    }
}
