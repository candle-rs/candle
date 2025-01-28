use super::*;
use half::{bf16, f16};
use metal::{Buffer, Device, MTLResourceOptions};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rand::Rng;

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = MTLResourceOptions::StorageModeManaged;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data) as u64;
    device.new_buffer_with_data(ptr, size, options)
}

fn device() -> Device {
    Device::system_default().unwrap()
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: 0,
    };
    let output = new_buffer(&device, v);
    call_unary_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        input,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

fn run_binary<T: Clone>(x: &[T], y: &[T], name: binary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device.new_buffer(std::mem::size_of_val(x) as u64, options);
    call_binary_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        x.len(),
        BufferOffset::zero_offset(&left),
        BufferOffset::zero_offset(&right),
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, x.len())
}

fn run_strided<T: Clone>(
    v: &[T],
    kernel: unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Vec<T> {
    let device = device();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: offset,
    };
    let output_b = new_buffer(&device, v);
    let output = BufferOffset {
        buffer: &output_b,
        offset_in_bytes: 0,
    };
    let kernels = Kernels::new();
    call_unary_strided(
        &device,
        command_buffer,
        &kernels,
        kernel,
        shape,
        input,
        strides,
        output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output_b, v.len())
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

    let v = vec![1.0f32; 10_000];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.16, 0.0, 0.84, 1.96, 3.0, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.955, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn silu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.27, 0.0, 0.73, 1.76, 2.86, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn silu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.269, 0.0, 0.731, 1.762, 2.858, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, binary::contiguous::add::FLOAT);
    let expected: Vec<_> = left
        .iter()
        .zip(right.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, binary::contiguous::$opname::BFLOAT);
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bf16, &bf16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(min, |x: bf16, y| x.min(y));
    binary_op!(max, |x: bf16, y| x.max(y));
}

fn run_cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let options = MTLResourceOptions::StorageModeManaged;
    let size = (v.len() * std::mem::size_of::<U>()) as u64;
    let output = device.new_buffer(size, options);

    call_cast_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        BufferOffset::zero_offset(&input),
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

#[test]
fn cast_f32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f32 -> f16
    let results: Vec<half::f16> = run_cast(&v_f32, "cast_f32_f16");
    assert_eq!(results, v_f16);

    // f32 -> bf16
    let results: Vec<bf16> = run_cast(&v_f32, "cast_f32_bf16");
    assert_eq!(results, v_bf16);

    // f32 -> u32
    let results: Vec<u32> = run_cast(&v_f32, "cast_f32_u32");
    assert_eq!(results, v_u32);

    // f32 -> u8
    let results: Vec<u8> = run_cast(&v_f32, "cast_f32_u8");
    assert_eq!(results, v_u8);

    // f32 -> i64
    let results: Vec<i64> = run_cast(&v_f32, "cast_f32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_f16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f16 -> f32
    let results: Vec<f32> = run_cast(&v_f16, "cast_f16_f32");
    assert_eq!(results, v_f32);

    // f16 -> bf16
    let results: Vec<bf16> = run_cast(&v_f16, "cast_f16_bf16");
    assert_eq!(results, v_bf16);

    // f16 -> u32
    let results: Vec<u32> = run_cast(&v_f16, "cast_f16_u32");
    assert_eq!(results, v_u32);

    // f16 -> u8
    let results: Vec<u8> = run_cast(&v_f16, "cast_f16_u8");
    assert_eq!(results, v_u8);

    // f16 -> i64
    let results: Vec<i64> = run_cast(&v_f16, "cast_f16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_bf16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // bf16 -> f32
    let results: Vec<f32> = run_cast(&v_bf16, "cast_bf16_f32");
    assert_eq!(results, v_f32);

    // bf16 -> f16
    let results: Vec<f16> = run_cast(&v_bf16, "cast_bf16_f16");
    assert_eq!(results, v_f16);

    // bf16 -> u32
    let results: Vec<u32> = run_cast(&v_bf16, "cast_bf16_u32");
    assert_eq!(results, v_u32);

    // bf16 -> u8
    let results: Vec<u8> = run_cast(&v_bf16, "cast_bf16_u8");
    assert_eq!(results, v_u8);

    // bf16 -> i64
    let results: Vec<i64> = run_cast(&v_bf16, "cast_bf16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u32 -> f32
    let results: Vec<f32> = run_cast(&v_u32, "cast_u32_f32");
    assert_eq!(results, v_f32);

    // u32 -> f16
    let results: Vec<f16> = run_cast(&v_u32, "cast_u32_f16");
    assert_eq!(results, v_f16);

    // u32 -> bf16
    let results: Vec<bf16> = run_cast(&v_u32, "cast_u32_bf16");
    assert_eq!(results, v_bf16);

    // u32 -> u8
    let results: Vec<u8> = run_cast(&v_u32, "cast_u32_u8");
    assert_eq!(results, v_u8);

    // u32 -> i64
    let results: Vec<i64> = run_cast(&v_u32, "cast_u32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u8() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u8 -> f32
    let results: Vec<f32> = run_cast(&v_u8, "cast_u8_f32");
    assert_eq!(results, v_f32);

    // u8 -> f16
    let results: Vec<f16> = run_cast(&v_u8, "cast_u8_f16");
    assert_eq!(results, v_f16);

    // u8 -> bf16
    let results: Vec<bf16> = run_cast(&v_u8, "cast_u8_bf16");
    assert_eq!(results, v_bf16);

    // u8 -> u32
    let results: Vec<u32> = run_cast(&v_u8, "cast_u8_u32");
    assert_eq!(results, v_u32);

    // u8 -> i64
    let results: Vec<i64> = run_cast(&v_u8, "cast_u8_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_i64() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // i64 -> f32
    let results: Vec<f32> = run_cast(&v_i64, "cast_i64_f32");
    assert_eq!(results, v_f32);

    // i64 -> f16
    let results: Vec<f16> = run_cast(&v_i64, "cast_i64_f16");
    assert_eq!(results, v_f16);

    // i64 -> bf16
    let results: Vec<bf16> = run_cast(&v_i64, "cast_i64_bf16");
    assert_eq!(results, v_bf16);

    // i64 -> u32
    let results: Vec<u32> = run_cast(&v_i64, "cast_i64_u32");
    assert_eq!(results, v_u32);

    // i64 -> u8
    let results: Vec<u8> = run_cast(&v_i64, "cast_i64_u8");
    assert_eq!(results, v_u8);
}

fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let size = v.len();

    call_affine(
        &device,
        command_buffer,
        &kernels,
        "affine_f32",
        size,
        BufferOffset::zero_offset(&input),
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, v.len())
}

fn run_affine_strided<T: Clone>(
    v: &[T],
    shape: &[usize],
    strides: &[usize],
    mul: f64,
    add: f64,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    call_affine_strided(
        &device,
        command_buffer,
        &kernels,
        "affine_f32_strided",
        shape,
        BufferOffset::zero_offset(&input),
        strides,
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let len: usize = shape.iter().product();
    read_to_vec(&output, len)
}

#[test]
fn affine() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

    let input = [1.0f32; 40_000];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6; 40_000]);
}

#[test]
fn affine_strided() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let shape = [4];
    let strides = [2];
    let result = run_affine_strided(&input, &shape, &strides, mul, add);
    // 1 on 2
    assert_eq!(result, vec![2.6, 5.6, 8.6, 11.6]);
}

#[test]
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let stride = [1, 2];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn index_select_strided() {
    let embedding = (0..16).map(|x| x as f32).collect::<Vec<_>>();
    let shape = [2, 2];
    let stride = [2, 4];
    let ids = [0u32];
    let dim = 0;
    let result = run_index_select_strided(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![0.0, 4.0]);
}

#[test]
fn index_select_f16() {
    let embedding: Vec<_> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f16");
    assert_eq!(
        approx_f16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u8_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_dim1() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 7.0, 8.0f32, 7.0, 9.0, 10.0, 9.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        true,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&dst_buffer, dst_el)
}

fn run_index_select_strided<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        false,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&dst_buffer, dst_el)
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let results = run(&v, unary::contiguous::cos::HALF);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

fn run_reduce<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);

    let options = MTLResourceOptions::StorageModeManaged;
    let output = device.new_buffer((out_length * core::mem::size_of::<U>()) as u64, options);
    let shape = vec![in_length];
    match call_reduce_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        &shape,
        out_length,
        BufferOffset::zero_offset(&input),
        &output,
    ) {
        Ok(_) => {}
        Err(e) => {
            println!("{e}");
            panic!();
        }
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    call_last_softmax(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        last_dim,
        &input,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, v.len())
}

const fn create_array<const N: usize>() -> [f32; N] {
    let mut array: [f32; N] = [0.0; N];
    let mut i = 1;
    while i <= N {
        array[i - 1] = i as f32;
        i += 1;
    }
    array
}

const fn correct_sum<const N: usize, const D: usize>() -> [f32; D] {
    let mut sum = 0;
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        sum += i;
        i += 1;
        if i > j * N / D {
            results[j - 1] = sum as f32;
            j += 1;
            sum = 0;
        }
    }
    results
}

const fn correct_max<const N: usize, const D: usize>() -> [f32; D] {
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        i += 1;
        if i > j * (N / D) {
            results[j - 1] = (i - 1) as f32;
            j += 1;
        }
    }
    results
}

fn correct_argmax<const N: usize, const D: usize>(arr: [f32; N]) -> [u32; D] {
    let mut max = 0.0;
    let mut max_index: u32 = 0;
    let mut results: [u32; D] = [0; D];
    let mut i = 0;
    let mut j = 1;
    while i <= N {
        if i >= (j * N / D) {
            results[j - 1] = max_index;
            max = 0.0;
            max_index = 0;
            j += 1;
        }
        if i == N {
            break;
        }
        if arr[i] > max {
            max = arr[i];
            max_index = i as u32;
        }
        i += 1;
    }
    results
}

fn reduce_sum_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut thread_rng());
    }
    let results = run_reduce(&v, N, D, "fast_sum_f32");
    assert_eq!(approx(results, 4), correct_sum::<N, D>());
}

fn reduce_max_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut thread_rng());
    }
    let results = run_reduce(&v, N, D, "fast_max_f32");
    assert_eq!(approx(results, 4), correct_max::<N, D>());
}

fn reduce_argmax_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut thread_rng());
    }
    let results: Vec<u32> = run_reduce(&v, N, D, "fast_argmax_f32");
    assert_eq!(results, correct_argmax::<N, D>(v));
}

#[test]
fn reduce_sum1() {
    reduce_sum_case::<9, 1>();
    reduce_sum_case::<6, 1>();
    reduce_sum_case::<10, 1>();
    reduce_sum_case::<64, 1>();
    reduce_sum_case::<128, 1>();
    reduce_sum_case::<256, 1>();
    reduce_sum_case::<512, 1>();
    reduce_sum_case::<1024, 1>();
    reduce_sum_case::<2048, 1>();
    reduce_sum_case::<4096, 1>();
}

#[test]
fn reduce_sum2() {
    reduce_sum_case::<6, 2>();
    reduce_sum_case::<10, 2>();
    reduce_sum_case::<64, 2>();
    reduce_sum_case::<128, 2>();
    reduce_sum_case::<256, 2>();
    reduce_sum_case::<512, 2>();
    reduce_sum_case::<1024, 2>();
    reduce_sum_case::<2048, 2>();
    reduce_sum_case::<4096, 2>();
}

#[test]
fn reduce_max() {
    reduce_max_case::<6, 1>();
    reduce_max_case::<9, 1>();
    reduce_max_case::<10, 1>();
    reduce_max_case::<64, 1>();
    reduce_max_case::<128, 1>();
    reduce_max_case::<256, 1>();
    reduce_max_case::<512, 1>();
    reduce_max_case::<1024, 1>();
    reduce_max_case::<2048, 1>();
    reduce_max_case::<4096, 1>();

    reduce_max_case::<6, 2>();
    reduce_max_case::<10, 2>();
    reduce_max_case::<64, 2>();
    reduce_max_case::<128, 2>();
    reduce_max_case::<256, 2>();
    reduce_max_case::<512, 2>();
    reduce_max_case::<1024, 2>();
    reduce_max_case::<2048, 2>();
    reduce_max_case::<4096, 2>();

    reduce_max_case::<6, 3>();
    reduce_max_case::<10, 3>();
    reduce_max_case::<64, 3>();
    reduce_max_case::<128, 3>();
    reduce_max_case::<256, 3>();
    reduce_max_case::<512, 3>();
    reduce_max_case::<1024, 3>();
    reduce_max_case::<2048, 3>();
    reduce_max_case::<4096, 3>();
}

#[test]
fn reduce_argmax() {
    reduce_argmax_case::<6, 1>();
    reduce_argmax_case::<9, 1>();
    reduce_argmax_case::<10, 1>();
    reduce_argmax_case::<64, 1>();
    reduce_argmax_case::<128, 1>();
    reduce_argmax_case::<256, 1>();
    reduce_argmax_case::<512, 1>();
    reduce_argmax_case::<1024, 1>();
    reduce_argmax_case::<2048, 1>();
}

#[test]
fn reduce_argmax2() {
    reduce_argmax_case::<6, 2>();
    reduce_argmax_case::<10, 2>();
    reduce_argmax_case::<64, 2>();
    reduce_argmax_case::<128, 2>();
    reduce_argmax_case::<256, 2>();
    reduce_argmax_case::<512, 2>();
    reduce_argmax_case::<1024, 2>();
    reduce_argmax_case::<2048, 2>();
    reduce_argmax_case::<4096, 2>();
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let last_dim = 4096;
    let n = 200;
    let mut v = vec![0.0; n * last_dim];
    for i in 0..n {
        v[i * last_dim] = 20.0;
    }
    let results = run_softmax(&v, last_dim, "softmax_f32");
    let results = approx(results, 4);
    assert_eq!(
        results.iter().map(|&s| s.round() as usize).sum::<usize>(),
        n
    );
    assert_eq!(results[0], 1.0);
    assert_eq!(results[1], 0.0);
    assert_eq!(results[last_dim], 1.0);
    assert_eq!(results[2 * last_dim], 1.0);

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f16");
    assert_eq!(
        approx_f16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2332, 0.6338]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_bf16");
    assert_eq!(
        approx_bf16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0859, 0.2324, 0.6328]
    );
}

#[allow(clippy::too_many_arguments)]
fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;

    let length = cond.len();
    let cond = device.new_buffer_with_data(
        cond.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(cond) as u64,
        options,
    );
    let left = device.new_buffer_with_data(
        left_true.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );
    let right = device.new_buffer_with_data(
        right_false.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );

    let output = device.new_buffer((length * core::mem::size_of::<T>()) as u64, options);
    let cond = BufferOffset {
        buffer: &cond,
        offset_in_bytes: cond_offset,
    };
    let left = BufferOffset {
        buffer: &left,
        offset_in_bytes: left_offset,
    };
    let right = BufferOffset {
        buffer: &right,
        offset_in_bytes: cond_offset,
    };
    call_where_cond_strided(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        cond,
        &cond_stride,
        left,
        &left_stride,
        right,
        &cond_stride,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}
#[test]
fn where_cond_u32_f32() {
    let shape = vec![6];
    let cond = vec![0u32, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u32_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}

#[allow(clippy::too_many_arguments)]
fn run_mlx_gemm<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    lhs_offset: usize,
    rhs: &[T],
    rhs_stride: &[usize],
    rhs_offset: usize,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;

    let lhs = device.new_buffer_with_data(
        lhs.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(lhs) as u64,
        options,
    );
    let rhs = device.new_buffer_with_data(
        rhs.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(rhs) as u64,
        options,
    );
    let length = b * m * n;
    let output = device.new_buffer((length * core::mem::size_of::<T>()) as u64, options);
    call_mlx_gemm(
        &device,
        command_buffer,
        &kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        &lhs,
        rhs_stride,
        rhs_offset,
        &rhs,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn mlx_gemm() {
    let (b, m, n, k) = (1, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
    );

    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![
            20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0, 488.0,
            518.0, 548.0, 578.0
        ]
    );

    // OFFSET
    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    // Manually set batch_size=1 and offset 12 elements * 4 the number of bytes for f32
    let results = run_mlx_gemm(
        GemmDType::F32,
        (1, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        12 * 4,
    );
    assert_eq!(
        approx(results, 4),
        vec![56.0, 59.0, 62.0, 65.0, 200.0, 212.0, 224.0, 236.0]
    );

    // bgemm sanity test
    {
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<bf16> = (0..b * m * k).map(|f| bf16::from_f32(f as f32)).collect();
        let rhs: Vec<bf16> = (0..b * n * k).map(|f| bf16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::BF16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_bf16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }

    {
        // hgemm sanity test
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::F16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_f16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }
}

fn run_random<T: Clone>(name: &'static str, seed: u32, length: usize, a: f32, b: f32) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let options = MTLResourceOptions::StorageModeManaged;
    let output = device.new_buffer((length * core::mem::size_of::<T>()) as NSUInteger, options);

    let seed = device.new_buffer_with_data(
        &seed as *const u32 as *const core::ffi::c_void,
        std::mem::size_of::<u32>() as NSUInteger,
        options,
    );

    if name.starts_with("rand_uniform") {
        call_random_uniform(
            &device,
            command_buffer,
            &kernels,
            name,
            a,
            b,
            length,
            &seed,
            &output,
        )
        .unwrap();
    } else {
        call_random_normal(
            &device,
            command_buffer,
            &kernels,
            name,
            a,
            b,
            length,
            &seed,
            &output,
        )
        .unwrap();
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn random() {
    fn calc_mean(data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>();
        let count = data.len();
        assert!(count > 0);
        sum / count as f32
    }

    fn calc_stddev(data: &[f32]) -> f32 {
        let mean = calc_mean(data);
        let count = data.len();
        assert!(count > 0);

        let variance = data
            .iter()
            .map(|value| {
                let diff = mean - *value;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;

        variance.sqrt()
    }

    let shape = [1024, 10];

    let length = shape.iter().product::<usize>();
    let seed = 299792458;

    let min = -30.0;
    let max = 30.0;
    let mean = 100.0;
    let stddev = 50.0;

    macro_rules! validate_random {
        ($type:ty) => {
            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_uniform_", stringify!($type)),
                seed,
                length,
                min,
                max,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            results.iter().for_each(|v| {
                assert!(*v >= min && *v <= max);
            });
            assert!(calc_mean(&results) > -1.0 && calc_mean(&results) < 1.0);

            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_normal_", stringify!($type)),
                seed,
                length,
                mean,
                stddev,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            assert!((calc_mean(&results) - mean).abs() < mean / 10.0);
            assert!((calc_stddev(&results) - stddev).abs() < stddev / 10.0);
        };
    }

    validate_random!(f32);
    validate_random!(f16);
    validate_random!(bf16);
}

fn run_scatter_add<T: Clone, I: Clone + std::fmt::Debug>(
    input: &[T],
    ids: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;
    let input_buffer = new_buffer(&device, input);
    let ids_buffer = new_buffer(&device, ids);
    let output = device.new_buffer(std::mem::size_of_val(input) as u64, options);
    call_scatter_add(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, input.len())
}

#[test]
fn scatter_add() {
    let ids_u8 = [0u8, 0, 1, 0, 2, 2, 3, 3];
    let ids_u32 = [0u32, 0, 1, 0, 2, 2, 3, 3];
    let ids_i64 = [0i64, 0, 1, 0, 2, 2, 3, 3];

    let input_f32 = [5.0f32, 1.0, 7.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let input_f16 = input_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let input_bf16 = input_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim1_f32 = vec![8.0, 7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let output_dim1_f16 = output_dim1_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim1_bf16 = output_dim1_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim2_f32 = vec![5.0, 3.0, 7.0, 0.0, 3.0, 2.0, 1.0, 3.0];
    let output_dim2_f16 = output_dim2_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim2_bf16 = output_dim2_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    for (shape, output_f32, output_f16, output_bf16) in [
        (vec![8], output_dim1_f32, output_dim1_f16, output_dim1_bf16),
        (
            vec![4, 2],
            output_dim2_f32,
            output_dim2_f16,
            output_dim2_bf16,
        ),
    ] {
        for results in [
            run_scatter_add(&input_f32, &ids_u8, &shape, 0, "sa_u8_f32"),
            run_scatter_add(&input_f32, &ids_u32, &shape, 0, "sa_u32_f32"),
            run_scatter_add(&input_f32, &ids_i64, &shape, 0, "sa_i64_f32"),
        ] {
            assert_eq!(results, output_f32);
        }
        for results in [
            run_scatter_add(&input_f16, &ids_u8, &shape, 0, "sa_u8_f16"),
            run_scatter_add(&input_f16, &ids_u32, &shape, 0, "sa_u32_f16"),
            run_scatter_add(&input_f16, &ids_i64, &shape, 0, "sa_i64_f16"),
        ] {
            assert_eq!(results, output_f16);
        }
        for results in [
            run_scatter_add(&input_bf16, &ids_u8, &shape, 0, "sa_u8_bf16"),
            run_scatter_add(&input_bf16, &ids_u32, &shape, 0, "sa_u32_bf16"),
            run_scatter_add(&input_bf16, &ids_i64, &shape, 0, "sa_i64_bf16"),
        ] {
            assert_eq!(results, output_bf16);
        }
    }
}

fn run_index_add<T: Clone, I: Clone + std::fmt::Debug>(
    left: &[T],
    right: &[T],
    indices: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input_buffer = new_buffer(&device, right);
    let output = new_buffer(&device, left);
    let indices_buffer = new_buffer(&device, indices);
    call_index_add(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&indices_buffer),
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, left.len())
}

#[test]
fn index_add() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let indices = vec![0u32, 1, 0, 1, 0, 1];
    let shape = vec![6];

    // u32, f32
    {
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, f16
    {
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, bf16
    {
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f32
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, bf16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f32
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, bf16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

fn run_pool2d<T: Clone>(
    v: &[T],
    (w_k, h_k): (usize, usize),
    (w_stride, h_stride): (usize, usize),
    shape: &[usize],
    strides: &[usize],
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let out_w = (shape[2] - w_k) / w_stride + 1;
    let out_h = (shape[3] - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();
    call_pool2d(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        strides,
        out_w,
        out_h,
        w_k,
        h_k,
        w_stride,
        h_stride,
        &input,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, dst_el)
}

#[test]
fn max_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 7.0, 13.0, 15.0];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f32",
    );
    let expected = vec![
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<f16> = (0..16).map(|v| f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| f16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<bf16> = (0..16).map(|v| bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_bf16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| bf16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u8",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u32",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    input_stride: &[usize],
    kernel: &[T],
    kernel_shape: &[usize],
    kernel_stride: &[usize],
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let c_out = kernel_shape[1];
    let k_size = kernel_shape[2];
    let b_size = input_shape[0];
    let l_in = input_shape[2];
    let l_out = (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1;
    let dst_el = c_out * l_out * b_size;

    let input = new_buffer(&device, input);
    let kernel = new_buffer(&device, kernel);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();

    call_conv_transpose1d(
        &device,
        command_buffer,
        &kernels,
        name,
        dilation,
        stride,
        padding,
        out_padding,
        c_out,
        l_out,
        b_size,
        input_shape,
        input_stride,
        kernel_shape,
        kernel_stride,
        &input,
        0,
        &kernel,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, dst_el)
}

#[test]
fn conv_transpose1d_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel = vec![1.0f32, 2.0, 3.0, 4.0];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f32",
    );

    let expected = vec![1., 4., 10., 20., 25., 24., 16.];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_f16() {
    let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_bf16() {
    let input: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_bf16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u8() {
    let input: Vec<u8> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u8> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u8",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u32() {
    let input: Vec<u32> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u32> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u32",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn const_fill() {
    fn constant_fill<T: Clone>(name: &'static str, len: usize, value: f32) -> Vec<T> {
        let dev = device();
        let kernels = Kernels::new();
        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let buffer = dev.new_buffer(
            (len * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        call_const_fill(&dev, command_buffer, &kernels, name, len, &buffer, value).unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        read_to_vec::<T>(&buffer, len)
    }
    fn test<T: Clone + PartialEq + std::fmt::Debug, F: FnOnce(f32) -> T>(name: &'static str, f: F) {
        let len = rand::thread_rng().gen_range(2..16) * rand::thread_rng().gen_range(4..16);
        let value = rand::thread_rng().gen_range(1. ..19.);
        let v = constant_fill::<T>(name, len, value);
        assert_eq!(v, vec![f(value); len])
    }
    test::<u8, _>("fill_u8", |v| v as u8);
    test::<u32, _>("fill_u32", |v| v as u32);
    test::<i64, _>("fill_i64", |v| v as i64);
    test::<f16, _>("fill_f16", f16::from_f32);
    test::<bf16, _>("fill_bf16", bf16::from_f32);
    test::<f32, _>("fill_f32", |v| v);
}
