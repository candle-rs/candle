#include <metal_stdlib>
#include <metal_math>
using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template <typename T> METAL_FUNC T sqr(T in){ return in * in; }
template <typename T> METAL_FUNC T recip(T in){ return T(1.0 / in); }
template <typename T> METAL_FUNC T neg(T in){ return -in; }

template <typename T> METAL_FUNC T erf(T in){
    float x = (float) in;
    // constants
    float a1 =  0.254829592;
    float a2 = -0.284496736;
    float a3 =  1.421413741;
    float a4 = -1.453152027;
    float a5 =  1.061405429;
    float p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return T(sign*y);
}
template <typename T> METAL_FUNC T id(T in) { return in; }
template <typename T> METAL_FUNC T gelu_erf(T x) {
    return T(x * (1 + erf(x * M_SQRT1_2_F)) / 2);
}
template <typename T> METAL_FUNC T gelu(T x) {
    if (x > 5) {
        return x;
    }
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    T beta =  (static_cast<T>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + T(tanh(beta)));
}
template <typename T> METAL_FUNC T relu(T in){
    if (in < 0) {
        return 0;
    }
    return in;
}
template <typename T> METAL_FUNC T silu(T in){
    return in / (static_cast<T>(1) + exp(-in));
}

#define UNARY(FN, TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = TYPENAME(FN(float(input[tid]))); \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = TYPENAME(FN(float(input[get_strided_index(tid, num_dims, dims, strides)]))); \
}

#define UNARY_OP(NAME) \
UNARY(NAME, float, NAME##_f32, NAME##_f32_strided); \
UNARY(NAME, half, NAME##_f16, NAME##_f16_strided);

#define BFLOAT_UNARY_OP(NAME) \
UNARY(NAME, bfloat, NAME##_bf16, NAME##_bf16_strided);


UNARY_OP(cos)
UNARY_OP(sin)
UNARY_OP(sqr)
UNARY_OP(sqrt)
UNARY_OP(neg)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(gelu)
UNARY_OP(silu)
UNARY_OP(abs)
UNARY_OP(ceil)
UNARY_OP(floor)
UNARY_OP(round)
UNARY_OP(gelu_erf)
UNARY_OP(erf)
UNARY_OP(tanh)
UNARY_OP(recip)
UNARY_OP(relu)
UNARY(id, float, copy_f32, copy_f32_strided)
UNARY(id, half, copy_f16, copy_f16_strided)
UNARY(id, uint8_t, copy_u8, copy_u8_strided)
UNARY(id, uint32_t, copy_u32, copy_u32_strided)

#if __METAL_VERSION__ >= 220
UNARY(id, int64_t, copy_i64, copy_i64_strided)
#endif

#if defined(__HAVE_BFLOAT__)
BFLOAT_UNARY_OP(cos)
BFLOAT_UNARY_OP(sin)
BFLOAT_UNARY_OP(sqr)
BFLOAT_UNARY_OP(sqrt)
BFLOAT_UNARY_OP(neg)
BFLOAT_UNARY_OP(exp)
BFLOAT_UNARY_OP(log)
BFLOAT_UNARY_OP(gelu)
BFLOAT_UNARY_OP(silu)
BFLOAT_UNARY_OP(abs)
BFLOAT_UNARY_OP(ceil)
BFLOAT_UNARY_OP(floor)
BFLOAT_UNARY_OP(round)
BFLOAT_UNARY_OP(gelu_erf)
BFLOAT_UNARY_OP(erf)
BFLOAT_UNARY_OP(tanh)
BFLOAT_UNARY_OP(recip)
BFLOAT_UNARY_OP(relu)

UNARY(id, bfloat, copy_bf16, copy_bf16_strided)
#endif
