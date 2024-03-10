#include <metal_stdlib>
using namespace metal;

template<typename T>
void fill(
    device T *buffer [[buffer(0)]],
    constant T &value,
    constant size_t &numel,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numel) return;
    buffer[gid] = value;
}

#define FILL_OP(T, FN_NAME)                 \
kernel void FN_NAME(                        \
    device T *buffer [[buffer(0)]],         \
    constant T &value,                      \
    constant size_t &numel,                 \
    uint gid [[thread_position_in_grid]]    \
) { fill<T>(buffer, value, numel, gid); }   \

FILL_OP(uint8_t, fill_u8)
FILL_OP(uint32_t, fill_u32)
FILL_OP(half, fill_f16)
FILL_OP(float, fill_f32)

#if __METAL_VERSION__ >= 220
FILL_OP(int64_t, fill_i64)
#endif

#if defined(__HAVE_BFLOAT__)
FILL_OP(bfloat, fill_bf16)
#endif
