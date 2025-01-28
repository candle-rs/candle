#pragma once
#include <metal_stdlib>
using namespace metal;

METAL_FUNC uint nonzero(uint n) {
    return n == 0 ? 1 : n;
}

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr ushort granularity() {
    return nonzero<vec_elements<T>::value>();
}

METAL_FUNC uint next_p2(uint x) {
    return 1 << (32 - clz(x - 1));
}

METAL_FUNC uint prev_p2(uint x) {
    return 1 << (31 - clz(x));
}

constant uint MAX_SHARED_MEM = 32767;

template<typename T>
METAL_FUNC uint max_shared_mem(uint n) {
    return min(n, prev_p2(MAX_SHARED_MEM / sizeof(T)));
}

METAL_FUNC uint get_strided_index(
    uint idx,
    constant const uint &num_dims,
    constant const size_t *dims,
    constant const size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}
