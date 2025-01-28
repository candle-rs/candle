#include <metal_stdlib>
#include <metal_limits>
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

struct Divide {
    template<typename T>
    METAL_FUNC T operator()(T a, T b) { return a / b; }
    METAL_FUNC float  operator()(float  a, float  b) { return fast::divide(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return divide(a, b); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a, bfloat  b) { return static_cast<bfloat>(fast::divide(a, b)); }
    #endif
};

struct Exp {
    template<typename T>
    METAL_FUNC T operator()(T a) { return fast::exp(a); }
    METAL_FUNC float  operator()(float  a) { return fast::exp(a); }
    METAL_FUNC half   operator()(half   a) { return exp(a); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a) { return static_cast<bfloat>(fast::exp(a)); }
    #endif
};


// Keeps track of the index of the value in the reduction operation (argmin, argmax, etc.)
// and the value itself. The index is also used to break ties in the reduction operation.
template <typename T>
struct indexed {
    uint i;
    T val;

    constexpr indexed<T>() threadgroup = default;
};

template <typename T>
struct is_indexed_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_indexed_t = is_indexed_type<T>::value;

template <typename T>
struct is_indexed_type<indexed<T>> {
    static constant constexpr bool value = true;
};

template <typename T>
constexpr constant bool not_indexed_t = !is_indexed_t<T>;

template<typename T>
constexpr METAL_FUNC bool operator<(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
constexpr METAL_FUNC bool operator>(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
struct _numeric_limits_impl<indexed<T>> {
    static constexpr METAL_FUNC indexed<T> lowest() {
        return indexed<T>{ 0, numeric_limits<T>::lowest() };
    }

    static constexpr METAL_FUNC indexed<T> max() {
        return indexed<T>{ 0, numeric_limits<T>::max() };
    }
};

#if __METAL_VERSION__ >= 220
METAL_FUNC int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(simd_shuffle_down(as_type<uint2>(data), delta));
}
#endif


#if defined(__HAVE_BFLOAT__)
// Metal does not have simd_shuffle_down for bfloat16
METAL_FUNC bfloat simd_shuffle_down(bfloat value, ushort delta) {
    return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(value), delta));
}
#endif

template <typename T>
METAL_FUNC indexed<T> simd_shuffle_down(indexed<T> iv, ushort delta) {
    return indexed<T> {
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    };
}

template<typename T>
struct Sum {
    static constexpr METAL_FUNC T init() {
        return 0;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_sum(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a + b;
    }
};

template<typename T>
struct Mul {
    static constexpr METAL_FUNC T init() {
        return 1;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_product(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a * b;
    }
};

template<typename T>
struct Min {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::max();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_min(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a < b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::min(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return min(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return min(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return min(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return min(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::min(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template<typename T>
struct Max {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_max(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a > b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::max(a, b); }
    METAL_FUNC half operator()(half a, half b) { return max(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return max(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return max(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return max(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::max(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template <typename T>
constexpr constant bool is_simd_t = __is_valid_simdgroup_type<T>::value;

template <typename T, typename _E = void>
struct is_valid_simd_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_valid_simd_t = is_valid_simd_type<T>::value;

template <typename T>
struct is_valid_simd_type<T, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <typename T>
struct is_valid_simd_type<indexed<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

#if __METAL_VERSION__ >= 220
template <>
struct is_valid_simd_type<int64_t> {
    static constant constexpr bool value = true;
};
#endif

#if defined(__HAVE_BFLOAT__)
template <>
struct is_valid_simd_type<bfloat> {
    static constant constexpr bool value = true;
};
#endif

template <typename T, typename _E = void>
struct is_simd_op {
    static constant constexpr bool value = false;
};
template <typename T>
struct is_simd_op<Sum<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Mul<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Min<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Max<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Helper struct for applying operators.
// The overloaded operator() function is used to apply an operator to two values.
template<typename OP, typename T>
struct operation;

// Specialization for scalar values.
template<typename OP, typename T>
struct operation {
    OP op;

    METAL_FUNC T operator()(T a, T b) {
        return op(a, b);
    }
};

// Specialization for indexed values.
template<typename OP, typename T>
struct operation<OP, indexed<T>> {
    OP op;

    METAL_FUNC indexed<T> operator()(indexed<T> a, indexed<T> b) {
        return op(a, b);
    }
    METAL_FUNC indexed<T> operator()(indexed<T> a, T b, uint idx) {
        return this->operator()(a, indexed<T>{ idx, b });
    }
};

// Load elements from global memory into shared memory.
// Handles both indexed and non-indexed types by using operate.
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    bool STRIDED = false,
    typename _E = void
>
struct loader;


// Contiguous
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE
>
struct loader<T, R, OP, BLOCKSIZE, false, typename metal::enable_if_t<not_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant ushort &el_per_block,
        device const T *src,
        const uint offset,
        const ushort tid
    ) {
        uint idx = tid + offset;
        const uint stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[i]);
        }
        return value;
    }

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant uint &num_dims,
        constant size_t *dims,
        constant size_t *strides,
        constant ushort &el_per_block,
        device const T *src,
        const uint offset,
        const ushort tid
    ) {
        return this->operator()(value, src_numel, el_per_block, src, offset, tid);
    }
};

// Strided
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE
>
struct loader<T, R, OP, BLOCKSIZE, true, typename metal::enable_if_t<not_indexed_t<R>>> {
    operation<OP, R> operate;


    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant uint &num_dims,
        constant size_t *dims,
        constant size_t *strides,
        constant ushort &el_per_block,
        device const T *src,
        const uint offset,
        const ushort tid
    ) {
        const uint idx = tid + offset;
        const uint stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[get_strided_index(i, num_dims, dims, strides)]);
        }
        return value;
    }
};

// Indexed contiguous
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE
>
struct loader<T, R, OP, BLOCKSIZE, false, typename metal::enable_if_t<is_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant uint &num_dims,
        constant size_t *dims,
        constant size_t *strides,
        constant ushort &el_per_block,
        device const T *src,
        const uint offset,
        const ushort tid
    ) {
        const uint thread_id = tid + offset;
        const uint stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[i], i % dims[num_dims - 1]);
        }
        return value;
    }
};

// Indexed strided
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE
>
struct loader<T, R, OP, BLOCKSIZE, true, typename metal::enable_if_t<is_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        constant uint &src_numel,
        constant uint &num_dims,
        constant size_t *dims,
        constant size_t *strides,
        constant ushort &el_per_block,
        device const T *src,
        const uint offset,
        const ushort tid
    ) {
        const uint thread_id = tid + offset;
        const uint stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (uint i = thread_id; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[get_strided_index(i, num_dims, dims, strides)], i % dims[num_dims - 1]);
        }
        return value;
    }
};

template<
    typename OP,
    ushort BLOCKSIZE,
    typename T,
    typename _E = void
>
struct simdgroup_reducer;

// Specialization for built-in simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    METAL_FUNC T operator()(T value) {
        return OP::simd_op(value);
    }
};

// Specialization for custom (non-built-in) simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    operation<OP, T> op;

    METAL_FUNC T operator()(T value) {
        if (BLOCKSIZE >= 32) value = op(value, simd_shuffle_down(value, 16));
        if (BLOCKSIZE >= 16) value = op(value, simd_shuffle_down(value,  8));
        if (BLOCKSIZE >=  8) value = op(value, simd_shuffle_down(value,  4));
        if (BLOCKSIZE >=  4) value = op(value, simd_shuffle_down(value,  2));
        if (BLOCKSIZE >=  2) value = op(value, simd_shuffle_down(value,  1));
        return value;
    }
};

template<typename T, typename OP, ushort BLOCKSIZE>
struct block_reducer {
    simdgroup_reducer<OP, BLOCKSIZE, T> simd_reduce;
    operation<OP, T> operate;
    threadgroup T *shared;

    block_reducer(threadgroup T shared[BLOCKSIZE]) {
        this->shared = shared;
    }

    METAL_FUNC T operator()(T value, const ushort tid) {
        if (BLOCKSIZE >= 64) {
            // Only store in threadgroup shared memory if needed.
            shared[tid] = value;
            // Threadgroup barrier is needed to ensure that all threads have written to shared memory
            threadgroup_barrier(mem_flags::mem_none);
        }

        #pragma clang loop unroll(full)
        for (ushort s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
            if (tid < s) shared[tid] = operate(shared[tid], shared[tid + s]);
            threadgroup_barrier(mem_flags::mem_none);
        }
        if (tid < 32) {
            // Last shared memory reduce can be done without tid < s check.
            if (BLOCKSIZE >= 64) {
                value = operate(shared[tid], shared[tid + 32]);
                simdgroup_barrier(mem_flags::mem_none);
            }
            // Remaining 32 threads can be reduced with simdgroup_reduce.
            value = simd_reduce(value);
        }
        return value;
    }
};

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    bool STRIDED = false
>
METAL_FUNC void reduce(
    constant uint &src_numel,
    constant uint &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant ushort &el_per_block,
    device const T *src,
    device R *dst,
    threadgroup R shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    loader<T, R, OP, BLOCKSIZE, STRIDED> load;
    block_reducer<T, OP, BLOCKSIZE> reduce(shared);

    // Calcluate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    auto value = load(
        OP::init(),
        src_numel,
        num_dims,
        dims,
        strides,
        el_per_block,
        src,
        offset,
        tid
    );
    // Complete reduction
    R result = reduce(value, tid);

    if (tid == 0) dst[dst_id] = result;
}

#define reduce_case(OP, T, R, N)                        \
case N: {                                               \
    threadgroup R shared[N];                            \
    reduce<T, R, OP<R>, N, STRIDED>(                    \
        src_numel,                                      \
        num_dims,                                       \
        dims,                                           \
        strides,                                        \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define ARG(...) __VA_ARGS__

#define impl_reduce_inner(OP, NAME, T)                  \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant ushort &el_per_block,                      \
    device const T *src,                                \
    device T *dst,                                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    constant size_t *strides = {};                      \
    const bool STRIDED = false;                         \
    switch (max_shared_mem<T>(block_dim)) {             \
        reduce_case(OP, ARG(T), ARG(T), 2048);          \
        reduce_case(OP, ARG(T), ARG(T), 1024);          \
        reduce_case(OP, ARG(T), ARG(T),  512);          \
        reduce_case(OP, ARG(T), ARG(T),  256);          \
        reduce_case(OP, ARG(T), ARG(T),  128);          \
        reduce_case(OP, ARG(T), ARG(T),   64);          \
        reduce_case(OP, ARG(T), ARG(T),   32);          \
        reduce_case(OP, ARG(T), ARG(T),   16);          \
        reduce_case(OP, ARG(T), ARG(T),    8);          \
        reduce_case(OP, ARG(T), ARG(T),    4);          \
        reduce_case(OP, ARG(T), ARG(T),    2);          \
        reduce_case(OP, ARG(T), ARG(T),    1);          \
    }                                                   \
}


#define impl_reduce_strided(OP, NAME, T)                \
kernel void NAME##_strided(                             \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant ushort &el_per_block,                      \
    device const T *src,                                \
    device T *dst,                                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    const bool STRIDED = true;                          \
    switch (max_shared_mem<T>(block_dim)) {             \
        reduce_case(OP, ARG(T), ARG(T), 2048);          \
        reduce_case(OP, ARG(T), ARG(T), 1024);          \
        reduce_case(OP, ARG(T), ARG(T),  512);          \
        reduce_case(OP, ARG(T), ARG(T),  256);          \
        reduce_case(OP, ARG(T), ARG(T),  128);          \
        reduce_case(OP, ARG(T), ARG(T),   64);          \
        reduce_case(OP, ARG(T), ARG(T),   32);          \
        reduce_case(OP, ARG(T), ARG(T),   16);          \
        reduce_case(OP, ARG(T), ARG(T),    8);          \
        reduce_case(OP, ARG(T), ARG(T),    4);          \
        reduce_case(OP, ARG(T), ARG(T),    2);          \
        reduce_case(OP, ARG(T), ARG(T),    1);          \
    }                                                   \
}

#define impl_reduce(OP, NAME, T)                        \
impl_reduce_inner(OP, NAME, T)                          \
impl_reduce_strided(OP, NAME, T)                        \

template<
    typename T,
    typename ReductionOp,
    ushort BLOCKSIZE,
    bool STRIDED = false
>
METAL_FUNC void reduce(
    constant uint &src_numel,
    constant uint &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant ushort &el_per_block,
    device const T *src,
    device uint *dst,
    threadgroup indexed<T> shared[BLOCKSIZE],
    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    using I = indexed<T>;
    loader<T, indexed<T>, ReductionOp, BLOCKSIZE, STRIDED> load;
    block_reducer<I, ReductionOp, BLOCKSIZE> reduce(shared);

    // Calcluate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    indexed<T> value = load(
        ReductionOp::init(),
        src_numel,
        num_dims,
        dims,
        strides,
        el_per_block,
        src,
        offset,
        tid
    );

    // Complete reduction
    I result = reduce(value, tid);

    // Return index of reduce result
    if (tid == 0) dst[dst_id] = result.i;
}

#define arg_reduce_case(OP, T, N)                       \
case N: {                                               \
    using I = indexed<T>;                               \
    threadgroup I shared[N];                            \
    reduce<T, OP<I>, N, STRIDED>(                       \
        src_numel,                                      \
        num_dims,                                       \
        dims,                                           \
        strides,                                        \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_arg_reduce_inner(OP, NAME, T)              \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant ushort &el_per_block,                      \
    device const T *src,                                \
    device uint *dst,                                   \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    constant size_t *strides = {};                      \
    const bool STRIDED = false;                         \
    switch (max_shared_mem<indexed<T>>(block_dim)) {    \
        arg_reduce_case(OP, ARG(T), 1024);              \
        arg_reduce_case(OP, ARG(T), 512);               \
        arg_reduce_case(OP, ARG(T), 256);               \
        arg_reduce_case(OP, ARG(T), 128);               \
        arg_reduce_case(OP, ARG(T), 64);                \
        arg_reduce_case(OP, ARG(T), 32);                \
        arg_reduce_case(OP, ARG(T), 16);                \
        arg_reduce_case(OP, ARG(T), 8);                 \
        arg_reduce_case(OP, ARG(T), 4);                 \
        arg_reduce_case(OP, ARG(T), 2);                 \
        arg_reduce_case(OP, ARG(T), 1);                 \
    }                                                   \
}                                                       \


#define impl_arg_reduce_strided(OP, NAME, T)            \
kernel void NAME##_strided(                             \
    constant uint &src_numel,                           \
    constant uint &num_dims,                            \
    constant size_t *dims,                              \
    constant size_t *strides,                           \
    constant ushort &el_per_block,                      \
    device const T *src,                                \
    device uint *dst,                                   \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    const bool STRIDED = true;                          \
    const bool INDEXED = true;                          \
    switch (max_shared_mem<indexed<T>>(block_dim)) {    \
        arg_reduce_case(OP, ARG(T), 1024);              \
        arg_reduce_case(OP, ARG(T), 512);               \
        arg_reduce_case(OP, ARG(T), 256);               \
        arg_reduce_case(OP, ARG(T), 128);               \
        arg_reduce_case(OP, ARG(T), 64);                \
        arg_reduce_case(OP, ARG(T), 32);                \
        arg_reduce_case(OP, ARG(T), 16);                \
        arg_reduce_case(OP, ARG(T), 8);                 \
        arg_reduce_case(OP, ARG(T), 4);                 \
        arg_reduce_case(OP, ARG(T), 2);                 \
        arg_reduce_case(OP, ARG(T), 1);                 \
    }                                                   \
}


#define impl_arg_reduce(OP, NAME, T)                    \
impl_arg_reduce_inner(OP, NAME, T)                      \
impl_arg_reduce_strided(OP, NAME, T)                    \

// Contains the intermediate results for the online softmax calculation.
// m: max
// d: sum of the exponentials
template <typename T>
struct MD {
    T m;
    float d;

    constexpr MD<T>() = default;
    constexpr MD<T>() threadgroup = default;
};

// Enable operations for softmax MD
template<typename OP, typename T>
struct operation<OP, MD<T>> {
    OP op;

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        return op(a, b);
    }

    METAL_FUNC MD<T> operator()(MD<T> a, T b) {
        return this->operator()(a, MD<T>{ b, static_cast<T>(1.0) });
    }
};

template <typename T>
METAL_FUNC MD<T> simd_shuffle_down(MD<T> md, ushort delta) {
    return MD<T> {
        simd_shuffle_down(md.m, delta),
        simd_shuffle_down(md.d, delta)
    };
}

// Enable simd_shuffle_down for softmax MD
template <typename T>
struct is_valid_simd_type<MD<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template<typename T>
struct MDReduceOp {
    Exp fast_exp;

    static constexpr METAL_FUNC MD<T> init() {
        return MD<T>{ numeric_limits<T>::lowest(), 0 };
    }

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d * fast_exp(smaller_m.m - bigger_m.m);
        res.m = bigger_m.m;
        return res;
    }
};


template<typename T, ushort BLOCKSIZE>
struct finalize_softmax {
    Divide fast_divide;
    Exp fast_exp;

    METAL_FUNC void operator()(
        device const T *src,
        device T *dst,
        threadgroup MD<T> &md_total,
        const uint thread_id,
        const uint stop_idx
    ) {
        const float d_total_inverse = fast_divide(1.0, md_total.d);
        for (uint idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
            dst[idx] = static_cast<T>(fast_exp(src[idx] - md_total.m) * d_total_inverse);
        }
    }
};

// Welford's algorithm approach for an online softmax implementation.
// Same as the Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867.pdf
template<typename T, ushort BLOCKSIZE>
METAL_FUNC void softmax(
    constant uint &src_numel,
    constant ushort &el_per_block,
    device const T *src,
    device T *dst,
    threadgroup MD<T> shared[BLOCKSIZE],
    threadgroup MD<T> &md_total,

    ushort tid [[ thread_index_in_threadgroup ]],
    ushort dst_id [[ threadgroup_position_in_grid ]]
) {
    using MDReduceOp = MDReduceOp<T>;

    loader<T, MD<T>, MDReduceOp, BLOCKSIZE> load;
    block_reducer<MD<T>, MDReduceOp, BLOCKSIZE> reduce(shared);
    finalize_softmax<T, BLOCKSIZE> softmax_finalize;

    // Calcluate offset for the threadgroup of current thread;
    const uint offset = dst_id * el_per_block;

    // Calculate partial result for current thread
    MD<T> md_partial = MD<T> { numeric_limits<T>::lowest(), 0 };
    md_partial = load(
        md_partial,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );

    // Reduce in shared memory
    MD<T> md = reduce(md_partial, tid);

    if (tid == 0) md_total = md;
    threadgroup_barrier(mem_flags::mem_none);

    // Finalize softmax
    const uint thread_id = tid + offset;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    softmax_finalize(src, dst, md_total, thread_id, stop_idx);
}

#define softmax_case(T, N)                              \
case N: {                                               \
    threadgroup MD<T> shared[N];                        \
    threadgroup MD<T> md_total;                         \
    softmax<T, N>(                                      \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        md_total,                                       \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

#define impl_softmax_inner(NAME, T)                     \
kernel void NAME(                                       \
    constant uint &src_numel,                           \
    constant ushort &el_per_block,                      \
    device const T *src,                                \
    device T *dst,                                      \
    ushort tid [[ thread_index_in_threadgroup ]],       \
    ushort dst_id [[ threadgroup_position_in_grid ]],   \
    ushort block_dim [[ threads_per_threadgroup ]]      \
) {                                                     \
    switch (max_shared_mem<T>(block_dim)) {             \
        softmax_case(T, 1024);                          \
        softmax_case(T,  512);                          \
        softmax_case(T,  256);                          \
        softmax_case(T,  128);                          \
        softmax_case(T,   64);                          \
        softmax_case(T,   32);                          \
        softmax_case(T,   16);                          \
        softmax_case(T,    8);                          \
        softmax_case(T,    4);                          \
        softmax_case(T,    2);                          \
        softmax_case(T,    1);                          \
    }                                                   \
}

#define impl_softmax(NAME, T)                           \
impl_softmax_inner(NAME, T)                             \


template<typename T>
METAL_FUNC void rmsnorm(
    constant size_t & src_numel,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    device const T * alpha,
    constant float & eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp = 0;
    while (idx < stop_idx) {
        tmp = tmp + float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(shared_memory[0] / float(el_to_sum_per_block) + eps);
    float inv_norm = 1.0f / norm;
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = float(src[idx]) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

template<typename T>
METAL_FUNC void layernorm(
    constant size_t & src_numel,
    constant size_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    device const T * alpha,
    device const T * beta,
    constant float & eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp1 = 0;
    float tmp2 = 0;
    while (idx < stop_idx) {
        tmp1 += float(src[idx]);
        tmp2 += float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp1;
    shared_memory[tid + block_dim] = tmp2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
            shared_memory[block_dim + tid] = shared_memory[block_dim + tid] + shared_memory[block_dim + tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_memory[0] / float(el_to_sum_per_block);
    float var = shared_memory[block_dim] / float(el_to_sum_per_block) - mean * mean;
    float inv_norm = 1.0f / sqrt(var + eps);
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = (float(src[idx]) - mean) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        if (beta != nullptr) {
            val += float(beta[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

constant int THREADGROUP_SIZE = 2048;

#define RMSNORM(NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    device const T *alpha, \
    constant float &eps, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup float shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = 0; \
    rmsnorm<T>(src_numel, el_to_sum_per_block, src, dst, alpha, eps, id, tid, dst_id, block_dim, shared_memory); \
} \

#define LAYERNORM(NAME, T) \
kernel void NAME( \
    constant size_t &src_numel, \
    constant size_t &el_to_sum_per_block, \
    device const T *src, \
    device T *dst, \
    device const T *alpha, \
    device const T *beta, \
    constant float &eps, \
    uint id [[ thread_position_in_grid ]], \
    uint tid [[ thread_index_in_threadgroup ]], \
    uint dst_id [[ threadgroup_position_in_grid ]], \
    uint block_dim [[ threads_per_threadgroup ]] \
) { \
    threadgroup float shared_memory[THREADGROUP_SIZE]; \
    shared_memory[tid] = 0; \
    layernorm<T>(src_numel, el_to_sum_per_block, src, dst, alpha, beta, eps, id, tid, dst_id, block_dim, shared_memory); \
} \

template<typename T>
METAL_FUNC void ropei(
    constant size_t &bh,
    constant size_t &td,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid
) {
    if (2 * tid >= bh * td) {
        return;
    }
    size_t rope_idx = tid % (td / 2);
    T c = cos[rope_idx];
    T s = sin[rope_idx];
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s;
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c;
}

template<typename T>
METAL_FUNC void rope(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= bh * td) {
        return;
    }
    size_t i_bh = idx / (td / 2);
    size_t i_td = idx - (td / 2) * i_bh;
    size_t i_t = i_td / (d / 2);
    size_t i_d = i_td - (d / 2) * i_t;
    size_t i1 = i_bh * td + i_t * d + i_d;
    size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
METAL_FUNC void rope_thd(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= b * t * h * d) {
        return;
    }
    const size_t i_bth = idx / (d / 2);
    const size_t i_d = idx - (d / 2) * i_bth;
    const size_t i_t = (i_bth / h) % t;
    const size_t i1 = i_bth * d + i_d;
    const size_t i2 = i1 + d / 2;
    const size_t i_cs = i_t * (d / 2) + i_d;
     T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

#define ROPE(FN_NAME, FN_NAME_I, FN_NAME_THD, TYPENAME) \
kernel void FN_NAME_I( \
    constant size_t &bh, \
    constant size_t &td, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    ropei<TYPENAME>(bh, td, src, cos, sin, dst, tid); \
}\
kernel void FN_NAME( \
    constant size_t &bh, \
    constant size_t &td, \
    constant size_t &d, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope<TYPENAME>(bh, td, d, src, cos, sin, dst, idx); \
}\
kernel void FN_NAME_THD( \
    constant size_t &b, \
    constant size_t &t, \
    constant size_t &h, \
    constant size_t &d, \
    device const TYPENAME *src,  \
    device const TYPENAME *cos,  \
    device const TYPENAME *sin,  \
    device TYPENAME *dst, \
    uint idx [[ thread_position_in_grid ]] \
) { \
    rope_thd<TYPENAME>(b, t, h, d, src, cos, sin, dst, idx); \
}\

RMSNORM(rmsnorm_f32, float)
RMSNORM(rmsnorm_f16, half)
LAYERNORM(layernorm_f32, float)
LAYERNORM(layernorm_f16, half)
ROPE(rope_f32, rope_i_f32, rope_thd_f32, float)
ROPE(rope_f16, rope_i_f16, rope_thd_f16, half)

impl_reduce(Sum, fast_sum_f32, float)
impl_reduce(Sum, fast_sum_u32, uint)
impl_reduce(Sum, fast_sum_f16, half)
impl_reduce(Sum, fast_sum_u8, uint8_t)

impl_reduce(Mul, fast_mul_f32, float)
impl_reduce(Mul, fast_mul_u32, uint)
impl_reduce(Mul, fast_mul_f16, half)
impl_reduce(Mul, fast_mul_u8, uint8_t)

impl_reduce(Max, fast_max_f32, float)
impl_reduce(Max, fast_max_u32, uint)
impl_reduce(Max, fast_max_f16, half)
impl_reduce(Max, fast_max_u8, uint8_t)

impl_reduce(Min, fast_min_f32, float)
impl_reduce(Min, fast_min_u32, uint)
impl_reduce(Min, fast_min_f16, half)
impl_reduce(Min, fast_min_u8, uint8_t)

impl_arg_reduce(Min, fast_argmin_f32, float)
impl_arg_reduce(Min, fast_argmin_f16, half)
impl_arg_reduce(Min, fast_argmin_u32, uint)
impl_arg_reduce(Min, fast_argmin_u8, uint8_t)

impl_arg_reduce(Max, fast_argmax_f32, float)
impl_arg_reduce(Max, fast_argmax_f16, half)
impl_arg_reduce(Max, fast_argmax_u32, uint)
impl_arg_reduce(Max, fast_argmax_u8, uint8_t)

impl_softmax(softmax_f32, float)
impl_softmax(softmax_f16, half)

#if __METAL_VERSION__ >= 220
impl_reduce(Sum, fast_sum_i64, int64_t)
impl_reduce(Mul, fast_mul_i64, int64_t)
impl_reduce(Min, fast_min_i64, int64_t)
impl_reduce(Max, fast_max_i64, int64_t)

impl_arg_reduce(Min, fast_argmin_i64, int64_t)
impl_arg_reduce(Max, fast_argmax_i64, int64_t)
#endif

#if defined(__HAVE_BFLOAT__)
impl_reduce(Sum, fast_sum_bf16, bfloat)
impl_reduce(Mul, fast_mul_bf16, bfloat)
impl_reduce(Max, fast_max_bf16, bfloat)
impl_reduce(Min, fast_min_bf16, bfloat)

impl_arg_reduce(Min, fast_argmin_bf16, bfloat)
impl_arg_reduce(Max, fast_argmax_bf16, bfloat)

impl_softmax(softmax_bf16, bfloat)

RMSNORM(rmsnorm_bf16, bfloat)
LAYERNORM(layernorm_bf16, bfloat)
ROPE(rope_bf16, rope_i_bf16, rope_thd_bf16, bfloat)
#endif
