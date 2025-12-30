#pragma once

#include <euler/core/types.hh>
#include <euler/core/compiler.hh>
#include <cstdint>

// Check if xsimd is available
#ifdef EULER_HAS_XSIMD
#if defined(EULER_COMPILER_MSVC)
#pragma warning( push )
#elif defined(EULER_COMPILER_GCC)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wuseless-cast"
# pragma GCC diagnostic ignored "-Wsign-conversion"
# pragma GCC diagnostic ignored "-Wold-style-cast"
# pragma GCC diagnostic ignored "-Wundef"
#elif defined(EULER_COMPILER_CLANG)
# pragma clang diagnostic push
#elif defined(EULER_COMPILER_WASM)
# pragma clang diagnostic push
#endif
#include <xsimd/xsimd.hpp>
#if defined(EULER_COMPILER_MSVC)
#pragma warning( pop )
#elif defined(EULER_COMPILER_GCC)
# pragma GCC diagnostic pop
#elif defined(EULER_COMPILER_CLANG)
# pragma clang diagnostic pop
#elif defined(EULER_COMPILER_WASM)
# pragma clang diagnostic pop
#endif

#endif

namespace euler {

// SIMD traits for different types
template<typename T>
struct simd_traits {
    static constexpr bool has_simd = false;
    static constexpr size_t batch_size = 1;
    using batch_type = T;
    using bool_batch_type = bool;
};

#ifdef EULER_HAS_XSIMD

// Specialization for float with SIMD
template<>
struct simd_traits<float> {
    static constexpr bool has_simd = true;
    using batch_type = xsimd::batch<float>;
    using bool_batch_type = xsimd::batch_bool<float>;
    static constexpr size_t batch_size = batch_type::size;
};

// Specialization for double with SIMD
template<>
struct simd_traits<double> {
    static constexpr bool has_simd = true;
    using batch_type = xsimd::batch<double>;
    using bool_batch_type = xsimd::batch_bool<double>;
    static constexpr size_t batch_size = batch_type::size;
};

#endif // EULER_HAS_XSIMD

// Compile-time alignment constant for SIMD
template<typename T>
struct simd_alignment_v {
    // Use a reasonable default alignment for SIMD types
    #ifdef EULER_HAS_XSIMD
    static constexpr size_t value = simd_traits<T>::has_simd ?
        (sizeof(T) == 4 ? 16 : 32) : alignof(T);  // 16 bytes for float, 32 for double
    #else
    static constexpr size_t value = alignof(T);
    #endif
};

// Runtime alignment helpers
template<typename T>
inline size_t simd_alignment() {
    #ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        // xsimd sometimes reports very large alignments that posix_memalign can't handle
        // We clamp to a reasonable maximum (typically cache line size)
        constexpr size_t max_align = 64;
        const size_t xsimd_align = xsimd::default_arch::alignment();
        return xsimd_align <= max_align ? xsimd_align : max_align;
    } else {
        return alignof(T);
    }
    #else
    return alignof(T);
    #endif
}

// Helper to check if a pointer is aligned
template<typename T>
inline bool is_aligned(const T* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % simd_alignment<T>() == 0;
}

} // namespace euler
