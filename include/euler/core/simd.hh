#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/core/compiler.hh>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdlib>
#ifdef _MSC_VER
#include <malloc.h>  // For _aligned_malloc and _aligned_free on Windows
#endif

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

// SIMD operations wrapper
template<typename T>
class simd_ops {
public:
    using batch_type = typename simd_traits<T>::batch_type;
    using bool_batch_type = typename simd_traits<T>::bool_batch_type;
    static constexpr size_t batch_size = simd_traits<T>::batch_size;
    
    // Load operations
    static batch_type load_aligned(const T* ptr) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::load_aligned(ptr);
        } else {
            return *ptr;
        }
        #else
        return *ptr;
        #endif
    }
    
    static batch_type load_unaligned(const T* ptr) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::load_unaligned(ptr);
        } else {
            return *ptr;
        }
        #else
        return *ptr;
        #endif
    }
    
    // Store operations
    static void store_aligned(T* ptr, const batch_type& val) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            xsimd::store_aligned(ptr, val);
        } else {
            *ptr = val;
        }
        #else
        *ptr = val;
        #endif
    }
    
    static void store_unaligned(T* ptr, const batch_type& val) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            xsimd::store_unaligned(ptr, val);
        } else {
            *ptr = val;
        }
        #else
        *ptr = val;
        #endif
    }
    
    // Arithmetic operations
    static batch_type add(const batch_type& a, const batch_type& b) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return a + b;
        } else {
            return a + b;
        }
        #else
        return a + b;
        #endif
    }
    
    static batch_type sub(const batch_type& a, const batch_type& b) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return a - b;
        } else {
            return a - b;
        }
        #else
        return a - b;
        #endif
    }
    
    static batch_type mul(const batch_type& a, const batch_type& b) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return a * b;
        } else {
            return a * b;
        }
        #else
        return a * b;
        #endif
    }
    
    static batch_type div(const batch_type& a, const batch_type& b) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return a / b;
        } else {
            return a / b;
        }
        #else
        return a / b;
        #endif
    }
    
    // Math functions
    static batch_type sqrt(const batch_type& a) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::sqrt(a);
        } else {
            return std::sqrt(a);
        }
        #else
        return std::sqrt(a);
        #endif
    }
    
    static batch_type abs(const batch_type& a) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::abs(a);
        } else {
            return std::abs(a);
        }
        #else
        return std::abs(a);
        #endif
    }
    
    // Reduction operations
    static T reduce_add(const batch_type& a) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::reduce_add(a);
        } else {
            return a;
        }
        #else
        return a;
        #endif
    }
    
    static T reduce_min(const batch_type& a) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::reduce_min(a);
        } else {
            return a;
        }
        #else
        return a;
        #endif
    }
    
    static T reduce_max(const batch_type& a) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            return xsimd::reduce_max(a);
        } else {
            return a;
        }
        #else
        return a;
        #endif
    }
};

// Helper to determine optimal processing strategy
template<typename T>
constexpr bool should_use_simd(size_t size) {
    return simd_traits<T>::has_simd && size >= simd_traits<T>::batch_size * 2;
}

// Aligned memory allocation helpers
template<typename T>
T* aligned_alloc(size_t count) {
    EULER_CHECK_POSITIVE(count, "allocation count");

    const size_t alignment = simd_alignment<T>();

    // posix_memalign requires alignment to be at least sizeof(void*) and a power of 2
    const size_t min_align = sizeof(void*);
    const size_t actual_align = alignment < min_align ? min_align : alignment;

    // Round up size to be a multiple of alignment
    // This is required by C11 aligned_alloc and prevents heap corruption on Windows
    size_t bytes = count * sizeof(T);
    bytes = (bytes + actual_align - 1) & ~(actual_align - 1);

    #ifdef _MSC_VER
    T* result = static_cast<T*>(_aligned_malloc(bytes, actual_align));
    #else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, actual_align, bytes) != 0) {
        ptr = nullptr;
    }
    T* result = static_cast<T*>(ptr);
    #endif

    EULER_CRITICAL_CHECK(result != nullptr, error_code::null_pointer,
                        "Failed to allocate aligned memory");
    return result;
}

template<typename T>
void aligned_free(T* ptr) {
    if (ptr != nullptr) {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
}

} // namespace euler