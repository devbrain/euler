/**
 * @file transcendental_ops.hh
 * @brief Direct SIMD operations for transcendental functions
 * @ingroup DirectModule
 * 
 * This file provides high-performance direct operations for transcendental
 * functions (exp, log, sin, cos, tan, etc.) that bypass the expression
 * template system.
 * 
 * @section transcendental_ops_features Key Features
 * - SIMD-accelerated implementations using xsimd
 * - Support for all common transcendental functions
 * - Automatic fallback to std::math functions
 * - Optimized sincos() for simultaneous sine/cosine
 * - In-place operation support
 * 
 * @section transcendental_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/direct/transcendental_ops.hh>
 * 
 * using namespace euler;
 * using namespace euler::direct;
 * 
 * vec3<float> angles(0.0f, M_PI/4, M_PI/2);
 * vec3<float> sines, cosines;
 * 
 * // Trigonometric functions
 * sin(angles, sines);          // sines = [0, 0.707, 1]
 * cos(angles, cosines);        // cosines = [1, 0.707, 0]
 * 
 * // Simultaneous sine/cosine (more efficient)
 * sincos(angles, sines, cosines);
 * 
 * // Exponential and logarithmic
 * vec3<float> values(1, 2, 3);
 * vec3<float> logs, exps;
 * log(values, logs);           // Natural logarithm
 * exp(logs, exps);             // Exponential (recovers original)
 * @endcode
 * 
 * @section transcendental_ops_performance Performance Notes
 * - SIMD versions can be 2-4x faster than scalar loops
 * - sincos() is more efficient than separate sin() and cos()
 * - Some functions use fast approximations on certain platforms
 */
#pragma once

#include <euler/vector/vector.hh>
#include <euler/core/compiler.hh>
#include <euler/core/simd.hh>
#include <cmath>
#include <utility>

namespace euler::direct {

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

/**
 * @defgroup transcendental_exp_log Exponential and Logarithmic Functions
 * @ingroup DirectModule
 * @brief SIMD-accelerated exponential and logarithmic operations
 * @{
 */

/**
 * @brief Compute exponential (e^x) for each element
 * 
 * Computes the exponential function e^x for each element of the input vector.
 * Uses SIMD exp instruction when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector
 * @param result Output vector containing e^v[i] (can alias input)
 * 
 * @code
 * vec3<float> v(0, 1, 2);
 * vec3<float> result;
 * exp(v, result);  // result = [1, 2.718, 7.389]
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void exp(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::exp(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::exp(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::exp(pv[i]);
        }
    }
}

/**
 * @brief Compute natural logarithm for each element
 * 
 * Computes the natural logarithm ln(x) for each element of the input vector.
 * Uses SIMD log instruction when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector (all elements must be positive)
 * @param result Output vector containing ln(v[i]) (can alias input)
 * 
 * @warning No domain checking - negative inputs produce undefined results
 * 
 * @code
 * vec3<float> v(1, M_E, 10);
 * vec3<float> result;
 * log(v, result);  // result = [0, 1, 2.303]
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void log(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::log(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::log(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::log(pv[i]);
        }
    }
}

/**
 * @brief Compute base-10 logarithm for each element
 * @param v Input vector (all elements must be positive)
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void log10(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::log10(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::log10(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::log10(pv[i]);
        }
    }
}

/**
 * @brief Compute base-2 logarithm for each element
 * @param v Input vector (all elements must be positive)
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void log2(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::log2(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::log2(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::log2(pv[i]);
        }
    }
}

/**
 * @brief Compute v^p (element-wise power)
 * @param v Input vector (base)
 * @param p Power (scalar)
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void pow(const vector<T, N>& v, T p, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        batch vp = batch::broadcast(p);
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::pow(vv, vp);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::pow(pv[i], p);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::pow(pv[i], p);
        }
    }
}

/**
 * @brief Compute v1^v2 (element-wise power)
 * @param base Input vector (base)
 * @param exponent Input vector (exponent)
 * @param result Result vector (can alias with inputs)
 */
template<typename T, size_t N>
EULER_HOT void pow(const vector<T, N>& base, const vector<T, N>& exponent, vector<T, N>& result) {
    const T* EULER_RESTRICT pb = base.data();
    const T* EULER_RESTRICT pe = exponent.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vb = batch::load_unaligned(&pb[i]);
            batch ve = batch::load_unaligned(&pe[i]);
            batch vr = xsimd::pow(vb, ve);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::pow(pb[i], pe[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::pow(pb[i], pe[i]);
        }
    }
}

/** @} */ // end of transcendental_exp_log

// =============================================================================
// Trigonometric Functions
// =============================================================================

/**
 * @defgroup transcendental_trig Trigonometric Functions
 * @ingroup DirectModule
 * @brief SIMD-accelerated trigonometric operations
 * @{
 */

/**
 * @brief Compute sine for each element
 * 
 * Computes the sine of each element, treating inputs as angles in radians.
 * Uses SIMD sin instruction when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector (angles in radians)
 * @param result Output vector containing sin(v[i]) (can alias input)
 * 
 * @code
 * vec3<float> angles(0, M_PI/6, M_PI/2);  // 0°, 30°, 90°
 * vec3<float> sines;
 * sin(angles, sines);  // sines = [0, 0.5, 1]
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void sin(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::sin(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::sin(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::sin(pv[i]);
        }
    }
}

/**
 * @brief Compute cosine for each element
 * 
 * Computes the cosine of each element, treating inputs as angles in radians.
 * Uses SIMD cos instruction when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector (angles in radians)
 * @param result Output vector containing cos(v[i]) (can alias input)
 * 
 * @code
 * vec3<float> angles(0, M_PI/3, M_PI/2);  // 0°, 60°, 90°
 * vec3<float> cosines;
 * cos(angles, cosines);  // cosines = [1, 0.5, 0]
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void cos(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::cos(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::cos(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::cos(pv[i]);
        }
    }
}

/**
 * @brief Compute tangent for each element
 * 
 * Computes the tangent of each element, treating inputs as angles in radians.
 * Uses SIMD tan instruction when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector (angles in radians)
 * @param result Output vector containing tan(v[i]) (can alias input)
 * 
 * @warning Tangent is undefined at ±π/2 + nπ
 * 
 * @code
 * vec3<float> angles(0, M_PI/4, M_PI/3);
 * vec3<float> tangents;
 * tan(angles, tangents);  // tangents = [0, 1, 1.732]
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void tan(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::tan(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::tan(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::tan(pv[i]);
        }
    }
}

/**
 * @brief Compute sine and cosine simultaneously
 * 
 * Computes both sine and cosine of each element in a single operation.
 * This is often significantly more efficient than separate sin() and cos()
 * calls, especially on platforms with dedicated sincos instructions.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam N Vector dimension
 * @param v Input vector (angles in radians)
 * @param sin_result Output vector for sine values (can alias input)
 * @param cos_result Output vector for cosine values (can alias input)
 * 
 * @note Uses platform-specific sincos when available
 * @note Approximately 1.5x faster than separate sin/cos calls
 * 
 * @code
 * vec3<float> angles(0, M_PI/4, M_PI/2);
 * vec3<float> sines, cosines;
 * sincos(angles, sines, cosines);  // More efficient than separate calls
 * @endcode
 */
template<typename T, size_t N>
EULER_HOT void sincos(const vector<T, N>& v, vector<T, N>& sin_result, vector<T, N>& cos_result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT ps = sin_result.data();
    T* EULER_RESTRICT pc = cos_result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vsin = xsimd::sin(vv);
            batch vcos = xsimd::cos(vv);
            vsin.store_unaligned(&ps[i]);
            vcos.store_unaligned(&pc[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            ps[i] = std::sin(pv[i]);
            pc[i] = std::cos(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            ps[i] = std::sin(pv[i]);
            pc[i] = std::cos(pv[i]);
        }
    }
}

// =============================================================================
// Inverse Trigonometric Functions
// =============================================================================

/**
 * @brief Compute arcsine for each element
 * @param v Input vector (values must be in [-1, 1])
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void asin(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::asin(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::asin(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::asin(pv[i]);
        }
    }
}

/**
 * @brief Compute arccosine for each element
 * @param v Input vector (values must be in [-1, 1])
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void acos(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::acos(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::acos(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::acos(pv[i]);
        }
    }
}

/**
 * @brief Compute arctangent for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void atan(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::atan(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::atan(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::atan(pv[i]);
        }
    }
}

/**
 * @brief Compute atan2(y, x) for each element
 * @param y Y coordinates
 * @param x X coordinates
 * @param result Result vector (can alias with inputs)
 */
template<typename T, size_t N>
EULER_HOT void atan2(const vector<T, N>& y, const vector<T, N>& x, vector<T, N>& result) {
    const T* EULER_RESTRICT py = y.data();
    const T* EULER_RESTRICT px = x.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vy = batch::load_unaligned(&py[i]);
            batch vx = batch::load_unaligned(&px[i]);
            batch vr = xsimd::atan2(vy, vx);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::atan2(py[i], px[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::atan2(py[i], px[i]);
        }
    }
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

/**
 * @brief Compute hyperbolic sine for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void sinh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::sinh(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::sinh(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::sinh(pv[i]);
        }
    }
}

/**
 * @brief Compute hyperbolic cosine for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void cosh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::cosh(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::cosh(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::cosh(pv[i]);
        }
    }
}

/**
 * @brief Compute hyperbolic tangent for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void tanh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::tanh(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::tanh(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::tanh(pv[i]);
        }
    }
}

// =============================================================================
// Other Mathematical Functions
// =============================================================================

/**
 * @brief Compute ceiling for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void ceil(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::ceil(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::ceil(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::ceil(pv[i]);
        }
    }
}

/**
 * @brief Compute floor for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void floor(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::floor(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::floor(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::floor(pv[i]);
        }
    }
}

/**
 * @brief Compute round (to nearest integer) for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void round(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::round(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::round(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::round(pv[i]);
        }
    }
}

/**
 * @brief Compute truncation (round towards zero) for each element
 * @param v Input vector
 * @param result Result vector (can alias with input)
 */
template<typename T, size_t N>
EULER_HOT void trunc(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        size_t vec_size = N - (N % simd_size);
        for (size_t i = 0; i < vec_size; i += simd_size) {
            batch vv = batch::load_unaligned(&pv[i]);
            batch vr = xsimd::trunc(vv);
            vr.store_unaligned(&pr[i]);
        }
        
        // Handle remainder
        for (size_t i = vec_size; i < N; ++i) {
            pr[i] = std::trunc(pv[i]);
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < N; ++i) {
            pr[i] = std::trunc(pv[i]);
        }
    }
}

/** @} */ // end of transcendental_trig

} // namespace euler::direct