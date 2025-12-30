/**
 * @file transcendental_ops.hh
 * @brief Direct operations for transcendental functions
 * @ingroup DirectModule
 *
 * This file provides direct operations for transcendental functions
 * (exp, log, sin, cos, tan, etc.) that bypass the expression template system.
 */
#pragma once

#include <euler/vector/vector.hh>
#include <euler/core/compiler.hh>
#include <cmath>

namespace euler::direct {

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

/**
 * @brief Compute exponential (e^x) for each element
 */
template<typename T, size_t N>
EULER_HOT void exp(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::exp(pv[i]);
    }
}

/**
 * @brief Compute natural logarithm for each element
 */
template<typename T, size_t N>
EULER_HOT void log(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::log(pv[i]);
    }
}

/**
 * @brief Compute base-10 logarithm for each element
 */
template<typename T, size_t N>
EULER_HOT void log10(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::log10(pv[i]);
    }
}

/**
 * @brief Compute base-2 logarithm for each element
 */
template<typename T, size_t N>
EULER_HOT void log2(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::log2(pv[i]);
    }
}

/**
 * @brief Compute v^p (element-wise power with scalar exponent)
 */
template<typename T, size_t N>
EULER_HOT void pow(const vector<T, N>& v, T p, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::pow(pv[i], p);
    }
}

/**
 * @brief Compute base^exponent (element-wise power)
 */
template<typename T, size_t N>
EULER_HOT void pow(const vector<T, N>& base, const vector<T, N>& exponent, vector<T, N>& result) {
    const T* EULER_RESTRICT pb = base.data();
    const T* EULER_RESTRICT pe = exponent.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::pow(pb[i], pe[i]);
    }
}

// =============================================================================
// Trigonometric Functions
// =============================================================================

/**
 * @brief Compute sine for each element
 */
template<typename T, size_t N>
EULER_HOT void sin(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::sin(pv[i]);
    }
}

/**
 * @brief Compute cosine for each element
 */
template<typename T, size_t N>
EULER_HOT void cos(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::cos(pv[i]);
    }
}

/**
 * @brief Compute tangent for each element
 */
template<typename T, size_t N>
EULER_HOT void tan(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::tan(pv[i]);
    }
}

/**
 * @brief Compute sine and cosine simultaneously
 */
template<typename T, size_t N>
EULER_HOT void sincos(const vector<T, N>& v, vector<T, N>& sin_result, vector<T, N>& cos_result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT ps = sin_result.data();
    T* EULER_RESTRICT pc = cos_result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        ps[i] = std::sin(pv[i]);
        pc[i] = std::cos(pv[i]);
    }
}

// =============================================================================
// Inverse Trigonometric Functions
// =============================================================================

/**
 * @brief Compute arcsine for each element
 */
template<typename T, size_t N>
EULER_HOT void asin(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::asin(pv[i]);
    }
}

/**
 * @brief Compute arccosine for each element
 */
template<typename T, size_t N>
EULER_HOT void acos(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::acos(pv[i]);
    }
}

/**
 * @brief Compute arctangent for each element
 */
template<typename T, size_t N>
EULER_HOT void atan(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::atan(pv[i]);
    }
}

/**
 * @brief Compute atan2(y, x) for each element
 */
template<typename T, size_t N>
EULER_HOT void atan2(const vector<T, N>& y, const vector<T, N>& x, vector<T, N>& result) {
    const T* EULER_RESTRICT py = y.data();
    const T* EULER_RESTRICT px = x.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::atan2(py[i], px[i]);
    }
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

/**
 * @brief Compute hyperbolic sine for each element
 */
template<typename T, size_t N>
EULER_HOT void sinh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::sinh(pv[i]);
    }
}

/**
 * @brief Compute hyperbolic cosine for each element
 */
template<typename T, size_t N>
EULER_HOT void cosh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::cosh(pv[i]);
    }
}

/**
 * @brief Compute hyperbolic tangent for each element
 */
template<typename T, size_t N>
EULER_HOT void tanh(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::tanh(pv[i]);
    }
}

// =============================================================================
// Rounding Functions
// =============================================================================

/**
 * @brief Compute ceiling for each element
 */
template<typename T, size_t N>
EULER_HOT void ceil(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::ceil(pv[i]);
    }
}

/**
 * @brief Compute floor for each element
 */
template<typename T, size_t N>
EULER_HOT void floor(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::floor(pv[i]);
    }
}

/**
 * @brief Compute round (to nearest integer) for each element
 */
template<typename T, size_t N>
EULER_HOT void round(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::round(pv[i]);
    }
}

/**
 * @brief Compute truncation (round towards zero) for each element
 */
template<typename T, size_t N>
EULER_HOT void trunc(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = v.data();
    T* EULER_RESTRICT pr = result.data();

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::trunc(pv[i]);
    }
}

} // namespace euler::direct
