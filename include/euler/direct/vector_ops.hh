/**
 * @file vector_ops.hh
 * @brief Direct operations for vectors
 * @ingroup DirectModule
 *
 * This header provides high-performance direct operations on vectors
 * that bypass the expression template system for maximum performance.
 *
 * @section vector_ops_features Key Features
 * - In-place operation support (result can alias inputs)
 * - Scalar broadcasting for mixed scalar-vector operations
 * - Support for all vector dimensions
 *
 * @section vector_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/direct/vector_ops.hh>
 *
 * using namespace euler;
 * using namespace euler::direct;
 *
 * vec3<float> a(1.0f, 2.0f, 3.0f);
 * vec3<float> b(4.0f, 5.0f, 6.0f);
 * vec3<float> result;
 *
 * // Basic arithmetic
 * add(a, b, result);      // result = a + b
 * mul(2.0f, a, result);   // result = 2.0 * a
 *
 * // Geometric operations
 * float d = dot(a, b);    // Dot product
 * cross(a, b, result);    // Cross product (3D only)
 * normalize(a, result);   // Unit vector
 *
 * // In-place operations
 * add(a, b, a);           // a = a + b (aliasing safe)
 * @endcode
 */
#pragma once

#include <euler/vector/vector.hh>
#include <euler/core/compiler.hh>
#include <euler/core/types.hh>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace euler::direct {

// Helper to extract data pointer from various container types
namespace detail {
    template<typename Container>
    EULER_ALWAYS_INLINE auto data_ptr(Container& c) -> decltype(c.data()) {
        return c.data();
    }

    template<typename Container>
    EULER_ALWAYS_INLINE auto data_ptr(const Container& c) -> decltype(c.data()) {
        return c.data();
    }
}

// =============================================================================
// Binary Operations
// =============================================================================

/**
 * @brief Vector addition: result = op1 + op2
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void add(const vector<T, N>& op1, const vector<T, N>& op2, vector<T, N>& result) {
    const T* EULER_RESTRICT p1 = detail::data_ptr(op1);
    const T* EULER_RESTRICT p2 = detail::data_ptr(op2);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = p1[i] + p2[i];
    }
}

/**
 * @brief Vector subtraction: result = op1 - op2
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void sub(const vector<T, N>& op1, const vector<T, N>& op2, vector<T, N>& result) {
    const T* EULER_RESTRICT p1 = detail::data_ptr(op1);
    const T* EULER_RESTRICT p2 = detail::data_ptr(op2);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = p1[i] - p2[i];
    }
}

/**
 * @brief Element-wise multiplication (Hadamard product): result = op1 * op2
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void mul(const vector<T, N>& op1, const vector<T, N>& op2, vector<T, N>& result) {
    const T* EULER_RESTRICT p1 = detail::data_ptr(op1);
    const T* EULER_RESTRICT p2 = detail::data_ptr(op2);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = p1[i] * p2[i];
    }
}

/**
 * @brief Element-wise division: result = op1 / op2
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void div(const vector<T, N>& op1, const vector<T, N>& op2, vector<T, N>& result) {
    const T* EULER_RESTRICT p1 = detail::data_ptr(op1);
    const T* EULER_RESTRICT p2 = detail::data_ptr(op2);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = p1[i] / p2[i];
    }
}

// =============================================================================
// Scalar Broadcasting Operations
// =============================================================================

/**
 * @brief Scalar + Vector: result = scalar + v
 */
template<typename T, size_t N>
EULER_HOT void add(T scalar, const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = scalar + pv[i];
    }
}

/**
 * @brief Vector + Scalar: result = v + scalar
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void add(const vector<T, N>& v, T scalar, vector<T, N>& result) {
    add(scalar, v, result);
}

/**
 * @brief Scalar - Vector: result = scalar - v
 */
template<typename T, size_t N>
EULER_HOT void sub(T scalar, const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = scalar - pv[i];
    }
}

/**
 * @brief Vector - Scalar: result = v - scalar
 */
template<typename T, size_t N>
EULER_HOT void sub(const vector<T, N>& v, T scalar, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = pv[i] - scalar;
    }
}

/**
 * @brief Scalar * Vector: result = scalar * v
 */
template<typename T, size_t N>
EULER_HOT void mul(T scalar, const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = scalar * pv[i];
    }
}

/**
 * @brief Vector * Scalar: result = v * scalar
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void mul(const vector<T, N>& v, T scalar, vector<T, N>& result) {
    mul(scalar, v, result);
}

/**
 * @brief Scalar multiplication (alias for mul)
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE void scale(const vector<T, N>& v, T scalar, vector<T, N>& result) {
    mul(scalar, v, result);
}

/**
 * @brief Scalar / Vector: result = scalar / v
 */
template<typename T, size_t N>
EULER_HOT void div(T scalar, const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = scalar / pv[i];
    }
}

/**
 * @brief Vector / Scalar: result = v / scalar
 */
template<typename T, size_t N>
EULER_HOT void div(const vector<T, N>& v, T scalar, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    const T inv_scalar = T(1) / scalar;

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = pv[i] * inv_scalar;
    }
}

// =============================================================================
// Geometric Operations
// =============================================================================

/**
 * @brief Dot product: returns a . b
 */
template<typename T, size_t N>
EULER_HOT T dot(const vector<T, N>& a, const vector<T, N>& b) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pb = detail::data_ptr(b);

    T sum = T(0);
    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        sum += pa[i] * pb[i];
    }
    return sum;
}

/**
 * @brief Cross product for 3D vectors: result = a x b
 */
template<typename T>
EULER_ALWAYS_INLINE void cross(const vector<T, 3>& a, const vector<T, 3>& b, vector<T, 3>& result) {
    // Handle aliasing by computing all components before storing
    const T r0 = a[1] * b[2] - a[2] * b[1];
    const T r1 = a[2] * b[0] - a[0] * b[2];
    const T r2 = a[0] * b[1] - a[1] * b[0];

    result[0] = r0;
    result[1] = r1;
    result[2] = r2;
}

/**
 * @brief Squared norm: returns ||v||^2
 */
template<typename T, size_t N>
EULER_HOT T norm_squared(const vector<T, N>& v) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);

    T sum = T(0);
    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        sum += pv[i] * pv[i];
    }
    return sum;
}

/**
 * @brief Euclidean norm: returns ||v||
 */
template<typename T, size_t N>
EULER_ALWAYS_INLINE T norm(const vector<T, N>& v) {
    return std::sqrt(norm_squared(v));
}

/**
 * @brief Vector normalization: result = v / ||v||
 * @note Returns zero vector if input has zero or near-zero length
 */
template<typename T, size_t N>
EULER_HOT void normalize(const vector<T, N>& v, vector<T, N>& result) {
    T n = norm(v);
    if (n <= std::numeric_limits<T>::epsilon()) {
        // Zero-length vector: return zero vector
        T* EULER_RESTRICT pr = detail::data_ptr(result);
        for (size_t i = 0; i < N; ++i) {
            pr[i] = T(0);
        }
        return;
    }
    T inv_norm = T(1) / n;
    mul(inv_norm, v, result);
}

// =============================================================================
// Unary Operations
// =============================================================================

/**
 * @brief Negation: result = -v
 */
template<typename T, size_t N>
EULER_HOT void negate(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = -pv[i];
    }
}

/**
 * @brief Absolute value: result = |v|
 */
template<typename T, size_t N>
EULER_HOT void abs(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::abs(pv[i]);
    }
}

/**
 * @brief Square root: result = sqrt(v)
 */
template<typename T, size_t N>
EULER_HOT void sqrt(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::sqrt(pv[i]);
    }
}

/**
 * @brief Reciprocal square root: result = 1/sqrt(v)
 */
template<typename T, size_t N>
EULER_HOT void rsqrt(const vector<T, N>& v, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = T(1) / std::sqrt(pv[i]);
    }
}

// =============================================================================
// Advanced Operations
// =============================================================================

/**
 * @brief Element-wise minimum: result = min(a, b)
 */
template<typename T, size_t N>
EULER_HOT void min(const vector<T, N>& a, const vector<T, N>& b, vector<T, N>& result) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pb = detail::data_ptr(b);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::min(pa[i], pb[i]);
    }
}

/**
 * @brief Element-wise maximum: result = max(a, b)
 */
template<typename T, size_t N>
EULER_HOT void max(const vector<T, N>& a, const vector<T, N>& b, vector<T, N>& result) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pb = detail::data_ptr(b);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::max(pa[i], pb[i]);
    }
}

/**
 * @brief Element-wise clamp: result = clamp(v, low, high)
 */
template<typename T, size_t N>
EULER_HOT void clamp(const vector<T, N>& v, const vector<T, N>& low, const vector<T, N>& high, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    const T* EULER_RESTRICT plow = detail::data_ptr(low);
    const T* EULER_RESTRICT phigh = detail::data_ptr(high);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::clamp(pv[i], plow[i], phigh[i]);
    }
}

/**
 * @brief Clamp with scalar bounds: result = clamp(v, low_scalar, high_scalar)
 */
template<typename T, size_t N>
EULER_HOT void clamp(const vector<T, N>& v, T low, T high, vector<T, N>& result) {
    const T* EULER_RESTRICT pv = detail::data_ptr(v);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = std::clamp(pv[i], low, high);
    }
}

// =============================================================================
// Fused Multiply-Add Operations
// =============================================================================

/**
 * @brief Fused multiply-add: result = a * b + c
 */
template<typename T, size_t N>
EULER_HOT void fma(const vector<T, N>& a, const vector<T, N>& b, const vector<T, N>& c, vector<T, N>& result) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pb = detail::data_ptr(b);
    const T* EULER_RESTRICT pc = detail::data_ptr(c);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = pa[i] * pb[i] + pc[i];
    }
}

/**
 * @brief Fused multiply-add with scalar: result = scalar * b + c
 */
template<typename T, size_t N>
EULER_HOT void fma(T a, const vector<T, N>& b, const vector<T, N>& c, vector<T, N>& result) {
    const T* EULER_RESTRICT pb = detail::data_ptr(b);
    const T* EULER_RESTRICT pc = detail::data_ptr(c);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = a * pb[i] + pc[i];
    }
}

/**
 * @brief Fused multiply-add with scalar: result = a * scalar + c
 */
template<typename T, size_t N>
EULER_HOT void fma(const vector<T, N>& a, T b, const vector<T, N>& c, vector<T, N>& result) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pc = detail::data_ptr(c);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = pa[i] * b + pc[i];
    }
}

/**
 * @brief Fused multiply-add with scalar: result = a * b + scalar
 */
template<typename T, size_t N>
EULER_HOT void fma(const vector<T, N>& a, const vector<T, N>& b, T c, vector<T, N>& result) {
    const T* EULER_RESTRICT pa = detail::data_ptr(a);
    const T* EULER_RESTRICT pb = detail::data_ptr(b);
    T* EULER_RESTRICT pr = detail::data_ptr(result);

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < N; ++i) {
        pr[i] = pa[i] * pb[i] + c;
    }
}

} // namespace euler::direct
