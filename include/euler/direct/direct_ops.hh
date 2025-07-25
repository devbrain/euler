/**
 * @file direct_ops.hh
 * @brief Main header for direct SIMD operations
 * @ingroup DirectModule
 * 
 * This header includes all direct SIMD operation headers.
 * 
 * @code
 * #include <euler/direct/direct_ops.hh>
 * 
 * using namespace euler::direct;
 * 
 * vec3<float> a(1, 2, 3), b(4, 5, 6), c;
 * add(a, b, c);        // Direct SIMD addition
 * mul(2.0f, a, a);     // In-place scaling
 * float d = dot(a, b); // Direct dot product
 * @endcode
 */
#pragma once

#include <euler/direct/vector_ops.hh>
#include <euler/direct/matrix_ops.hh>
#include <euler/direct/quaternion_ops.hh>
#include <euler/direct/transcendental_ops.hh>

/**
 * @defgroup DirectModule Direct SIMD Operations
 * @brief High-performance direct operations that bypass expression templates
 * 
 * The Direct module provides optimized implementations of common operations
 * that operate directly on memory without using expression templates. This
 * provides better performance for simple operations and when the result is
 * immediately needed.
 * 
 * Key features:
 * - In-place operation support
 * - Aliasing-safe implementations
 * - SIMD optimization with graceful fallback
 * - Scalar broadcasting support
 * - Portable across compilers (MSVC, GCC, Clang)
 * 
 * @{
 */

namespace euler::direct {

/**
 * @brief Direct operations for vectors
 * 
 * Binary operations:
 * - add(a, b, result) - Vector addition
 * - sub(a, b, result) - Vector subtraction  
 * - mul(a, b, result) - Element-wise multiplication
 * - div(a, b, result) - Element-wise division
 * - min(a, b, result) - Element-wise minimum
 * - max(a, b, result) - Element-wise maximum
 * 
 * Scalar operations (broadcasting):
 * - add(scalar, v, result) - Scalar + vector
 * - sub(scalar, v, result) - Scalar - vector
 * - mul(scalar, v, result) - Scalar * vector
 * - div(scalar, v, result) - Scalar / vector
 * - scale(v, scalar, result) - Alias for scalar multiplication
 * 
 * Geometric operations:
 * - dot(a, b) - Dot product (returns scalar)
 * - cross(a, b, result) - Cross product (3D only)
 * - norm(v) - Euclidean norm (returns scalar)
 * - norm_squared(v) - Squared norm (returns scalar)
 * - normalize(v, result) - Vector normalization
 * 
 * Unary operations:
 * - negate(v, result) - Negation
 * - abs(v, result) - Absolute value
 * - sqrt(v, result) - Square root
 * - rsqrt(v, result) - Reciprocal square root
 * 
 * Advanced operations:
 * - clamp(v, low, high, result) - Element-wise clamping
 * - clamp(v, low_scalar, high_scalar, result) - Clamp with scalar bounds
 * - fma(a, b, c, result) - Fused multiply-add: a*b + c
 * - fma(scalar, b, c, result) - FMA with scalar broadcasting
 * 
 * @brief Direct operations for quaternions
 * 
 * Basic operations:
 * - add(q1, q2, result) - Quaternion addition
 * - sub(q1, q2, result) - Quaternion subtraction
 * - mul(q1, q2, result) - Quaternion multiplication (Hamilton product)
 * - scale(q, scalar, result) - Scalar multiplication
 * - conjugate(q, result) - Quaternion conjugate
 * - negate(q, result) - Negation
 * 
 * Geometric operations:
 * - dot(q1, q2) - Quaternion dot product (returns scalar)
 * - norm(q) - Quaternion norm (returns scalar)
 * - norm_squared(q) - Squared norm (returns scalar)
 * - normalize(q, result) - Quaternion normalization
 * - inverse(q, result) - Quaternion inverse
 * 
 * Conversion operations:
 * - quat_to_mat3(q, result) - Convert quaternion to 3x3 rotation matrix
 * - quat_to_mat4(q, result) - Convert quaternion to 4x4 transformation matrix
 * - mat3_to_quat(m, result) - Convert 3x3 rotation matrix to quaternion
 * - mat4_to_quat(m, result) - Convert 4x4 transformation matrix to quaternion
 * 
 * @brief Direct operations for transcendental functions
 * 
 * Exponential and logarithmic:
 * - exp(v, result) - Exponential (e^x)
 * - log(v, result) - Natural logarithm
 * - log10(v, result) - Base-10 logarithm
 * - log2(v, result) - Base-2 logarithm
 * - pow(v, p, result) - Power function v^p
 * - pow(base, exp, result) - Element-wise power
 * 
 * Trigonometric:
 * - sin(v, result) - Sine
 * - cos(v, result) - Cosine
 * - tan(v, result) - Tangent
 * - sincos(v, sin_result, cos_result) - Simultaneous sine and cosine
 * - asin(v, result) - Arcsine
 * - acos(v, result) - Arccosine
 * - atan(v, result) - Arctangent
 * - atan2(y, x, result) - Two-argument arctangent
 * 
 * Hyperbolic:
 * - sinh(v, result) - Hyperbolic sine
 * - cosh(v, result) - Hyperbolic cosine
 * - tanh(v, result) - Hyperbolic tangent
 * 
 * Rounding:
 * - ceil(v, result) - Round up to nearest integer
 * - floor(v, result) - Round down to nearest integer
 * - round(v, result) - Round to nearest integer
 * - trunc(v, result) - Truncate decimal part
 */

} // namespace euler::direct

/** @} */ // end of DirectModule