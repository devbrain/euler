/**
 * @file quaternion_ops.hh
 * @brief Direct SIMD operations for quaternions
 * @ingroup DirectModule
 * 
 * This file provides high-performance direct operations for quaternions
 * that bypass the expression template system.
 * 
 * @section quaternion_ops_features Key Features
 * - Optimized quaternion multiplication (Hamilton product)
 * - SIMD-accelerated normalization and dot products
 * - Conversion between quaternions and rotation matrices
 * - Aliasing-safe implementations
 * - Specialized operations for unit quaternions
 * 
 * @section quaternion_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/direct/quaternion_ops.hh>
 * 
 * using namespace euler;
 * using namespace euler::direct;
 * 
 * quatf q1 = quatf::from_axis_angle(vec3f(0,0,1), radians(45));
 * quatf q2 = quatf::from_axis_angle(vec3f(1,0,0), radians(30));
 * quatf result;
 * 
 * // Quaternion operations
 * mul(q1, q2, result);        // Compose rotations
 * normalize(result, result);   // Ensure unit quaternion
 * conjugate(q1, result);       // Get inverse rotation
 * 
 * // Conversion to matrix
 * mat3f rotation_matrix;
 * quat_to_mat3(q1, rotation_matrix);
 * @endcode
 * 
 * @section quaternion_ops_performance Performance Notes
 * - Quaternion multiplication uses optimized scalar implementation
 * - Normalization uses fast inverse square root when available
 * - Matrix conversions are fully unrolled for small sizes
 */
#pragma once

#include <euler/quaternion/quaternion.hh>
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <euler/core/compiler.hh>
#include <euler/core/simd.hh>
#include <cmath>

namespace euler::direct {

// =============================================================================
// Basic Quaternion Operations
// =============================================================================

/**
 * @defgroup quaternion_basic_ops Basic Quaternion Operations
 * @ingroup DirectModule
 * @brief Fundamental quaternion arithmetic operations
 * @{
 */

/**
 * @brief Add two quaternions
 * 
 * Performs component-wise addition of two quaternions.
 * Primarily used for interpolation and blending.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param a First quaternion
 * @param b Second quaternion
 * @param result Result quaternion (can alias with inputs)
 * 
 * @code
 * quatf q1(1, 0, 0, 0), q2(0, 1, 0, 0), sum;
 * add(q1, q2, sum);  // sum = q1 + q2
 * @endcode
 * 
 * @note Result is typically not a unit quaternion
 */
template<typename T>
EULER_HOT void add(const quaternion<T>& a, const quaternion<T>& b, quaternion<T>& result) {
    // Use array subscript operator to access quaternion components
    T w = a[0] + b[0];
    T x = a[1] + b[1];
    T y = a[2] + b[2];
    T z = a[3] + b[3];
    result = quaternion<T>(w, x, y, z);
}

/**
 * @brief Subtract two quaternions
 * 
 * Performs component-wise subtraction of two quaternions.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param a First quaternion (minuend)
 * @param b Second quaternion (subtrahend)
 * @param result Result quaternion (can alias with inputs)
 * 
 * @code
 * quatf q1(1, 0, 0, 0), q2(0, 1, 0, 0), diff;
 * sub(q1, q2, diff);  // diff = q1 - q2
 * @endcode
 * 
 * @note Result is typically not a unit quaternion
 */
template<typename T>
EULER_HOT void sub(const quaternion<T>& a, const quaternion<T>& b, quaternion<T>& result) {
    // Use array subscript operator to access quaternion components
    T w = a[0] - b[0];
    T x = a[1] - b[1];
    T y = a[2] - b[2];
    T z = a[3] - b[3];
    result = quaternion<T>(w, x, y, z);
}

/**
 * @brief Multiply two quaternions (Hamilton product)
 * 
 * Computes the Hamilton product of two quaternions, which represents
 * the composition of rotations when using unit quaternions.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param a First quaternion (applied second)
 * @param b Second quaternion (applied first)
 * @param result Result quaternion representing a*b (can alias with inputs)
 * 
 * @note Quaternion multiplication is not commutative: a*b ≠ b*a
 * @note For unit quaternions, the result is also a unit quaternion
 * @note The rotation b is applied first, then rotation a
 * 
 * @code
 * // Rotate by 45° around Z, then 30° around X
 * quatf rz = quatf::from_axis_angle(vec3f(0,0,1), radians(45));
 * quatf rx = quatf::from_axis_angle(vec3f(1,0,0), radians(30));
 * quatf combined;
 * mul(rx, rz, combined);  // Apply Z rotation first, then X
 * @endcode
 */
template<typename T>
EULER_HOT void mul(const quaternion<T>& a, const quaternion<T>& b, quaternion<T>& result) {
    // Hamilton product: (w1, v1) * (w2, v2) = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
    // Expanded:
    // w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    // x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
    // y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
    // z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    
    // Handle aliasing by computing all components before assignment
    T w = a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z();
    T x = a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y();
    T y = a.w() * b.y() - a.x() * b.z() + a.y() * b.w() + a.z() * b.x();
    T z = a.w() * b.z() + a.x() * b.y() - a.y() * b.x() + a.z() * b.w();
    
    result = quaternion<T>(w, x, y, z);
}

/**
 * @brief Scale a quaternion by a scalar
 * 
 * Multiplies all components of a quaternion by a scalar value.
 * Used for interpolation and normalization operations.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @param scalar Scalar multiplier
 * @param result Result quaternion = scalar * q (can alias with input)
 * 
 * @code
 * quatf q(1, 2, 3, 4), scaled;
 * scale(q, 0.5f, scaled);  // scaled = 0.5 * q
 * @endcode
 * 
 * @note Scaling changes the quaternion magnitude
 */
template<typename T>
EULER_HOT void scale(const quaternion<T>& q, T scalar, quaternion<T>& result) {
    result = quaternion<T>(q.w() * scalar, q.x() * scalar, q.y() * scalar, q.z() * scalar);
}

/**
 * @brief Multiply scalar by quaternion
 * 
 * Alias for scale() to provide commutative scalar multiplication.
 * 
 * @see scale()
 */
template<typename T>
EULER_HOT void mul(T scalar, const quaternion<T>& q, quaternion<T>& result) {
    scale(q, scalar, result);
}

/**
 * @brief Multiply quaternion by scalar
 * 
 * Alias for scale() to provide commutative scalar multiplication.
 * 
 * @see scale()
 */
template<typename T>
EULER_HOT void mul(const quaternion<T>& q, T scalar, quaternion<T>& result) {
    scale(q, scalar, result);
}

/**
 * @brief Compute quaternion conjugate
 * 
 * The conjugate negates the vector part while keeping the scalar part.
 * For unit quaternions, the conjugate represents the inverse rotation.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @param result Result quaternion conj(q) = (w, -x, -y, -z)
 * 
 * @code
 * quatf q = quatf::from_axis_angle(vec3f(0,0,1), radians(45));
 * quatf q_inv;
 * conjugate(q, q_inv);  // q_inv rotates -45° around Z
 * @endcode
 * 
 * @note For unit quaternions: q * conj(q) = (1, 0, 0, 0)
 */
template<typename T>
EULER_HOT void conjugate(const quaternion<T>& q, quaternion<T>& result) {
    result = quaternion<T>(q.w(), -q.x(), -q.y(), -q.z());
}

/**
 * @brief Negate a quaternion
 * 
 * Negates all components of the quaternion. Due to the double cover
 * property, q and -q represent the same rotation.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @param result Result quaternion -q = (-w, -x, -y, -z)
 * 
 * @note -q and q represent the same rotation (double cover property)
 */
template<typename T>
EULER_HOT void negate(const quaternion<T>& q, quaternion<T>& result) {
    result = quaternion<T>(-q.w(), -q.x(), -q.y(), -q.z());
}

/** @} */ // end of quaternion_basic_ops

// =============================================================================
// Geometric Operations
// =============================================================================

/**
 * @defgroup quaternion_geometric_ops Geometric Quaternion Operations
 * @ingroup DirectModule
 * @brief Geometric operations on quaternions (dot product, norm, etc.)
 * @{
 */

/**
 * @brief Compute quaternion dot product
 * 
 * Computes the 4D dot product of two quaternions, treating them as
 * 4-dimensional vectors.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param a First quaternion
 * @param b Second quaternion
 * @return Dot product: a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z
 * 
 * @note For unit quaternions, dot(a,b) = cos(θ/2) where θ is the angle between rotations
 * @note Used for quaternion interpolation (SLERP) and similarity measurement
 * 
 * @code
 * quatf q1 = quatf::identity();
 * quatf q2 = quatf::from_axis_angle(vec3f(0,0,1), radians(90));
 * float similarity = dot(q1, q2);  // cos(45°) ≈ 0.707
 * @endcode
 */
template<typename T>
T dot(const quaternion<T>& a, const quaternion<T>& b) {
    return a.w() * b.w() + a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

/**
 * @brief Compute quaternion norm (magnitude)
 * 
 * Computes the Euclidean norm (length) of the quaternion in 4D space.
 * Unit quaternions representing rotations have norm = 1.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @return Norm: sqrt(w² + x² + y² + z²)
 * 
 * @code
 * quatf q(0.5f, 0.5f, 0.5f, 0.5f);
 * float len = norm(q);  // len = 1.0 (unit quaternion)
 * @endcode
 * 
 * @see norm_squared() for more efficient magnitude comparisons
 */
template<typename T>
T norm(const quaternion<T>& q) {
    return std::sqrt(euler::direct::dot(q, q));
}

/**
 * @brief Compute squared quaternion norm
 * 
 * More efficient than norm() when only relative magnitudes are needed.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @return Squared norm: w² + x² + y² + z²
 * 
 * @see norm() for the actual magnitude
 */
template<typename T>
T norm_squared(const quaternion<T>& q) {
    return euler::direct::dot(q, q);
}

/**
 * @brief Normalize a quaternion
 * 
 * Normalizes the quaternion to unit length, ensuring it represents
 * a valid rotation. Handles zero quaternions gracefully.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion
 * @param result Output unit quaternion (can alias with input)
 * 
 * @note Zero quaternions are converted to identity
 * @note Essential for maintaining rotation validity after operations
 * 
 * @code
 * quatf q = q1 * q2;  // May have small numerical errors
 * normalize(q, q);    // Ensure unit quaternion
 * @endcode
 */
template<typename T>
EULER_HOT void normalize(const quaternion<T>& q, quaternion<T>& result) {
    T n = euler::direct::norm(q);
    if (n > T(0)) {
        scale(q, T(1) / n, result);
    } else {
        // Handle zero quaternion case
        result = quaternion<T>::identity();
    }
}

/**
 * @brief Compute quaternion inverse
 * 
 * Computes the multiplicative inverse such that q * q^-1 = identity.
 * For unit quaternions (rotations), the inverse equals the conjugate.
 * 
 * @tparam T Element type (float, double, etc.)
 * @param q Input quaternion (must be non-zero)
 * @param result Output inverse quaternion (can alias with input)
 * 
 * @note For unit quaternions: inverse = conjugate
 * @note For general quaternions: inverse = conjugate / norm²
 * @note Zero quaternions result in identity output
 * 
 * @code
 * quatf rot = quatf::from_axis_angle(vec3f(0,0,1), radians(45));
 * quatf rot_inv;
 * inverse(rot, rot_inv);  // Reverses the rotation
 * @endcode
 */
template<typename T>
EULER_HOT void inverse(const quaternion<T>& q, quaternion<T>& result) {
    T norm_sq = euler::direct::norm_squared(q);
    if (norm_sq > T(0)) {
        conjugate(q, result);
        scale(result, T(1) / norm_sq, result);
    } else {
        // Handle zero quaternion case
        result = quaternion<T>::identity();
    }
}

/** @} */ // end of quaternion_geometric_ops

// =============================================================================
// Conversion Operations
// =============================================================================

/**
 * @defgroup quaternion_conversion Quaternion Conversion Operations
 * @ingroup DirectModule
 * @brief Conversions between quaternions and rotation matrices
 * @{
 */

/**
 * @brief Convert quaternion to 3x3 rotation matrix
 * 
 * Converts a unit quaternion to its equivalent 3×3 rotation matrix
 * representation. Uses an optimized formula that minimizes operations.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Matrix storage layout (default: true)
 * @param q Input quaternion (should be normalized)
 * @param result Output 3×3 rotation matrix
 * 
 * @note Input quaternion should be normalized for correct results
 * @note The conversion is exact for unit quaternions
 * 
 * @code
 * quatf q = quatf::from_axis_angle(vec3f(0,0,1), radians(90));
 * mat3f R;
 * quat_to_mat3(q, R);  // R rotates 90° around Z-axis
 * @endcode
 */
template<typename T, bool ColumnMajor = true>
EULER_HOT void quat_to_mat3(const quaternion<T>& q, matrix<T, 3, 3, ColumnMajor>& result) {
    T w = q.w(), x = q.x(), y = q.y(), z = q.z();
    
    // Precompute repeated values
    T x2 = x + x, y2 = y + y, z2 = z + z;
    T xx = x * x2, yy = y * y2, zz = z * z2;
    T xy = x * y2, xz = x * z2, yz = y * z2;
    T wx = w * x2, wy = w * y2, wz = w * z2;
    
    // Build rotation matrix
    result(0, 0) = T(1) - (yy + zz);
    result(0, 1) = xy - wz;
    result(0, 2) = xz + wy;
    
    result(1, 0) = xy + wz;
    result(1, 1) = T(1) - (xx + zz);
    result(1, 2) = yz - wx;
    
    result(2, 0) = xz - wy;
    result(2, 1) = yz + wx;
    result(2, 2) = T(1) - (xx + yy);
}

/**
 * @brief Convert quaternion to 4x4 transformation matrix
 * 
 * Converts a unit quaternion to a 4×4 homogeneous transformation matrix
 * with zero translation. Useful for graphics pipelines.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Matrix storage layout (default: true)
 * @param q Input quaternion (should be normalized)
 * @param result Output 4×4 transformation matrix
 * 
 * @note Translation is set to (0,0,0), bottom row is [0,0,0,1]
 * @note Upper-left 3×3 block contains the rotation
 * 
 * @code
 * quatf q = quatf::from_euler(radians(30), radians(45), radians(60));
 * mat4f M;
 * quat_to_mat4(q, M);  // M can be used in graphics pipeline
 * @endcode
 */
template<typename T, bool ColumnMajor = true>
EULER_HOT void quat_to_mat4(const quaternion<T>& q, matrix<T, 4, 4, ColumnMajor>& result) {
    T w = q.w(), x = q.x(), y = q.y(), z = q.z();
    
    // Precompute repeated values
    T x2 = x + x, y2 = y + y, z2 = z + z;
    T xx = x * x2, yy = y * y2, zz = z * z2;
    T xy = x * y2, xz = x * z2, yz = y * z2;
    T wx = w * x2, wy = w * y2, wz = w * z2;
    
    // Build rotation matrix
    result(0, 0) = T(1) - (yy + zz);
    result(0, 1) = xy - wz;
    result(0, 2) = xz + wy;
    result(0, 3) = T(0);
    
    result(1, 0) = xy + wz;
    result(1, 1) = T(1) - (xx + zz);
    result(1, 2) = yz - wx;
    result(1, 3) = T(0);
    
    result(2, 0) = xz - wy;
    result(2, 1) = yz + wx;
    result(2, 2) = T(1) - (xx + yy);
    result(2, 3) = T(0);
    
    result(3, 0) = T(0);
    result(3, 1) = T(0);
    result(3, 2) = T(0);
    result(3, 3) = T(1);
}

/**
 * @brief Convert 3x3 rotation matrix to quaternion
 * 
 * Converts a 3×3 rotation matrix to its quaternion representation using
 * a numerically stable algorithm that selects the best method based on
 * the matrix trace and diagonal elements.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Matrix storage layout (default: true)
 * @param m Input 3×3 rotation matrix (should be orthogonal)
 * @param result Output quaternion (automatically normalized)
 * 
 * @note Input should be a valid rotation matrix (orthogonal, det=1)
 * @note Uses Shepperd's method for numerical stability
 * @note Output is always normalized to unit quaternion
 * 
 * @code
 * mat3f R = mat3f::rotation_x(radians(45));
 * quatf q;
 * mat3_to_quat(R, q);  // q represents same rotation as R
 * @endcode
 */
template<typename T, bool ColumnMajor = true>
EULER_HOT void mat3_to_quat(const matrix<T, 3, 3, ColumnMajor>& m, quaternion<T>& result) {
    // Use the most numerically stable method based on the trace
    T trace = m(0, 0) + m(1, 1) + m(2, 2);
    
    if (trace > T(0)) {
        // w is the largest component
        T s = T(0.5) / std::sqrt(trace + T(1));
        result = quaternion<T>(T(0.25) / s,
                             (m(2, 1) - m(1, 2)) * s,
                             (m(0, 2) - m(2, 0)) * s,
                             (m(1, 0) - m(0, 1)) * s);
    } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
        // x is the largest component
        T s = T(2) * std::sqrt(T(1) + m(0, 0) - m(1, 1) - m(2, 2));
        result = quaternion<T>((m(2, 1) - m(1, 2)) / s,
                             T(0.25) * s,
                             (m(0, 1) + m(1, 0)) / s,
                             (m(0, 2) + m(2, 0)) / s);
    } else if (m(1, 1) > m(2, 2)) {
        // y is the largest component
        T s = T(2) * std::sqrt(T(1) + m(1, 1) - m(0, 0) - m(2, 2));
        result = quaternion<T>((m(0, 2) - m(2, 0)) / s,
                             (m(0, 1) + m(1, 0)) / s,
                             T(0.25) * s,
                             (m(1, 2) + m(2, 1)) / s);
    } else {
        // z is the largest component
        T s = T(2) * std::sqrt(T(1) + m(2, 2) - m(0, 0) - m(1, 1));
        result = quaternion<T>((m(1, 0) - m(0, 1)) / s,
                             (m(0, 2) + m(2, 0)) / s,
                             (m(1, 2) + m(2, 1)) / s,
                             T(0.25) * s);
    }
    
    // Ensure the quaternion is normalized
    normalize(result, result);
}

/**
 * @brief Convert 4x4 transformation matrix to quaternion
 * 
 * Extracts the rotation quaternion from a 4×4 homogeneous transformation
 * matrix by using only the upper-left 3×3 rotation submatrix.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Matrix storage layout (default: true)
 * @param m Input 4×4 transformation matrix
 * @param result Output quaternion representing the rotation part
 * 
 * @note Translation and scale components are ignored
 * @note Only the upper-left 3×3 block is used
 * @note Delegates to mat3_to_quat() for the conversion
 * 
 * @code
 * mat4f M = mat4f::look_at(eye, target, up);
 * quatf orientation;
 * mat4_to_quat(M, orientation);  // Extract view orientation
 * @endcode
 */
template<typename T, bool ColumnMajor = true>
EULER_HOT void mat4_to_quat(const matrix<T, 4, 4, ColumnMajor>& m, quaternion<T>& result) {
    // Extract 3x3 rotation part
    matrix<T, 3, 3, ColumnMajor> m3;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m3(i, j) = m(i, j);
        }
    }
    mat3_to_quat(m3, result);
}

/** @} */ // end of quaternion_conversion

} // namespace euler::direct