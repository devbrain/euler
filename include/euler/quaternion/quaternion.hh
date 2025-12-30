/**
 * @file quaternion.hh
 * @brief Quaternion class for 3D rotations and orientations
 */
#pragma once

#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <euler/core/error.hh>
#include <euler/core/approx_equal.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/math/trigonometry.hh>
#include <euler/math/basic.hh>
#include <cmath>
#include <type_traits>

namespace euler {

/**
 * @enum euler_order
 * @brief Euler angle rotation order for conversions
 * 
 * Specifies the order in which rotations are applied when converting
 * between Euler angles and quaternions.
 */
enum class euler_order {
    XYZ, ///< Rotate around X, then Y, then Z
    XZY, ///< Rotate around X, then Z, then Y
    YXZ, ///< Rotate around Y, then X, then Z
    YZX, ///< Rotate around Y, then Z, then X
    ZXY, ///< Rotate around Z, then X, then Y
    ZYX  ///< Rotate around Z, then Y, then X
};

// Forward declarations
template<typename T> class quaternion;

/**
 * @class quaternion
 * @brief Unit quaternion for representing 3D rotations
 * 
 * Quaternions provide a singularity-free representation of 3D rotations.
 * This class represents unit quaternions (quaternions with norm 1) which
 * correspond to rotations in 3D space.
 * 
 * The quaternion is stored as q = w + xi + yj + zk where:
 * - w is the scalar (real) part
 * - (x,y,z) is the vector (imaginary) part
 * 
 * @tparam T The scalar type (typically float or double)
 */
template<typename T>
class quaternion {
public:
    using value_type = T; ///< The scalar type
    
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /**
     * @brief Default constructor - creates identity quaternion
     * 
     * The identity quaternion represents no rotation: q = (1, 0, 0, 0)
     */
    constexpr quaternion() noexcept 
        : w_(T(1)), x_(T(0)), y_(T(0)), z_(T(0)) {}
    
    /**
     * @brief Construct from components
     * @param w Scalar (real) part
     * @param x First imaginary component (i)
     * @param y Second imaginary component (j)
     * @param z Third imaginary component (k)
     */
    constexpr quaternion(T w, T x, T y, T z) noexcept
        : w_(w), x_(x), y_(y), z_(z) {}
    
    /**
     * @brief Copy constructor
     */
    constexpr quaternion(const quaternion&) noexcept = default;
    
    /**
     * @brief Move constructor
     */
    constexpr quaternion(quaternion&&) noexcept = default;
    
    // ========================================================================
    // Factory methods
    // ========================================================================
    
    /**
     * @brief Create identity quaternion
     * @return Quaternion representing no rotation
     */
    static constexpr quaternion identity() noexcept {
        return quaternion(T(1), T(0), T(0), T(0));
    }
    
    /**
     * @brief Create quaternion from axis-angle representation
     * @tparam Unit The angle unit type (radian_tag or degree_tag)
     * @param axis Rotation axis (must be normalized)
     * @param theta Rotation angle around the axis
     * @return Quaternion representing the rotation
     * @throws euler_error if axis is not normalized
     */
    template<typename Unit>
    static quaternion from_axis_angle(const vector<T, 3>& axis, 
                                     const angle<T, Unit>& theta) {
        #ifdef EULER_DEBUG
        EULER_CHECK(approx_equal(axis.length_squared(), T(1), T(1e-6)),
                    error_code::invalid_argument,
                    "quaternion::from_axis_angle: axis must be normalized");
        #endif
        
        radian<T> half_angle = theta / T(2);
        T s = sin(half_angle);
        T c = cos(half_angle);
        
        return quaternion(c, axis[0] * s, axis[1] * s, axis[2] * s);
    }
    
    /**
     * @brief Create quaternion from Euler angles
     * 
     * Converts Euler angles to a quaternion representation using the specified
     * rotation order. The angles represent rotations around the principal axes:
     * - roll: rotation around X axis
     * - pitch: rotation around Y axis  
     * - yaw: rotation around Z axis
     * 
     * @tparam Unit The angle unit type (radian_tag or degree_tag)
     * @param roll Rotation angle around X axis
     * @param pitch Rotation angle around Y axis
     * @param yaw Rotation angle around Z axis
     * @param order The order in which rotations are applied
     * @return Quaternion representing the combined rotation
     * 
     * @note The XZY order bug has been fixed as of version 1.1
     */
    template<typename Unit>
    static quaternion from_euler(const angle<T, Unit>& roll,   // X rotation
                                const angle<T, Unit>& pitch,  // Y rotation  
                                const angle<T, Unit>& yaw,    // Z rotation
                                euler_order order = euler_order::XYZ) {
        // Convert to radians
        radian<T> r = roll / T(2);
        radian<T> p = pitch / T(2);
        radian<T> y = yaw / T(2);
        
        // Compute sin/cos for half angles
        T sr = sin(r), cr = cos(r);
        T sp = sin(p), cp = cos(p);
        T sy = sin(y), cy = cos(y);
        
        // Compute quaternion based on rotation order
        switch (order) {
            case euler_order::XYZ:
                return quaternion(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                );
            case euler_order::XZY:
                return quaternion(
                    cp * cy * cr - sp * sy * sr,
                    cp * cy * sr + sp * sy * cr,
                    cp * sy * sr + sp * cy * cr,
                    cp * sy * cr - sp * cy * sr
                );
            case euler_order::YXZ:
                return quaternion(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy + cr * sp * sy,
                    cr * sp * cy - sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                );
            case euler_order::YZX:
                return quaternion(
                    cr * cp * cy - sr * sp * sy,
                    sr * cp * cy + cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                );
            case euler_order::ZXY:
                return quaternion(
                    cr * cp * cy - sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy + sr * sp * cy
                );
            case euler_order::ZYX:
                return quaternion(
                    cr * cp * cy - sr * sp * sy,
                    sr * cp * cy + cr * sp * sy,
                    cr * sp * cy - sr * cp * sy,
                    cr * cp * sy + sr * sp * cy
                );
            default:
                return identity();
        }
    }
    
    // From rotation matrix (3x3)
    static quaternion from_matrix(const matrix<T, 3, 3>& m) {
        T trace = m(0, 0) + m(1, 1) + m(2, 2);
        
        if (trace > T(0)) {
            T s = T(0.5) / sqrt(trace + T(1));
            return quaternion(
                T(0.25) / s,
                (m(2, 1) - m(1, 2)) * s,
                (m(0, 2) - m(2, 0)) * s,
                (m(1, 0) - m(0, 1)) * s
            );
        } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
            T s = T(2) * sqrt(T(1) + m(0, 0) - m(1, 1) - m(2, 2));
            return quaternion(
                (m(2, 1) - m(1, 2)) / s,
                T(0.25) * s,
                (m(0, 1) + m(1, 0)) / s,
                (m(0, 2) + m(2, 0)) / s
            );
        } else if (m(1, 1) > m(2, 2)) {
            T s = T(2) * sqrt(T(1) + m(1, 1) - m(0, 0) - m(2, 2));
            return quaternion(
                (m(0, 2) - m(2, 0)) / s,
                (m(0, 1) + m(1, 0)) / s,
                T(0.25) * s,
                (m(1, 2) + m(2, 1)) / s
            );
        } else {
            T s = T(2) * sqrt(T(1) + m(2, 2) - m(0, 0) - m(1, 1));
            return quaternion(
                (m(1, 0) - m(0, 1)) / s,
                (m(0, 2) + m(2, 0)) / s,
                (m(1, 2) + m(2, 1)) / s,
                T(0.25) * s
            );
        }
    }
    
    // From rotation matrix (4x4 - extract upper-left 3x3)
    static quaternion from_matrix(const matrix<T, 4, 4>& m) {
        matrix<T, 3, 3> m3;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m3(i, j) = m(i, j);
            }
        }
        return from_matrix(m3);
    }
    
    // From rotation from one vector to another
    static quaternion from_vectors(const vector<T, 3>& from, const vector<T, 3>& to) {
        #ifdef EULER_DEBUG
        EULER_CHECK(approx_equal(from.length_squared(), T(1), T(1e-6)),
                    error_code::invalid_argument,
                    "quaternion::from_vectors: 'from' must be normalized");
        EULER_CHECK(approx_equal(to.length_squared(), T(1), T(1e-6)),
                    error_code::invalid_argument,
                    "quaternion::from_vectors: 'to' must be normalized");
        #endif
        
        T d = dot(from, to);
        
        // Handle parallel vectors
        if (d >= T(1) - constants<T>::epsilon) {
            return identity();
        }
        
        // Handle anti-parallel vectors (use wider margin to avoid numerical instability)
        // When d is close to -1, sqrt((1+d)*2) becomes very small, causing large invs
        constexpr T antiparallel_threshold = T(1e-4);
        if (d <= -T(1) + antiparallel_threshold) {
            // Find an orthogonal vector
            vector<T, 3> axis = std::abs(from[0]) > T(0.9)
                ? cross(from, vector<T, 3>(0, 1, 0))
                : cross(from, vector<T, 3>(1, 0, 0));
            axis = euler::normalize(axis);
            return from_axis_angle(axis, radian<T>(constants<T>::pi));
        }

        // General case
        vector<T, 3> axis = cross(from, to);
        T s = sqrt((T(1) + d) * T(2));
        T invs = T(1) / s;
        
        return quaternion(
            s * T(0.5),
            axis[0] * invs,
            axis[1] * invs,
            axis[2] * invs
        );
    }
    
    // ========================================================================
    // Assignment operators
    // ========================================================================
    
    quaternion& operator=(const quaternion&) noexcept = default;
    quaternion& operator=(quaternion&&) noexcept = default;
    
    // ========================================================================
    // Component access
    // ========================================================================
    
    T& w() noexcept { return w_; }
    T& x() noexcept { return x_; }
    T& y() noexcept { return y_; }
    T& z() noexcept { return z_; }
    
    constexpr const T& w() const noexcept { return w_; }
    constexpr const T& x() const noexcept { return x_; }
    constexpr const T& y() const noexcept { return y_; }
    constexpr const T& z() const noexcept { return z_; }
    
    // Array-like access (order: w, x, y, z)
    constexpr T& operator[](size_t idx) {
        EULER_CHECK_INDEX(idx, 4);
        if (idx == 0) return w_;
        if (idx == 1) return x_;
        if (idx == 2) return y_;
        return z_;
    }

    constexpr const T& operator[](size_t idx) const {
        EULER_CHECK_INDEX(idx, 4);
        if (idx == 0) return w_;
        if (idx == 1) return x_;
        if (idx == 2) return y_;
        return z_;
    }
    
    // Vector part (imaginary components)
    constexpr vector<T, 3> vec() const noexcept { 
        return vector<T, 3>(x_, y_, z_); 
    }
    
    // Scalar part (real component)
    constexpr T scalar() const noexcept { return w_; }
    
    // ========================================================================
    // Properties
    // ========================================================================
    
    // Squared norm (w² + x² + y² + z²)
    constexpr T norm_squared() const noexcept {
        return w_*w_ + x_*x_ + y_*y_ + z_*z_;
    }
    
    // Norm (length)
    T norm() const noexcept {
        return sqrt(norm_squared());
    }

    // Alias for norm
    T length() const noexcept {
        return norm();
    }

    // Check if normalized (unit quaternion)
    bool is_normalized() const noexcept {
        return is_normalized(T(1e-6));
    }

    bool is_normalized(T tolerance) const noexcept {
        return approx_equal(norm_squared(), T(1), tolerance);
    }
    
    // Check if pure quaternion (zero scalar part)
    bool is_pure() const {
        return is_pure(constants<T>::epsilon);
    }
    
    bool is_pure(T tolerance) const {
        return approx_equal(w_, T(0), tolerance);
    }
    
    // ========================================================================
    // Normalization
    // ========================================================================
    
    // Normalize in place
    quaternion& normalize() {
        T len = norm();
        #ifdef EULER_DEBUG
        EULER_CHECK(len > constants<T>::epsilon, error_code::invalid_argument,
                    "quaternion::normalize: cannot normalize zero quaternion");
        #endif
        
        T inv_len = T(1) / len;
        w_ *= inv_len;
        x_ *= inv_len;
        y_ *= inv_len;
        z_ *= inv_len;
        return *this;
    }
    
    // Return normalized copy
    quaternion normalized() const {
        quaternion q = *this;
        return q.normalize();
    }
    
    // ========================================================================
    // Angle and axis extraction
    // ========================================================================
    
    // Get rotation angle
    radian<T> angle() const {
        // For a unit quaternion q = (w, x, y, z) representing rotation by angle θ:
        // w = cos(θ/2), ||(x,y,z)|| = sin(θ/2)
        
        // For very small angles, use the vector part directly
        // since sin(θ/2) ≈ θ/2 for small θ
        T vec_len_sq = x_*x_ + y_*y_ + z_*z_;
        if (vec_len_sq < T(1e-8)) {
            // Very small angle - use approximation
            return radian<T>(T(2) * sqrt(vec_len_sq));
        }
        
        // Normal case - use atan2 for better numerical stability
        return radian<T>(T(2) * atan2(sqrt(vec_len_sq), w_));
    }
    
    // Get rotation axis (normalized)
    vector<T, 3> axis() const {
        T s_squared = T(1) - w_*w_;
        
        // Handle near-identity quaternion
        if (s_squared < T(1e-10)) {
            // For very small angles, the axis is approximately the vector part normalized
            T len = sqrt(x_*x_ + y_*y_ + z_*z_);
            if (len < constants<T>::epsilon) {
                // True identity quaternion - return arbitrary axis
                return vector<T, 3>::unit_x();
            }
            return vector<T, 3>(x_/len, y_/len, z_/len);
        }
        
        T s = sqrt(s_squared);
        return vector<T, 3>(x_/s, y_/s, z_/s);
    }
    
    // Get axis and angle together
    std::pair<vector<T, 3>, radian<T>> to_axis_angle() const {
        return {axis(), angle()};
    }
    
    // ========================================================================
    // Vector rotation
    // ========================================================================
    
    // Rotate a vector using this quaternion
    vector<T, 3> rotate(const vector<T, 3>& v) const {
        // Optimized formula: v' = v + 2w(q.vec × v) + 2(q.vec × (q.vec × v))
        vector<T, 3> qv = vec();
        vector<T, 3> uv = cross(qv, v);
        vector<T, 3> uuv = cross(qv, uv);
        
        return v + T(2) * (w_ * uv + uuv);
    }
    
    // Operator version of rotate
    vector<T, 3> operator*(const vector<T, 3>& v) const {
        return rotate(v);
    }
    
    // ========================================================================
    // Matrix conversion
    // ========================================================================
    
    // Convert to 3x3 rotation matrix
    matrix<T, 3, 3> to_matrix3() const {
        T xx = x_ * x_;
        T xy = x_ * y_;
        T xz = x_ * z_;
        T xw = x_ * w_;
        
        T yy = y_ * y_;
        T yz = y_ * z_;
        T yw = y_ * w_;
        
        T zz = z_ * z_;
        T zw = z_ * w_;
        
        matrix<T, 3, 3> m;
        m(0, 0) = T(1) - T(2) * (yy + zz);
        m(0, 1) = T(2) * (xy - zw);
        m(0, 2) = T(2) * (xz + yw);
        
        m(1, 0) = T(2) * (xy + zw);
        m(1, 1) = T(1) - T(2) * (xx + zz);
        m(1, 2) = T(2) * (yz - xw);
        
        m(2, 0) = T(2) * (xz - yw);
        m(2, 1) = T(2) * (yz + xw);
        m(2, 2) = T(1) - T(2) * (xx + yy);
        
        return m;
    }
    
    // Convert to 4x4 homogeneous transformation matrix
    matrix<T, 4, 4> to_matrix4() const {
        matrix<T, 4, 4> m = matrix<T, 4, 4>::identity();
        matrix<T, 3, 3> m3 = to_matrix3();
        
        // Copy 3x3 rotation part
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m(i, j) = m3(i, j);
            }
        }
        
        return m;
    }
    
    // ========================================================================
    // Euler angle extraction
    // ========================================================================
    
    // Extract Euler angles (returns radians)
    vector<radian<T>, 3> to_euler(euler_order order = euler_order::XYZ) const {
        matrix<T, 3, 3> m = to_matrix3();
        radian<T> x, y, z;
        constexpr T gimbal_threshold = T(1e-6);

        switch (order) {
            case euler_order::XYZ: {
                T sy = sqrt(m(0,0)*m(0,0) + m(1,0)*m(1,0));
                if (sy > gimbal_threshold) {
                    x = atan2(m(2,1), m(2,2));
                    y = atan2(-m(2,0), sy);
                    z = atan2(m(1,0), m(0,0));
                } else {
                    x = atan2(-m(1,2), m(1,1));
                    y = atan2(-m(2,0), sy);
                    z = radian<T>(0);
                }
                break;
            }

            case euler_order::XZY: {
                T sz = sqrt(m(0,0)*m(0,0) + m(2,0)*m(2,0));
                if (sz > gimbal_threshold) {
                    x = atan2(-m(1,2), m(1,1));
                    z = atan2(-m(2,0), m(0,0));
                    y = atan2(m(1,0), sz);
                } else {
                    x = atan2(m(2,1), m(2,2));
                    z = radian<T>(0);
                    y = atan2(m(1,0), sz);
                }
                break;
            }

            case euler_order::YXZ: {
                T sx = sqrt(m(1,1)*m(1,1) + m(2,1)*m(2,1));
                if (sx > gimbal_threshold) {
                    y = atan2(m(0,2), m(0,0));
                    x = atan2(m(2,1), sx);
                    z = atan2(-m(0,1), sqrt(m(0,0)*m(0,0) + m(0,2)*m(0,2)));
                } else {
                    y = atan2(-m(2,0), m(2,2));
                    x = atan2(m(2,1), sx);
                    z = radian<T>(0);
                }
                break;
            }

            case euler_order::YZX: {
                T sz = sqrt(m(0,0)*m(0,0) + m(0,2)*m(0,2));
                if (sz > gimbal_threshold) {
                    y = atan2(-m(2,0), m(0,0));
                    z = atan2(m(0,1), sz);
                    x = atan2(-m(2,1), m(1,1));
                } else {
                    y = atan2(m(0,2), m(2,2));
                    z = atan2(m(0,1), sz);
                    x = radian<T>(0);
                }
                break;
            }

            case euler_order::ZXY: {
                T sx = sqrt(m(0,0)*m(0,0) + m(0,1)*m(0,1));
                if (sx > gimbal_threshold) {
                    z = atan2(-m(0,1), m(0,0));
                    x = atan2(m(1,2), sx);
                    y = atan2(-m(0,2), sqrt(m(0,0)*m(0,0) + m(0,1)*m(0,1)));
                } else {
                    z = atan2(m(1,0), m(1,1));
                    x = atan2(m(1,2), sx);
                    y = radian<T>(0);
                }
                break;
            }

            case euler_order::ZYX: {
                T sy = sqrt(m(0,0)*m(0,0) + m(0,1)*m(0,1));
                if (sy > gimbal_threshold) {
                    z = atan2(m(1,0), m(0,0));
                    y = atan2(-m(2,0), sy);
                    x = atan2(m(2,1), m(2,2));
                } else {
                    z = atan2(-m(0,1), m(1,1));
                    y = atan2(-m(2,0), sy);
                    x = radian<T>(0);
                }
                break;
            }
        }

        return vector<radian<T>, 3>(x, y, z);
    }
    
    // ========================================================================
    // Arithmetic operators
    // ========================================================================
    
    // Unary negation
    quaternion operator-() const {
        return quaternion(-w_, -x_, -y_, -z_);
    }
    
    // Addition
    quaternion operator+(const quaternion& q) const {
        return quaternion(w_ + q.w_, x_ + q.x_, y_ + q.y_, z_ + q.z_);
    }
    
    quaternion& operator+=(const quaternion& q) {
        w_ += q.w_;
        x_ += q.x_;
        y_ += q.y_;
        z_ += q.z_;
        return *this;
    }
    
    // Subtraction
    quaternion operator-(const quaternion& q) const {
        return quaternion(w_ - q.w_, x_ - q.x_, y_ - q.y_, z_ - q.z_);
    }
    
    quaternion& operator-=(const quaternion& q) {
        w_ -= q.w_;
        x_ -= q.x_;
        y_ -= q.y_;
        z_ -= q.z_;
        return *this;
    }
    
    // Scalar multiplication
    quaternion operator*(T s) const {
        return quaternion(w_ * s, x_ * s, y_ * s, z_ * s);
    }
    
    quaternion& operator*=(T s) {
        w_ *= s;
        x_ *= s;
        y_ *= s;
        z_ *= s;
        return *this;
    }
    
    // Scalar division
    quaternion operator/(T s) const {
        #ifdef EULER_DEBUG
        EULER_CHECK(std::abs(s) > constants<T>::epsilon, error_code::invalid_argument,
                    "quaternion: division by zero");
        #endif
        T inv_s = T(1) / s;
        return quaternion(w_ * inv_s, x_ * inv_s, y_ * inv_s, z_ * inv_s);
    }
    
    quaternion& operator/=(T s) {
        #ifdef EULER_DEBUG
        EULER_CHECK(std::abs(s) > constants<T>::epsilon, error_code::invalid_argument,
                    "quaternion: division by zero");
        #endif
        T inv_s = T(1) / s;
        w_ *= inv_s;
        x_ *= inv_s;
        y_ *= inv_s;
        z_ *= inv_s;
        return *this;
    }
    
    // ========================================================================
    // Comparison operators
    // ========================================================================
    
    bool operator==(const quaternion& q) const {
        return w_ == q.w_ && x_ == q.x_ && y_ == q.y_ && z_ == q.z_;
    }
    
    bool operator!=(const quaternion& q) const {
        return !(*this == q);
    }
    
private:
    // Storage: w (scalar/real), x, y, z (vector/imaginary)
    T w_, x_, y_, z_;
};

// ============================================================================
// Type aliases
// ============================================================================

using quatf = quaternion<float>;
using quatd = quaternion<double>;

// Complex quaternions (biquaternions)
template<typename T>
using biquaternion = quaternion<complex<T>>;

using biquatf = biquaternion<float>;
using biquatd = biquaternion<double>;

// ============================================================================
// Non-member operators
// ============================================================================

// Scalar multiplication (scalar first)
template<typename T>
inline quaternion<T> operator*(T s, const quaternion<T>& q) {
    return q * s;
}

// ============================================================================
// Utility functions
// ============================================================================

// Approximate equality
template<typename T>
inline bool approx_equal(const quaternion<T>& a, const quaternion<T>& b) {
    return approx_equal(a, b, constants<T>::epsilon);
}

template<typename T>
inline bool approx_equal(const quaternion<T>& a, const quaternion<T>& b, 
                        T tolerance) {
    return approx_equal(a.w(), b.w(), tolerance) &&
           approx_equal(a.x(), b.x(), tolerance) &&
           approx_equal(a.y(), b.y(), tolerance) &&
           approx_equal(a.z(), b.z(), tolerance);
}

} // namespace euler