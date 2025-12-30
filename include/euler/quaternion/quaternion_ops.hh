#pragma once

#include <euler/quaternion/quaternion.hh>
#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>
#include <cmath>

namespace euler {

// ============================================================================
// Conjugate and Inverse
// ============================================================================

// Conjugate: q* = (w, -x, -y, -z)
template<typename T>
inline quaternion<T> conjugate(const quaternion<T>& q) {
    return quaternion<T>(q.w(), -q.x(), -q.y(), -q.z());
}

// Inverse: q^(-1) = q* / |q|²
template<typename T>
inline quaternion<T> inverse(const quaternion<T>& q) {
    T norm_sq = q.norm_squared();
    EULER_CHECK(norm_sq > constants<T>::epsilon, error_code::invalid_argument,
                "quaternion::inverse: cannot invert zero quaternion");

    T inv_norm = T(1) / norm_sq;
    return quaternion<T>(
        q.w() * inv_norm,
        -q.x() * inv_norm,
        -q.y() * inv_norm,
        -q.z() * inv_norm
    );
}

// ============================================================================
// Hamilton Product (Quaternion Multiplication)
// ============================================================================

// Quaternion multiplication: p * q
// Formula: (p₀ + p⃗) * (q₀ + q⃗) = (p₀q₀ - p⃗·q⃗) + (p₀q⃗ + q₀p⃗ + p⃗×q⃗)
template<typename T>
inline quaternion<T> operator*(const quaternion<T>& p, const quaternion<T>& q) {
    return quaternion<T>(
        p.w()*q.w() - p.x()*q.x() - p.y()*q.y() - p.z()*q.z(),
        p.w()*q.x() + p.x()*q.w() + p.y()*q.z() - p.z()*q.y(),
        p.w()*q.y() - p.x()*q.z() + p.y()*q.w() + p.z()*q.x(),
        p.w()*q.z() + p.x()*q.y() - p.y()*q.x() + p.z()*q.w()
    );
}

// Compound assignment multiplication
template<typename T>
inline quaternion<T>& operator*=(quaternion<T>& p, const quaternion<T>& q) {
    p = p * q;
    return p;
}

// ============================================================================
// Quaternion Division
// ============================================================================

// Quaternion division: p / q = p * q^(-1)
template<typename T>
inline quaternion<T> operator/(const quaternion<T>& p, const quaternion<T>& q) {
    return p * inverse(q);
}

// Compound assignment division
template<typename T>
inline quaternion<T>& operator/=(quaternion<T>& p, const quaternion<T>& q) {
    p = p / q;
    return p;
}

// ============================================================================
// Dot Product
// ============================================================================

// Dot product: p·q = p₀q₀ + p₁q₁ + p₂q₂ + p₃q₃
template<typename T>
inline T dot(const quaternion<T>& p, const quaternion<T>& q) {
    return p.w()*q.w() + p.x()*q.x() + p.y()*q.y() + p.z()*q.z();
}

// ============================================================================
// Exponential and Logarithm
// ============================================================================

// Quaternion exponential: e^q
// For pure quaternion q = (0, v⃗): e^q = (cos|v⃗|, v̂ sin|v⃗|)
// For general quaternion: e^q = e^w * (cos|v⃗|, v̂ sin|v⃗|)
template<typename T>
inline quaternion<T> exp(const quaternion<T>& q) {
    T vlen = sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
    
    if (vlen < constants<T>::epsilon) {
        // Near-scalar quaternion
        return quaternion<T>(exp(q.w()), T(0), T(0), T(0));
    }
    
    T exp_w = exp(q.w());
    T sinc_vlen = sin(vlen) / vlen;
    
    return quaternion<T>(
        exp_w * cos(vlen),
        exp_w * q.x() * sinc_vlen,
        exp_w * q.y() * sinc_vlen,
        exp_w * q.z() * sinc_vlen
    );
}

// Quaternion logarithm: log(q)
// For unit quaternion q = (cos θ, v̂ sin θ): log(q) = (0, v̂ θ)
// For general quaternion: log(q) = (log|q|, v̂ θ)
template<typename T>
inline quaternion<T> log(const quaternion<T>& q) {
    #ifdef EULER_DEBUG
    T qlen = q.length();
    EULER_CHECK(qlen > constants<T>::epsilon, error_code::invalid_argument,
                "quaternion::log: cannot take log of zero quaternion");
    #else
    T qlen = q.length();
    #endif
    
    T vlen = sqrt(q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
    
    if (vlen < constants<T>::epsilon) {
        // Near-scalar quaternion
        return quaternion<T>(log(qlen), T(0), T(0), T(0));
    }
    
    T theta = atan2(vlen, q.w());
    T scale = theta / vlen;
    
    return quaternion<T>(
        log(qlen),
        q.x() * scale,
        q.y() * scale,
        q.z() * scale
    );
}

// Quaternion power: q^t
// Formula: q^t = exp(t * log(q))
template<typename T>
inline quaternion<T> pow(const quaternion<T>& q, T t) {
    // Special cases
    if (approx_equal(t, T(0))) {
        return quaternion<T>::identity();
    }
    if (approx_equal(t, T(1))) {
        return q;
    }
    
    // For unit quaternions, we can use a more efficient formula
    if (q.is_normalized()) {
        radian<T> theta = q.angle();
        vector<T, 3> axis = q.axis();
        return quaternion<T>::from_axis_angle(axis, theta * t);
    }
    
    // General case: q^t = exp(t * log(q))
    return exp(log(q) * t);
}

// ============================================================================
// Linear Interpolation
// ============================================================================

// Scalar linear interpolation helper
template<typename T>
inline T lerp_scalar(T a, T b, T t) {
    return a + t * (b - a);
}

// Linear interpolation (lerp)
// Note: Result is normalized to maintain unit quaternion property
template<typename T>
inline quaternion<T> lerp(const quaternion<T>& q1, const quaternion<T>& q2, T t) {
    EULER_CHECK(t >= T(0) && t <= T(1), error_code::invalid_argument,
                "quaternion::lerp: t must be in [0, 1]");
    
    // Ensure shortest path
    T d = dot(q1, q2);
    quaternion<T> q2_adjusted = d < T(0) ? -q2 : q2;
    
    quaternion<T> result(
        lerp_scalar(q1.w(), q2_adjusted.w(), t),
        lerp_scalar(q1.x(), q2_adjusted.x(), t),
        lerp_scalar(q1.y(), q2_adjusted.y(), t),
        lerp_scalar(q1.z(), q2_adjusted.z(), t)
    );
    
    return result.normalized();
}

// ============================================================================
// Spherical Linear Interpolation
// ============================================================================

// Spherical linear interpolation (slerp)
// Interpolates along the shortest great circle arc
template<typename T>
inline quaternion<T> slerp(const quaternion<T>& q1, const quaternion<T>& q2, T t) {
    EULER_CHECK(t >= T(0) && t <= T(1), error_code::invalid_argument,
                "quaternion::slerp: t must be in [0, 1]");
    
    T cos_theta = dot(q1, q2);
    
    // Take shortest path
    quaternion<T> q2_adjusted = q2;
    if (cos_theta < T(0)) {
        q2_adjusted = -q2;
        cos_theta = -cos_theta;
    }
    
    // Use lerp for very similar quaternions (avoids division by small numbers)
    if (cos_theta > T(0.995)) {
        return lerp(q1, q2_adjusted, t);
    }
    
    // Clamp to handle numerical errors
    cos_theta = clamp(cos_theta, T(-1), T(1));

    // Calculate coefficients
    T theta = acos(cos_theta);
    T sin_theta = sin(theta);

    // Safety check: if sin_theta is too small, fall back to lerp
    // (this shouldn't happen given the 0.995 threshold, but guards against edge cases)
    if (sin_theta < constants<T>::epsilon) {
        return lerp(q1, q2_adjusted, t);
    }

    T s1 = sin((T(1) - t) * theta) / sin_theta;
    T s2 = sin(t * theta) / sin_theta;
    
    return quaternion<T>(
        s1 * q1.w() + s2 * q2_adjusted.w(),
        s1 * q1.x() + s2 * q2_adjusted.x(),
        s1 * q1.y() + s2 * q2_adjusted.y(),
        s1 * q1.z() + s2 * q2_adjusted.z()
    );
}

// ============================================================================
// Normalized Linear Interpolation (Faster Alternative to Slerp)
// ============================================================================

// Normalized linear interpolation (nlerp)
// Faster than slerp but doesn't maintain constant angular velocity
template<typename T>
inline quaternion<T> nlerp(const quaternion<T>& q1, const quaternion<T>& q2, T t) {
    return lerp(q1, q2, t);  // lerp already normalizes
}

// ============================================================================
// Squad Interpolation (Smooth Interpolation Through Multiple Quaternions)
// ============================================================================

// Helper for squad: compute intermediate quaternion
template<typename T>
inline quaternion<T> squad_intermediate(const quaternion<T>& q_prev,
                                       const quaternion<T>& q_curr,
                                       const quaternion<T>& q_next) {
    quaternion<T> q_curr_inv = inverse(q_curr);
    quaternion<T> log_prev = log(q_curr_inv * q_prev);
    quaternion<T> log_next = log(q_curr_inv * q_next);
    
    quaternion<T> sum = (log_prev + log_next) * T(-0.25);
    return q_curr * exp(sum);
}

// Spherical quadrangle interpolation (squad)
// Provides C¹ continuous interpolation through quaternion keyframes
template<typename T>
inline quaternion<T> squad(const quaternion<T>& q1, const quaternion<T>& a,
                          const quaternion<T>& b, const quaternion<T>& q2,
                          T t) {
    quaternion<T> slerp1 = slerp(q1, q2, t);
    quaternion<T> slerp2 = slerp(a, b, t);
    return slerp(slerp1, slerp2, T(2) * t * (T(1) - t));
}

// ============================================================================
// Vector Rotation
// ============================================================================

// Rotate a vector by a quaternion (free function version)
template<typename T>
inline vector<T, 3> rotate(const vector<T, 3>& v, const quaternion<T>& q) {
    return q.rotate(v);
}

// ============================================================================
// Utility Functions
// ============================================================================

// Normalize a quaternion (alternative to member function)
template<typename T>
inline quaternion<T> normalize(const quaternion<T>& q) {
    return q.normalized();
}

// Check if two quaternions represent the same rotation
// (considering that q and -q represent the same rotation)
template<typename T>
inline bool same_rotation(const quaternion<T>& q1, const quaternion<T>& q2) {
    return same_rotation(q1, q2, constants<T>::epsilon);
}

template<typename T>
inline bool same_rotation(const quaternion<T>& q1, const quaternion<T>& q2,
                         T tolerance) {
    return approx_equal(q1, q2, tolerance) || approx_equal(q1, -q2, tolerance);
}

// Compute angular difference between two quaternions
template<typename T>
inline radian<T> angle_between(const quaternion<T>& q1, const quaternion<T>& q2) {
    // Ensure both are normalized
    quaternion<T> q1_norm = q1.normalized();
    quaternion<T> q2_norm = q2.normalized();
    
    // Compute relative rotation: q_rel = q1^(-1) * q2
    quaternion<T> q_rel = inverse(q1_norm) * q2_norm;
    
    // Extract angle from relative rotation
    return q_rel.angle();
}

// Create a quaternion that rotates from one direction to another
// (Alternative to quaternion::from_vectors for convenience)
template<typename T>
inline quaternion<T> rotation_between(const vector<T, 3>& from, 
                                     const vector<T, 3>& to) {
    return quaternion<T>::from_vectors(from, to);
}

// ============================================================================
// Expression Template Integration
// ============================================================================

// Make quaternion work with expression templates for component-wise operations
template<typename T>
struct expression_traits<quaternion<T>> {
    using value_type = T;
    static constexpr size_t rows = 4;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

// Component-wise operations for quaternion expressions
template<typename T>
inline auto make_quaternion_expression(const quaternion<T>& q) {
    return make_vector_expression(
        vector<T, 4>(q.w(), q.x(), q.y(), q.z())
    );
}

} // namespace euler