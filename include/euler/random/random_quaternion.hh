#pragma once

#include <euler/core/types.hh>
#include <euler/quaternion/quaternion.hh>
#include <euler/random/random.hh>
#include <euler/random/random_angle.hh>
#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>

namespace euler {

// Forward declarations to avoid circular dependency
template<typename T, size_t N, typename Generator>
vector<T, N> random_unit_vector(Generator& g);

// ============================================================================
// Random Quaternion Generation
// ============================================================================

// Generate uniformly distributed quaternion on SO(3)
// Uses Shoemake's method for uniform quaternion generation
template<typename T, typename Generator>
quaternion<T> random_quaternion(Generator& g) {
    // Generate three uniform random numbers
    T u1 = g.template uniform<T>(T(0), T(1));
    T u2 = g.template uniform<T>(T(0), T(1));
    T u3 = g.template uniform<T>(T(0), T(1));
    
    // Shoemake's algorithm
    T sqrt_1_u1 = sqrt(T(1) - u1);
    T sqrt_u1 = sqrt(u1);
    T two_pi_u2 = T(2) * constants<T>::pi * u2;
    T two_pi_u3 = T(2) * constants<T>::pi * u3;
    
    return quaternion<T>(
        sqrt_u1 * cos(two_pi_u3),      // w
        sqrt_1_u1 * sin(two_pi_u2),    // x
        sqrt_1_u1 * cos(two_pi_u2),    // y
        sqrt_u1 * sin(two_pi_u3)       // z
    );
}

// Generate quaternion with limited rotation angle
template<typename T, typename Generator>
quaternion<T> random_quaternion(Generator& g, const angle<T, radian_tag>& max_angle) {
    // Generate random axis
    vector<T, 3> axis = random_unit_vector<T, 3>(g);
    
    // Generate random angle up to max_angle
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g, 
                                                            angle<T, radian_tag>(0), 
                                                            max_angle);
    
    return quaternion<T>::from_axis_angle(axis, theta);
}

// Generate quaternion near identity (small rotation)
template<typename T, typename Generator>
quaternion<T> random_quaternion_small(Generator& g, T max_angle_rad = T(0.1)) {
    vector<T, 3> axis = random_unit_vector<T, 3>(g);
    T angle = g.uniform(T(0), max_angle_rad);
    return quaternion<T>::from_axis_angle(axis, radian<T>(angle));
}

// Generate quaternion with normal distribution around identity
template<typename T, typename Generator>
quaternion<T> random_quaternion_normal(Generator& g, T stddev_rad = T(0.1)) {
    // Generate small rotation vector
    T x = g.normal(T(0), stddev_rad);
    T y = g.normal(T(0), stddev_rad);
    T z = g.normal(T(0), stddev_rad);
    
    // Convert to quaternion using exponential map
    T angle = sqrt(x*x + y*y + z*z);
    
    if (angle < constants<T>::epsilon) {
        // Very small angle - use first-order approximation
        return quaternion<T>(T(1), x/T(2), y/T(2), z/T(2)).normalized();
    }
    
    T half_angle = angle / T(2);
    T sinc_half = sin(half_angle) / angle;
    
    return quaternion<T>(
        cos(half_angle),
        x * sinc_half,
        y * sinc_half,
        z * sinc_half
    );
}

// Generate quaternion that rotates around specific axis
template<typename T, typename Generator>
quaternion<T> random_quaternion_axis(Generator& g, 
                                    const vector<T, 3>& axis,
                                    const angle<T, radian_tag>& min_angle = angle<T, radian_tag>(0),
                                    const angle<T, radian_tag>& max_angle = angle<T, radian_tag>(2 * constants<T>::pi)) {
    #ifdef EULER_DEBUG
    EULER_CHECK(approx_equal(axis.length_squared(), T(1), T(1e-6)),
                error_code::invalid_argument,
                "random_quaternion_axis: axis must be normalized");
    #endif
    
    angle<T, radian_tag> theta = random_angle<T, radian_tag>(g, min_angle, max_angle);
    return quaternion<T>::from_axis_angle(axis, theta);
}

// Generate quaternion in specific Euler angle ranges
template<typename T, typename Unit, typename Generator>
quaternion<T> random_quaternion_euler(Generator& g,
                                     const angle<T, Unit>& min_roll,
                                     const angle<T, Unit>& max_roll,
                                     const angle<T, Unit>& min_pitch,
                                     const angle<T, Unit>& max_pitch,
                                     const angle<T, Unit>& min_yaw,
                                     const angle<T, Unit>& max_yaw,
                                     euler_order order = euler_order::XYZ) {
    angle<T, Unit> roll = random_angle<T, Unit>(g, min_roll, max_roll);
    angle<T, Unit> pitch = random_angle<T, Unit>(g, min_pitch, max_pitch);
    angle<T, Unit> yaw = random_angle<T, Unit>(g, min_yaw, max_yaw);
    
    return quaternion<T>::from_euler(roll, pitch, yaw, order);
}

// Generate quaternion that satisfies certain constraints
template<typename T, typename Generator>
quaternion<T> random_quaternion_constrained(Generator& g,
                                           const vector<T, 3>& forward_dir,
                                           T max_deviation_rad) {
    // Generate a quaternion that keeps the forward direction within max_deviation
    // of the specified forward_dir
    
    // First, find quaternion that rotates default forward (0,0,1) to forward_dir
    vector<T, 3> default_forward(0, 0, 1);
    quaternion<T> base_rotation = quaternion<T>::from_vectors(default_forward, forward_dir);
    
    // Then add small random rotation
    quaternion<T> small_rotation = random_quaternion_small(g, max_deviation_rad);
    
    // Compose rotations
    return base_rotation * small_rotation;
}

// Generate set of quaternions with specific distribution of rotations
template<typename T, typename Generator>
std::vector<quaternion<T>> random_quaternions_distributed(Generator& g,
                                                         size_t count,
                                                         T clustering = T(1)) {
    // clustering > 1: more clustered around identity
    // clustering < 1: more spread out
    // clustering = 1: uniform distribution
    
    std::vector<quaternion<T>> quats;
    quats.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        if (clustering == T(1)) {
            quats.push_back(random_quaternion<T>(g));
        } else {
            // Use power distribution for angle
            T u = g.template uniform<T>(T(0), T(1));
            T angle_fraction = pow(u, clustering);
            angle<T, radian_tag> angle(angle_fraction * constants<T>::pi);
            quats.push_back(random_quaternion(g, angle));
        }
    }
    
    return quats;
}

// ============================================================================
// Convenience functions using thread-local generator
// ============================================================================

template<typename T>
inline quaternion<T> random_quaternion() {
    return random_quaternion<T>(thread_local_rng());
}

template<typename T>
inline quaternion<T> random_quaternion(const angle<T, radian_tag>& max_angle) {
    return random_quaternion(thread_local_rng(), max_angle);
}

template<typename T>
inline quaternion<T> random_quaternion_small(T max_angle_rad = T(0.1)) {
    return random_quaternion_small(thread_local_rng(), max_angle_rad);
}

template<typename T>
inline quaternion<T> random_quaternion_normal(T stddev_rad = T(0.1)) {
    return random_quaternion_normal(thread_local_rng(), stddev_rad);
}

} // namespace euler