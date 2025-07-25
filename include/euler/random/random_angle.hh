#pragma once

#include <euler/core/types.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/math/basic.hh>
#include <euler/random/random.hh>
#include <algorithm>
#include <vector>

namespace euler {

// ============================================================================
// Random Angle Generation
// ============================================================================

// Generate random angle in range [min, max]
template<typename T, typename Unit, typename Generator>
angle<T, Unit> random_angle(Generator& g, 
                           const angle<T, Unit>& min,
                           const angle<T, Unit>& max) {
    T value = g.uniform(min.value(), max.value());
    return angle<T, Unit>(value);
}

// Generate random angle in [0, 2π) or [0, 360°)
template<typename T, typename Unit, typename Generator>
angle<T, Unit> random_angle(Generator& g) {
    if constexpr (std::is_same_v<Unit, degree_tag>) {
        return angle<T, Unit>(g.uniform(T(0), T(360)));
    } else {
        return angle<T, Unit>(g.uniform(T(0), T(2) * constants<T>::pi));
    }
}

// Generate random angle with normal distribution
template<typename T, typename Unit, typename Generator>
angle<T, Unit> random_angle_normal(Generator& g, 
                                  const angle<T, Unit>& mean,
                                  const angle<T, Unit>& stddev) {
    T value = g.normal(mean.value(), stddev.value());
    
    // Wrap to valid range
    if constexpr (std::is_same_v<Unit, degree_tag>) {
        value = mod(value, T(360));
        if (value < T(0)) value += T(360);
    } else {
        value = mod(value, T(2) * constants<T>::pi);
        if (value < T(0)) value += T(2) * constants<T>::pi;
    }
    
    return angle<T, Unit>(value);
}

// Generate random angle with von Mises distribution (circular normal)
template<typename T, typename Unit, typename Generator>
angle<T, Unit> random_angle_von_mises(Generator& g,
                                     const angle<T, Unit>& mean,
                                     T kappa) {
    // Von Mises distribution approximation
    // For large kappa (>10), use normal approximation
    if (kappa > T(10)) {
        T stddev_rad = T(1) / sqrt(kappa);
        if constexpr (std::is_same_v<Unit, degree_tag>) {
            T stddev_deg = stddev_rad * T(180) / constants<T>::pi;
            return random_angle_normal(g, mean, angle<T, Unit>(stddev_deg));
        } else {
            return random_angle_normal(g, mean, angle<T, Unit>(stddev_rad));
        }
    }
    
    // For small kappa, use rejection sampling
    T a = T(1) + sqrt(T(1) + T(4) * kappa * kappa);
    T b = (a - sqrt(T(2) * a)) / (T(2) * kappa);
    T r = (T(1) + b * b) / (T(2) * b);
    
    T theta_rad;
    while (true) {
        T u1 = g.template uniform<T>();
        T z = static_cast<T>(cos(constants<T>::pi * u1));
        T f = (T(1) + r * z) / (r + z);
        T c = kappa * (r - f);
        
        T u2 = g.template uniform<T>();
        if (u2 < c * (T(2) - c) || u2 <= c * exp(T(1) - c)) {
            T u3 = g.template uniform<T>();
            theta_rad = (u3 < T(0.5)) ? static_cast<T>(acos(f)) : static_cast<T>(-acos(f));
            break;
        }
    }
    
    // Add mean and convert to appropriate unit
    if constexpr (std::is_same_v<Unit, degree_tag>) {
        T mean_rad = mean.value() * constants<T>::pi / T(180);
        T result_rad = theta_rad + mean_rad;
        T result_deg = result_rad * T(180) / constants<T>::pi;
        result_deg = mod(result_deg, T(360));
        if (result_deg < T(0)) result_deg += T(360);
        return angle<T, Unit>(result_deg);
    } else {
        T result_rad = theta_rad + mean.value();
        result_rad = mod(result_rad, T(2) * constants<T>::pi);
        if (result_rad < T(0)) result_rad += T(2) * constants<T>::pi;
        return angle<T, Unit>(result_rad);
    }
}

// Generate multiple random angles that sum to a specific value
template<typename T, typename Unit, typename Generator>
std::vector<angle<T, Unit>> random_angles_constrained_sum(
    Generator& g,
    size_t count,
    const angle<T, Unit>& sum) {
    
    if (count == 0) {
        return {};
    }
    
    if (count == 1) {
        return {sum};
    }
    
    // Generate count-1 random values in [0, 1]
    std::vector<T> cuts;
    cuts.reserve(count + 1);
    cuts.push_back(T(0));
    
    for (size_t i = 0; i < count - 1; ++i) {
        cuts.push_back(g.template uniform<T>());
    }
    cuts.push_back(T(1));
    
    // Sort the cuts
    std::sort(cuts.begin(), cuts.end());
    
    // Calculate differences and scale by sum
    std::vector<angle<T, Unit>> angles;
    angles.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        T fraction = cuts[i + 1] - cuts[i];
        angles.push_back(angle<T, Unit>(sum.value() * fraction));
    }
    
    return angles;
}

// ============================================================================
// Convenience functions using thread-local generator
// ============================================================================

template<typename T, typename Unit>
inline angle<T, Unit> random_angle() {
    return random_angle<T, Unit>(thread_local_rng());
}

template<typename T, typename Unit>
inline angle<T, Unit> random_angle(const angle<T, Unit>& min,
                                  const angle<T, Unit>& max) {
    return random_angle(thread_local_rng(), min, max);
}

template<typename T, typename Unit>
inline angle<T, Unit> random_angle_normal(const angle<T, Unit>& mean,
                                         const angle<T, Unit>& stddev) {
    return random_angle_normal(thread_local_rng(), mean, stddev);
}

} // namespace euler