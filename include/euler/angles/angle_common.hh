/**
 * @file angle_common.hh
 * @brief Common implementation details for angle operations
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides common implementation functions to reduce
 * code duplication between degree and radian specializations.
 */
#pragma once

#include <euler/angles/angle.hh>
#include <cmath>

namespace euler {
namespace detail {

/**
 * @brief Generic angle wrapping to principal range
 * @tparam T Numeric type
 * @tparam Unit Angle unit tag
 * @param value The angle value to wrap
 * @param half_period Half the period (180 for degrees, π for radians)
 * @param full_period Full period (360 for degrees, 2π for radians)
 * @return Wrapped value in range [-half_period, half_period]
 */
template<typename T>
inline T wrap_angle_value(T value, T half_period, T full_period) {
    // Use modulo for O(1) performance
    value = std::fmod(value + half_period, full_period);
    if (value <= T(0)) {
        value += full_period;
    }
    return value - half_period;
}

/**
 * @brief Generic angle wrapping to positive range
 * @tparam T Numeric type
 * @param value The angle value to wrap
 * @param full_period Full period (360 for degrees, 2π for radians)
 * @return Wrapped value in range [0, full_period)
 */
template<typename T>
inline T wrap_angle_positive_value(T value, T full_period) {
    // Use modulo for O(1) performance
    value = std::fmod(value, full_period);
    if (value < T(0)) {
        value += full_period;
    }
    return value;
}

/**
 * @brief Generic angle linear interpolation
 * @tparam T Numeric type
 * @param a Start angle value
 * @param b End angle value
 * @param t Interpolation parameter [0, 1]
 * @param half_period Half the period (180 for degrees, π for radians)
 * @param full_period Full period (360 for degrees, 2π for radians)
 * @return Interpolated angle value
 */
template<typename T>
inline T lerp_angle_value(T a, T b, T t, T half_period, T full_period) {
    // Handle wrap-around for shortest path
    T diff = b - a;
    
    // Wrap difference to [-half_period, half_period]
    while (diff > half_period) {
        diff -= full_period;
    }
    while (diff <= -half_period) {
        diff += full_period;
    }
    
    return a + diff * t;
}

} // namespace detail
} // namespace euler