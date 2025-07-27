/**
 * @file radian.hh
 * @brief Radian angle specialization and utilities
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides convenient type aliases and utilities specifically
 * for working with angles in radians. It includes:
 * - Type aliases for common radian types
 * - User-defined literals for creating radian angles
 * - Specialized wrapping functions optimized for radians
 * - Interpolation functions that handle wrap-around correctly
 * - Common radian angle constants (π, π/2, 2π, etc.)
 * 
 * @section radian_usage Usage Example
 * @code{.cpp}
 * #include <euler/angles/radian.hh>
 * 
 * using namespace euler;
 * using namespace euler::literals;
 * using namespace euler::angle_constants;
 * 
 * // Create radian angles using literals
 * auto angle1 = 1.57_rad;   // ≈ π/2 radians
 * auto angle2 = 3.14_radd;  // ≈ π radians (double precision)
 * 
 * // Use predefined constants
 * auto right_angle = half_pi_rad<float>;
 * auto full_circle = two_pi_rad<float>;
 * 
 * // Wrapping
 * auto wrapped = wrap(radian<float>(7.0f));  // ≈ 0.717 rad
 * auto positive = wrap_positive(radian<float>(-1.0f));  // ≈ 5.28 rad
 * 
 * // Interpolation with proper wrap-around
 * auto start = 0.1_rad;
 * auto end = 6.1_rad;  // Close to 2π
 * auto mid = lerp(start, end, 0.5f);  // ≈ 0.0 rad (not 3.1 rad!)
 * @endcode
 */
#pragma once

#include <euler/angles/angle.hh>
#include <euler/angles/angle_common.hh>
#include <cmath>

namespace euler {

/**
 * @brief Type alias for radian angles
 * @ingroup AnglesModule
 * @tparam T The numeric type (e.g., float, double)
 * 
 * @details
 * This alias provides a more intuitive name for angles measured in radians.
 * It's equivalent to `angle<T, radian_tag>` but more readable.
 * 
 * @see angle
 */
template<typename T>
using radian = angle<T, radian_tag>;

/**
 * @brief Single-precision radian angle
 * @ingroup AnglesModule
 * 
 * @details
 * Convenient type alias for `radian<float>`. This is the most commonly
 * used radian type for graphics, physics simulations, and game programming.
 */
using radianf = radian<float>;

/**
 * @brief Double-precision radian angle
 * @ingroup AnglesModule
 * 
 * @details
 * Convenient type alias for `radian<double>`. Use this when you need
 * higher precision for scientific or engineering calculations.
 */
using radiand = radian<double>;

/**
 * @brief User-defined literals for radian angles
 * @ingroup AnglesModule
 * 
 * @details
 * This inline namespace provides user-defined literal operators for
 * creating radian angles with intuitive syntax. The literals are:
 * - `_rad` - Creates a `radianf` (single precision)
 * - `_radf` - Explicitly creates a `radianf`
 * - `_radd` - Explicitly creates a `radiand` (double precision)
 */
inline namespace literals {
    /**
     * @brief Create single-precision radian angle from floating-point literal
     * @param value The angle value in radians
     * @return A radianf angle
     * @note Example: `auto angle = 3.14159_rad;`
     */
    constexpr radianf operator""_rad(long double value) {
        return radianf(static_cast<float>(value));
    }
    
    /**
     * @brief Explicitly create single-precision radian angle
     * @param value The angle value in radians
     * @return A radianf angle
     * @note Example: `auto angle = 1.5708_radf;`
     */
    constexpr radianf operator""_radf(long double value) {
        return radianf(static_cast<float>(value));
    }
    
    /**
     * @brief Create double-precision radian angle from floating-point literal
     * @param value The angle value in radians
     * @return A radiand angle
     * @note Example: `auto angle = 3.141592653589793_radd;`
     */
    constexpr radiand operator""_radd(long double value) {
        return radiand(static_cast<double>(value));
    }
    
    /**
     * @brief Create single-precision radian angle from integer literal
     * @param value The angle value in radians
     * @return A radianf angle
     * @note Example: `auto angle = 2_rad;`
     * @warning Integer radians are rarely useful - consider using floating-point
     */
    constexpr radianf operator""_rad(unsigned long long value) {
        return radianf(static_cast<float>(value));
    }
    
    /**
     * @brief Explicitly create single-precision radian angle from integer
     * @param value The angle value in radians
     * @return A radianf angle
     * @note Example: `auto angle = 1_radf;`
     */
    constexpr radianf operator""_radf(unsigned long long value) {
        return radianf(static_cast<float>(value));
    }
    
    /**
     * @brief Create double-precision radian angle from integer literal
     * @param value The angle value in radians
     * @return A radiand angle
     * @note Example: `auto angle = 3_radd;`
     */
    constexpr radiand operator""_radd(unsigned long long value) {
        return radiand(static_cast<double>(value));
    }
}

/**
 * @brief Wrap radian angle to [-π, π] range
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param r The radian angle to wrap
 * @return Wrapped angle in range [-π, π]
 * 
 * @details
 * This specialization efficiently wraps radian angles to the principal
 * value range. It's optimized for radians, using the exact π value for
 * the numeric type.
 * 
 * @note The implementation uses a simple loop which is efficient for
 *       angles that are already close to the target range
 * 
 * @see wrap_positive() for [0, 2π) range
 */
template<typename T>
radian<T> wrap(const radian<T>& r) {
    const T two_pi_val = T(2) * constants<T>::pi;
    return radian<T>(detail::wrap_angle_value(r.value(), constants<T>::pi, two_pi_val));
}

/**
 * @brief Wrap radian angle to [0, 2π) range
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param r The radian angle to wrap
 * @return Wrapped angle in range [0, 2π)
 * 
 * @details
 * This specialization efficiently wraps radian angles to the positive
 * range. Useful when negative angles are not desired, such as for
 * phase calculations or polar coordinates.
 * 
 * @note The range is [0, 2π), meaning 2π itself wraps to 0
 * 
 * @see wrap() for [-π, π] range
 */
template<typename T>
radian<T> wrap_positive(const radian<T>& r) {
    const T two_pi_val = T(2) * constants<T>::pi;
    return radian<T>(detail::wrap_angle_positive_value(r.value(), two_pi_val));
}

/**
 * @brief Linear interpolation between radian angles
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param a Starting angle
 * @param b Ending angle
 * @param t Interpolation parameter [0, 1]
 * @return Interpolated angle
 * 
 * @details
 * This function performs linear interpolation between two radian angles,
 * automatically choosing the shortest path around the circle. This means
 * interpolating from 6.0 rad to 0.5 rad will go through 0 (not through π).
 * 
 * @note
 * - When t = 0, returns angle a
 * - When t = 1, returns angle b
 * - When t = 0.5, returns the angle halfway along the shortest arc
 * 
 * @warning The interpolation parameter t is typically in [0, 1], but
 *          values outside this range will extrapolate
 * 
 * @example
 * @code
 * auto start = 6.1_rad;  // Close to 2π
 * auto end = 0.2_rad;
 * auto mid = lerp(start, end, 0.5f);  // Returns ≈ 0.15 rad, not 3.15 rad
 * @endcode
 */
template<typename T>
radian<T> lerp(const radian<T>& a, const radian<T>& b, T t) {
    const T two_pi_val = T(2) * constants<T>::pi;
    return radian<T>(detail::lerp_angle_value(a.value(), b.value(), t, constants<T>::pi, two_pi_val));
}

/**
 * @brief Common radian angle constants
 * @ingroup AnglesModule
 * 
 * @details
 * This namespace provides commonly used radian angle constants as
 * compile-time values. Each constant is a template to allow choosing
 * the desired numeric precision.
 */
namespace angle_constants {
    /**
     * @brief π radians (180 degrees)
     * @tparam T The numeric type (default: float)
     */
    template<typename T = float>
    inline constexpr radian<T> pi_rad(constants<T>::pi);
    
    /**
     * @brief π/2 radians (90 degrees)
     * @tparam T The numeric type (default: float)
     */
    template<typename T = float>
    inline constexpr radian<T> half_pi_rad(constants<T>::half_pi);
    
    /**
     * @brief 2π radians (360 degrees, full circle)
     * @tparam T The numeric type (default: float)
     */
    template<typename T = float>
    inline constexpr radian<T> two_pi_rad(T(2) * constants<T>::pi);
    
    /**
     * @brief π/4 radians (45 degrees)
     * @tparam T The numeric type (default: float)
     */
    template<typename T = float>
    inline constexpr radian<T> quarter_pi_rad(constants<T>::pi / T(4));
}

} // namespace euler