/**
 * @file degree.hh
 * @brief Degree angle specialization and utilities
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides convenient type aliases and utilities specifically
 * for working with angles in degrees. It includes:
 * - Type aliases for common degree types
 * - User-defined literals for creating degree angles
 * - Specialized wrapping functions optimized for degrees
 * - Interpolation functions that handle wrap-around correctly
 * 
 * @section degree_usage Usage Example
 * @code{.cpp}
 * #include <euler/angles/degree.hh>
 * 
 * using namespace euler;
 * using namespace euler::literals;
 * 
 * // Create degree angles using literals
 * auto angle1 = 45.0_deg;   // degreef
 * auto angle2 = 90.0_degd;  // degreed
 * 
 * // Arithmetic operations
 * auto sum = angle1 + 30.0_deg;  // 75°
 * 
 * // Wrapping
 * auto wrapped = wrap(degree<float>(370.0f));  // 10°
 * auto positive = wrap_positive(degree<float>(-30.0f));  // 330°
 * 
 * // Interpolation with proper wrap-around
 * auto start = 350.0_deg;
 * auto end = 10.0_deg;
 * auto mid = lerp(start, end, 0.5f);  // 0° (not 180°!)
 * @endcode
 */
#pragma once

#include <euler/angles/angle.hh>
#include <euler/angles/angle_common.hh>
#include <cmath>

namespace euler {

/**
 * @brief Type alias for degree angles
 * @ingroup AnglesModule
 * @tparam T The numeric type (e.g., float, double)
 * 
 * @details
 * This alias provides a more intuitive name for angles measured in degrees.
 * It's equivalent to `angle<T, degree_tag>` but more readable.
 * 
 * @see angle
 */
template<typename T>
using degree = angle<T, degree_tag>;

/**
 * @brief Single-precision degree angle
 * @ingroup AnglesModule
 * 
 * @details
 * Convenient type alias for `degree<float>`. This is the most commonly
 * used degree type for graphics and game programming.
 */
using degreef = degree<float>;

/**
 * @brief Double-precision degree angle
 * @ingroup AnglesModule
 * 
 * @details
 * Convenient type alias for `degree<double>`. Use this when you need
 * higher precision for scientific or engineering calculations.
 */
using degreed = degree<double>;

/**
 * @brief User-defined literals for degree angles
 * @ingroup AnglesModule
 * 
 * @details
 * This inline namespace provides user-defined literal operators for
 * creating degree angles with intuitive syntax. The literals are:
 * - `_deg` - Creates a `degreef` (single precision)
 * - `_degf` - Explicitly creates a `degreef`
 * - `_degd` - Explicitly creates a `degreed` (double precision)
 */
inline namespace literals {
    /**
     * @brief Create single-precision degree angle from floating-point literal
     * @param value The angle value in degrees
     * @return A degreef angle
     * @note Example: `auto angle = 45.5_deg;`
     */
    constexpr degreef operator""_deg(long double value) {
        return degreef(static_cast<float>(value));
    }
    
    /**
     * @brief Explicitly create single-precision degree angle
     * @param value The angle value in degrees
     * @return A degreef angle
     * @note Example: `auto angle = 45.5_degf;`
     */
    constexpr degreef operator""_degf(long double value) {
        return degreef(static_cast<float>(value));
    }
    
    /**
     * @brief Create double-precision degree angle from floating-point literal
     * @param value The angle value in degrees
     * @return A degreed angle
     * @note Example: `auto angle = 45.5_degd;`
     */
    constexpr degreed operator""_degd(long double value) {
        return degreed(static_cast<double>(value));
    }
    
    /**
     * @brief Create single-precision degree angle from integer literal
     * @param value The angle value in degrees
     * @return A degreef angle
     * @note Example: `auto angle = 45_deg;`
     */
    constexpr degreef operator""_deg(unsigned long long value) {
        return degreef(static_cast<float>(value));
    }
    
    /**
     * @brief Explicitly create single-precision degree angle from integer
     * @param value The angle value in degrees
     * @return A degreef angle
     * @note Example: `auto angle = 45_degf;`
     */
    constexpr degreef operator""_degf(unsigned long long value) {
        return degreef(static_cast<float>(value));
    }
    
    /**
     * @brief Create double-precision degree angle from integer literal
     * @param value The angle value in degrees
     * @return A degreed angle
     * @note Example: `auto angle = 45_degd;`
     */
    constexpr degreed operator""_degd(unsigned long long value) {
        return degreed(static_cast<double>(value));
    }
}

/**
 * @brief Wrap degree angle to [-180°, 180°] range
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param d The degree angle to wrap
 * @return Wrapped angle in range [-180°, 180°]
 * 
 * @details
 * This specialization efficiently wraps degree angles to the principal
 * value range. It's optimized for degrees, avoiding unnecessary conversions
 * to radians.
 * 
 * @note The implementation uses a simple loop which is efficient for
 *       angles that are already close to the target range
 * 
 * @see wrap_positive() for [0°, 360°) range
 */
template<typename T>
degree<T> wrap(const degree<T>& d) {
    return degree<T>(detail::wrap_angle_value(d.value(), T(180), T(360)));
}

/**
 * @brief Wrap degree angle to [0°, 360°) range
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param d The degree angle to wrap
 * @return Wrapped angle in range [0°, 360°)
 * 
 * @details
 * This specialization efficiently wraps degree angles to the positive
 * range. Useful when negative angles are not desired, such as for
 * compass headings or user interface elements.
 * 
 * @note The range is [0°, 360°), meaning 360° itself wraps to 0°
 * 
 * @see wrap() for [-180°, 180°] range
 */
template<typename T>
degree<T> wrap_positive(const degree<T>& d) {
    return degree<T>(detail::wrap_angle_positive_value(d.value(), T(360)));
}

/**
 * @brief Linear interpolation between degree angles
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param a Starting angle
 * @param b Ending angle
 * @param t Interpolation parameter [0, 1]
 * @return Interpolated angle
 * 
 * @details
 * This function performs linear interpolation between two degree angles,
 * automatically choosing the shortest path around the circle. This means
 * interpolating from 350° to 10° will go through 0° (not through 180°).
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
 * auto start = 350.0_deg;
 * auto end = 10.0_deg;
 * auto mid = lerp(start, end, 0.5f);  // Returns 0°, not 180°
 * @endcode
 */
template<typename T>
degree<T> lerp(const degree<T>& a, const degree<T>& b, T t) {
    return degree<T>(detail::lerp_angle_value(a.value(), b.value(), t, T(180), T(360)));
}

} // namespace euler