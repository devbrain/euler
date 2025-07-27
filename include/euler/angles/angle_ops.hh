/**
 * @file angle_ops.hh
 * @brief Additional operations and utilities for angle types
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides additional operations for angle types that don't
 * fit naturally as member functions. It includes:
 * - Conversion functions between degrees and radians
 * - Mathematical operations (abs, min, max, clamp)
 * - Angle difference calculations (shortest path)
 * - Conversion to/from Cartesian components
 * - Utility functions for angle manipulation
 * 
 * @section angle_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/angles/angle_ops.hh>
 * 
 * using namespace euler;
 * 
 * // Convert between units
 * auto deg = degree<float>(45.0f);
 * auto rad = to_radians(deg);  // π/4
 * 
 * // Calculate shortest angle difference
 * auto a = degree<float>(350.0f);
 * auto b = degree<float>(10.0f);
 * auto diff = angle_difference(a, b);  // 20°, not 340°
 * 
 * // Convert to unit vector components
 * auto components = angle_to_components(radian<float>(π/4));
 * // components.cos ≈ 0.707, components.sin ≈ 0.707
 * 
 * // Create angle from vector components
 * auto angle = angle_from_components(1.0f, 1.0f);  // π/4 rad
 * @endcode
 */
#pragma once

#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_traits.hh>
#include <cmath>

namespace euler {

/**
 * @brief Convert degrees to radians
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param d Angle in degrees
 * @return Angle in radians
 * 
 * @details
 * This function provides an explicit conversion from degrees to radians.
 * It's more readable than using the angle constructor directly.
 * 
 * @example
 * @code
 * auto deg = degree<float>(45.0f);
 * auto rad = to_radians(deg);  // π/4 radians
 * @endcode
 */
template<typename T>
constexpr radian<T> to_radians(const degree<T>& d) {
    return radian<T>(d);
}

/**
 * @brief Convert radians to degrees
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param r Angle in radians
 * @return Angle in degrees
 * 
 * @details
 * This function provides an explicit conversion from radians to degrees.
 * It's more readable than using the angle constructor directly.
 * 
 * @example
 * @code
 * auto rad = radian<float>(π/4);
 * auto deg = to_degrees(rad);  // 45°
 * @endcode
 */
template<typename T>
constexpr degree<T> to_degrees(const radian<T>& r) {
    return degree<T>(r);
}

/**
 * @brief Identity conversion for radians
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param r Angle already in radians
 * @return The same angle
 * 
 * @details
 * This overload allows generic code to call to_radians() without
 * checking if the angle is already in radians.
 */
template<typename T>
constexpr radian<T> to_radians(const radian<T>& r) {
    return r;
}

/**
 * @brief Identity conversion for degrees
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param d Angle already in degrees
 * @return The same angle
 * 
 * @details
 * This overload allows generic code to call to_degrees() without
 * checking if the angle is already in degrees.
 */
template<typename T>
constexpr degree<T> to_degrees(const degree<T>& d) {
    return d;
}

/**
 * @brief Absolute value of an angle
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a The angle
 * @return Absolute value of the angle
 * 
 * @details
 * Returns the absolute value of an angle, preserving its unit type.
 * Useful for magnitude calculations or ensuring positive angles.
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> abs(const angle<T, Unit>& a) {
    return angle<T, Unit>(std::abs(a.value()));
}

/**
 * @brief Minimum of two angles
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a First angle
 * @param b Second angle
 * @return The smaller angle
 * 
 * @warning This compares raw angle values without wrapping consideration
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> min(const angle<T, Unit>& a, const angle<T, Unit>& b) {
    return a.value() < b.value() ? a : b;
}

/**
 * @brief Maximum of two angles
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a First angle
 * @param b Second angle
 * @return The larger angle
 * 
 * @warning This compares raw angle values without wrapping consideration
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> max(const angle<T, Unit>& a, const angle<T, Unit>& b) {
    return a.value() > b.value() ? a : b;
}

/**
 * @brief Clamp angle to range
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param val The angle to clamp
 * @param low Lower bound
 * @param high Upper bound
 * @return Clamped angle in range [low, high]
 * 
 * @details
 * Clamps an angle to the specified range. This operates on raw angle
 * values without wrapping consideration.
 * 
 * @warning For circular clamping with wrap-around, use custom logic
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> clamp(const angle<T, Unit>& val, 
                               const angle<T, Unit>& low, 
                               const angle<T, Unit>& high) {
    return min(max(val, low), high);
}

/**
 * @brief Create degree angle from numeric value
 * @ingroup AnglesModule
 * @tparam T The numeric type (default: float)
 * @param value The angle value in degrees
 * @return A degree angle
 * 
 * @details
 * Factory function for creating degree angles. Useful when the type
 * needs to be deduced from the argument.
 * 
 * @example
 * @code
 * auto angle = degrees(45.0);  // Creates degree<double>
 * @endcode
 */
template<typename T = float>
constexpr degree<T> degrees(T value) {
    return degree<T>(value);
}

/**
 * @brief Create radian angle from numeric value
 * @ingroup AnglesModule
 * @tparam T The numeric type (default: float)
 * @param value The angle value in radians
 * @return A radian angle
 * 
 * @details
 * Factory function for creating radian angles. Useful when the type
 * needs to be deduced from the argument.
 * 
 * @example
 * @code
 * auto angle = radians(3.14159);  // Creates radian<double>
 * @endcode
 */
template<typename T = float>
constexpr radian<T> radians(T value) {
    return radian<T>(value);
}

/**
 * @brief Calculate shortest angle difference (degrees)
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param a Start angle
 * @param b End angle
 * @return Shortest angle from a to b in range [-180°, 180°]
 * 
 * @details
 * Calculates the shortest angular distance from angle a to angle b,
 * taking wrap-around into account. The result is always in the range
 * [-180°, 180°], where positive values indicate clockwise rotation.
 * 
 * @example
 * @code
 * auto diff1 = angle_difference(degrees(10.0f), degrees(350.0f));
 * // Returns -20° (shortest path is counterclockwise)
 * 
 * auto diff2 = angle_difference(degrees(350.0f), degrees(10.0f));
 * // Returns 20° (shortest path is clockwise)
 * @endcode
 */
template<typename T>
degree<T> angle_difference(const degree<T>& a, const degree<T>& b) {
    degree<T> diff = b - a;
    return wrap(diff);
}

/**
 * @brief Calculate shortest angle difference (radians)
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param a Start angle
 * @param b End angle
 * @return Shortest angle from a to b in range [-π, π]
 * 
 * @details
 * Calculates the shortest angular distance from angle a to angle b,
 * taking wrap-around into account. The result is always in the range
 * [-π, π], where positive values indicate clockwise rotation.
 */
template<typename T>
radian<T> angle_difference(const radian<T>& a, const radian<T>& b) {
    radian<T> diff = b - a;
    return wrap(diff);
}

/**
 * @note
 * For comparing angles with tolerance, use the general approx_equal()
 * function from core/approx_equal.hh. It handles angle types automatically
 * and correctly deals with wrap-around when comparing angles near the
 * discontinuity (e.g., 359° ≈ 1°).
 */

/**
 * @struct angle_components
 * @brief Cartesian components of a unit vector at given angle
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * 
 * @details
 * Represents the x and y components of a unit vector pointing in the
 * direction specified by an angle. This is equivalent to the cosine
 * and sine of the angle.
 */
template<typename T>
struct angle_components {
    T cos;  ///< Cosine of the angle (x-component)
    T sin;  ///< Sine of the angle (y-component)
};

/**
 * @brief Convert angle to Cartesian components (radians)
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param angle The angle in radians
 * @return Structure containing cos and sin of the angle
 * 
 * @details
 * Converts an angle to its unit vector representation in Cartesian
 * coordinates. This is useful for converting from polar to Cartesian
 * coordinates or for rotation calculations.
 * 
 * @example
 * @code
 * auto comp = angle_to_components(radians(π/4));
 * // comp.cos ≈ 0.707, comp.sin ≈ 0.707
 * @endcode
 */
template<typename T>
angle_components<T> angle_to_components(const radian<T>& angle) {
    return { std::cos(angle.value()), std::sin(angle.value()) };
}

/**
 * @brief Convert angle to Cartesian components (degrees)
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param angle The angle in degrees
 * @return Structure containing cos and sin of the angle
 * 
 * @details
 * Converts an angle to its unit vector representation. Internally
 * converts to radians before computing trigonometric functions.
 */
template<typename T>
angle_components<T> angle_to_components(const degree<T>& angle) {
    return angle_to_components(to_radians(angle));
}

/**
 * @brief Create angle from Cartesian components
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @param y Y-component (sine)
 * @param x X-component (cosine)
 * @return Angle in radians using atan2
 * 
 * @details
 * Computes the angle of a vector from the origin to point (x, y).
 * Uses atan2 which correctly handles all quadrants and returns
 * angles in the range [-π, π].
 * 
 * @note The order is (y, x) to match std::atan2 convention
 * 
 * @example
 * @code
 * auto angle = angle_from_components(1.0f, 1.0f);  // π/4 rad (45°)
 * auto angle2 = angle_from_components(0.0f, -1.0f); // π rad (180°)
 * @endcode
 */
template<typename T>
radian<T> angle_from_components(T y, T x) {
    return radian<T>(std::atan2(y, x));
}

/**
 * @brief Create angle from Cartesian components with specified type
 * @ingroup AnglesModule
 * @tparam AngleType The desired angle type (degree<T> or radian<T>)
 * @tparam T The numeric type
 * @param y Y-component (sine)
 * @param x X-component (cosine)
 * @return Angle in the specified unit type
 * 
 * @details
 * This overload allows specifying whether you want the result in
 * degrees or radians. Useful in generic code where the angle type
 * is a template parameter.
 * 
 * @example
 * @code
 * auto deg = angle_from_components<degree<float>>(1.0f, 0.0f);  // 90°
 * auto rad = angle_from_components<radian<float>>(1.0f, 0.0f);  // π/2
 * @endcode
 */
template<typename AngleType, typename T,
         typename = std::enable_if_t<is_angle_v<AngleType>>>
AngleType angle_from_components(T y, T x) {
    if constexpr (std::is_same_v<AngleType, radian<T>>) {
        return radian<T>(std::atan2(y, x));
    } else if constexpr (std::is_same_v<AngleType, degree<T>>) {
        return to_degrees(radian<T>(std::atan2(y, x)));
    }
}


/**
 * @brief Modulo operation for angles
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a The angle to modulate
 * @param modulus The modulus angle
 * @return Remainder after division: a % modulus
 * 
 * @details
 * Performs floating-point modulo operation on angle values.
 * Unlike wrap functions, this uses an arbitrary modulus.
 * 
 * @note Result has the same sign as the dividend (a)
 */
template<typename T, typename Unit>
angle<T, Unit> mod_angle(const angle<T, Unit>& a, const angle<T, Unit>& modulus) {
    return angle<T, Unit>(std::fmod(a.value(), modulus.value()));
}

/**
 * @brief Get sign of angle
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a The angle
 * @return -1 for negative, 0 for zero, 1 for positive
 * 
 * @details
 * Returns the sign of an angle as an integer. This is useful for
 * determining rotation direction or angle quadrant.
 */
template<typename T, typename Unit>
constexpr int sign(const angle<T, Unit>& a) {
    return (a.value() > T(0)) - (a.value() < T(0));
}

} // namespace euler