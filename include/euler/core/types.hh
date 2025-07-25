/**
 * @file types.hh
 * @brief Core type definitions and mathematical constants for the Euler library
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <limits>

/**
 * @namespace euler
 * @brief Main namespace for the Euler mathematical library
 */
namespace euler {

/**
 * @typedef float32
 * @brief 32-bit floating point type
 */
using float32 = float;

/**
 * @typedef float64
 * @brief 64-bit floating point type
 */
using float64 = double;

/**
 * @typedef size_t
 * @brief Type for sizes and counts
 */
using size_t = std::size_t;

/**
 * @typedef index_t
 * @brief Type for array/matrix indices
 */
using index_t = std::size_t;

/**
 * @typedef scalar
 * @brief Default scalar type (float or double based on EULER_DEFAULT_PRECISION_DOUBLE)
 * 
 * Defaults to float32 unless EULER_DEFAULT_PRECISION_DOUBLE is defined,
 * in which case it defaults to float64.
 */
#ifdef EULER_DEFAULT_PRECISION_DOUBLE
using scalar = float64;
#else
using scalar = float32;
#endif

/**
 * @struct constants
 * @brief Mathematical constants for a given numeric type
 * @tparam T The numeric type (typically float or double)
 */
template<typename T>
struct constants {
    /** @brief π (pi) */
    static constexpr T pi = T(3.14159265358979323846);
    /** @brief 2π */
    static constexpr T two_pi = T(6.28318530717958647692);
    /** @brief π/2 */
    static constexpr T half_pi = T(1.57079632679489661923);
    /** @brief π/4 */
    static constexpr T quarter_pi = T(0.78539816339744830961);
    /** @brief 1/π */
    static constexpr T inv_pi = T(0.31830988618379067154);
    /** @brief 1/(2π) */
    static constexpr T inv_two_pi = T(0.15915494309189533577);
    
    /** @brief Euler's number (e) */
    static constexpr T e = T(2.71828182845904523536);
    /** @brief √2 */
    static constexpr T sqrt2 = T(1.41421356237309504880);
    /** @brief 1/√2 */
    static constexpr T inv_sqrt2 = T(0.70710678118654752440);
    /** @brief √3 */
    static constexpr T sqrt3 = T(1.73205080756887729352);
    /** @brief 1/√3 */
    static constexpr T inv_sqrt3 = T(0.57735026918962576450);
    
    /** @brief Conversion factor from degrees to radians */
    static constexpr T deg_to_rad = pi / T(180);
    /** @brief Conversion factor from radians to degrees */
    static constexpr T rad_to_deg = T(180) / pi;
    
    /** @brief Machine epsilon for type T */
    #ifdef EULER_DEFAULT_EPSILON
    static constexpr T epsilon = T(EULER_DEFAULT_EPSILON);
    #else
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();
    #endif
    /** @brief Positive infinity for type T */
    static constexpr T infinity = std::numeric_limits<T>::infinity();
};

/**
 * @defgroup ConvenienceConstants Convenience Constants
 * @brief Global constants using the default scalar type
 * @{
 */
constexpr auto pi = constants<scalar>::pi;           ///< π using default scalar type
constexpr auto two_pi = constants<scalar>::two_pi;   ///< 2π using default scalar type
constexpr auto half_pi = constants<scalar>::half_pi; ///< π/2 using default scalar type
constexpr auto quarter_pi = constants<scalar>::quarter_pi; ///< π/4 using default scalar type
constexpr auto e = constants<scalar>::e;             ///< Euler's number using default scalar type
constexpr auto sqrt2 = constants<scalar>::sqrt2;     ///< √2 using default scalar type
constexpr auto epsilon = constants<scalar>::epsilon; ///< Machine epsilon using default scalar type
/** @} */

// Helper to determine if a type is a floating point type
template<typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

// Helper to determine if a type is an arithmetic type
template<typename T>
constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

} // namespace euler