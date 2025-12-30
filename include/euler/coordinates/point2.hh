/**
 * @file point2.hh
 * @brief 2D point type for the Euler library
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <euler/vector/vector.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/angle_traits.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/math/trigonometry.hh>
#include <cmath>
#include <limits>
#include <type_traits>

namespace euler {

// Forward declarations
template<typename T> struct projective2;

/**
 * @brief 2D point in Cartesian coordinates
 * @tparam T The scalar type (float, double, or int)
 * 
 * Represents a position in 2D space. Satisfies the point_like concept
 * and provides conversions to/from vectors and projective coordinates.
 */
template<typename T>
struct point2 {
    static_assert(std::is_arithmetic_v<T>, "point2 requires arithmetic type");
    
    using value_type = T;
    
    /// X coordinate
    T x;
    /// Y coordinate  
    T y;
    
    /**
     * @brief Default constructor - initializes to origin (0, 0)
     */
    constexpr point2() : x{}, y{} {}
    
    /**
     * @brief Construct from coordinates
     * @param x_ X coordinate
     * @param y_ Y coordinate
     */
    constexpr point2(T x_, T y_) : x(x_), y(y_) {}
    
    /**
     * @brief Convert from point with different scalar type
     * @tparam U Source scalar type
     * @param p Source point
     */
    template<typename U>
    explicit constexpr point2(const point2<U>& p) 
        : x(static_cast<T>(p.x)), y(static_cast<T>(p.y)) {}
    
    // Note: We rely on projective2's implicit constructor from point2
    // instead of providing a conversion operator to avoid ambiguity
    
    /**
     * @brief Create zero point at origin
     * @return Point at (0, 0)
     */
    static constexpr point2 zero() { return {0, 0}; }
    
    /**
     * @brief Create point from polar coordinates
     * @param r Radius (distance from origin)
     * @param theta Angle from positive X axis (accepts degree or radian)
     * @return Point at given polar coordinates
     */
    template<typename Angle>
    static constexpr point2 polar(T r, const Angle& theta) {
        static_assert(is_angle_v<Angle>, "polar() requires angle type");
        auto theta_rad = to_radians(theta);
        return {r * cos(theta_rad), r * sin(theta_rad)};
    }
    
    /**
     * @brief Access coordinates by index
     * @param i Index (0=x, 1=y)
     * @return Reference to coordinate
     */
    constexpr T& operator[](size_t i) {
        EULER_CHECK_INDEX(i, 2);
        if (i == 0) return x;
        return y;
    }

    /**
     * @brief Access coordinates by index (const)
     * @param i Index (0=x, 1=y)
     * @return Const reference to coordinate
     */
    constexpr const T& operator[](size_t i) const {
        EULER_CHECK_INDEX(i, 2);
        if (i == 0) return x;
        return y;
    }
    
    // Swizzling operations
    
    /**
     * @brief Get xy swizzle (identity)
     * @return Copy of this point
     */
    constexpr point2 xy() const { return *this; }
    
    /**
     * @brief Get yx swizzle (coordinates swapped)
     * @return Point with x and y swapped
     */
    constexpr point2 yx() const { return {y, x}; }
    
    /**
     * @brief Get xx swizzle
     * @return Point with both coordinates set to x
     */
    constexpr point2 xx() const { return {x, x}; }
    
    /**
     * @brief Get yy swizzle
     * @return Point with both coordinates set to y
     */
    constexpr point2 yy() const { return {y, y}; }
    
    // Alternative coordinate access (useful for colors)
    
    /**
     * @brief Access x coordinate as r (red)
     */
    constexpr T& r() { return x; }
    constexpr const T& r() const { return x; }
    
    /**
     * @brief Access y coordinate as g (green)
     */
    constexpr T& g() { return y; }
    constexpr const T& g() const { return y; }
    
    /**
     * @brief Convert to 2D vector
     * @return Vector with same components
     */
    constexpr vector<T, 2> vec() const { return {x, y}; }
    
    /**
     * @brief Construct from 2D vector
     * @param v Source vector
     */
    explicit constexpr point2(const vector<T, 2>& v) : x(v[0]), y(v[1]) {}
};

// Type aliases for common use cases
using point2i = point2<int>;
using point2f = point2<float>;
using point2d = point2<double>;

} // namespace euler