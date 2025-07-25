/**
 * @file point3.hh
 * @brief 3D point type for the Euler library
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/coordinates/point2.hh>
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
template<typename T> struct projective3;

/**
 * @brief 3D point in Cartesian coordinates
 * @tparam T The scalar type (float, double, or int)
 * 
 * Represents a position in 3D space. Provides conversions to/from vectors
 * and projective coordinates, plus swizzling operations.
 */
template<typename T>
struct point3 {
    static_assert(std::is_arithmetic_v<T>, "point3 requires arithmetic type");
    
    using value_type = T;
    
    /// X coordinate
    T x;
    /// Y coordinate
    T y;
    /// Z coordinate
    T z;
    
    /**
     * @brief Default constructor - initializes to origin (0, 0, 0)
     */
    constexpr point3() : x{}, y{}, z{} {}
    
    /**
     * @brief Construct from coordinates
     * @param x_ X coordinate
     * @param y_ Y coordinate
     * @param z_ Z coordinate
     */
    constexpr point3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    
    /**
     * @brief Construct from 2D point and Z coordinate
     * @param p 2D point for X and Y
     * @param z_ Z coordinate (default 0)
     */
    constexpr point3(const point2<T>& p, T z_ = 0) : x(p.x), y(p.y), z(z_) {}
    
    /**
     * @brief Convert from point with different scalar type
     * @tparam U Source scalar type
     * @param p Source point
     */
    template<typename U>
    explicit constexpr point3(const point3<U>& p)
        : x(static_cast<T>(p.x)), y(static_cast<T>(p.y)), z(static_cast<T>(p.z)) {}
    
    // Note: We rely on projective3's implicit constructor from point3
    // instead of providing a conversion operator to avoid ambiguity
    
    /**
     * @brief Create zero point at origin
     * @return Point at (0, 0, 0)
     */
    static constexpr point3 zero() { return {0, 0, 0}; }
    
    /**
     * @brief Create point from spherical coordinates
     * @param r Radius (distance from origin)
     * @param theta Azimuthal angle in XY plane from positive X axis
     * @param phi Polar angle from positive Z axis
     * @return Point at given spherical coordinates
     */
    template<typename Angle1, typename Angle2>
    static constexpr point3 spherical(T r, const Angle1& theta, const Angle2& phi) {
        static_assert(is_angle_v<Angle1> && is_angle_v<Angle2>, 
                      "spherical() requires angle types");
        auto theta_rad = to_radians(theta);
        auto phi_rad = to_radians(phi);
        T sin_phi = sin(phi_rad);
        return {
            r * sin_phi * cos(theta_rad),
            r * sin_phi * sin(theta_rad),
            r * cos(phi_rad)
        };
    }
    
    /**
     * @brief Create point from cylindrical coordinates
     * @param r Radius in XY plane
     * @param theta Angle in XY plane from positive X axis
     * @param z Z coordinate
     * @return Point at given cylindrical coordinates
     */
    template<typename Angle>
    static constexpr point3 cylindrical(T r, const Angle& theta, T z) {
        static_assert(is_angle_v<Angle>, "cylindrical() requires angle type");
        auto theta_rad = to_radians(theta);
        return {r * cos(theta_rad), r * sin(theta_rad), z};
    }
    
    /**
     * @brief Access coordinates by index
     * @param i Index (0=x, 1=y, 2=z)
     * @return Reference to coordinate
     */
    constexpr T& operator[](size_t i) { 
        EULER_CHECK_INDEX(i, 3);
        return (&x)[i]; 
    }
    
    /**
     * @brief Access coordinates by index (const)
     * @param i Index (0=x, 1=y, 2=z)
     * @return Const reference to coordinate
     */
    constexpr const T& operator[](size_t i) const { 
        EULER_CHECK_INDEX(i, 3);
        return (&x)[i]; 
    }
    
    // 2D projection swizzles
    
    /**
     * @brief Get XY projection
     * @return 2D point with X and Y coordinates
     */
    constexpr point2<T> xy() const { return {x, y}; }
    
    /**
     * @brief Get XZ projection
     * @return 2D point with X and Z coordinates
     */
    constexpr point2<T> xz() const { return {x, z}; }
    
    /**
     * @brief Get YX projection
     * @return 2D point with Y and X coordinates
     */
    constexpr point2<T> yx() const { return {y, x}; }
    
    /**
     * @brief Get YZ projection
     * @return 2D point with Y and Z coordinates
     */
    constexpr point2<T> yz() const { return {y, z}; }
    
    /**
     * @brief Get ZX projection
     * @return 2D point with Z and X coordinates
     */
    constexpr point2<T> zx() const { return {z, x}; }
    
    /**
     * @brief Get ZY projection
     * @return 2D point with Z and Y coordinates
     */
    constexpr point2<T> zy() const { return {z, y}; }
    
    // 3D swizzles (selected useful ones)
    
    /**
     * @brief Get XYZ swizzle (identity)
     * @return Copy of this point
     */
    constexpr point3 xyz() const { return *this; }
    
    /**
     * @brief Get ZYX swizzle (reversed)
     * @return Point with coordinates reversed
     */
    constexpr point3 zyx() const { return {z, y, x}; }
    
    /**
     * @brief Get XZY swizzle
     * @return Point with Y and Z swapped
     */
    constexpr point3 xzy() const { return {x, z, y}; }
    
    /**
     * @brief Get YXZ swizzle
     * @return Point with X and Y swapped
     */
    constexpr point3 yxz() const { return {y, x, z}; }
    
    /**
     * @brief Get YZX swizzle
     * @return Point with coordinates rotated
     */
    constexpr point3 yzx() const { return {y, z, x}; }
    
    /**
     * @brief Get ZXY swizzle
     * @return Point with coordinates rotated
     */
    constexpr point3 zxy() const { return {z, x, y}; }
    
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
     * @brief Access z coordinate as b (blue)
     */
    constexpr T& b() { return z; }
    constexpr const T& b() const { return z; }
    
    /**
     * @brief Convert to 3D vector
     * @return Vector with same components
     */
    constexpr vector<T, 3> vec() const { return {x, y, z}; }
    
    /**
     * @brief Construct from 3D vector
     * @param v Source vector
     */
    explicit constexpr point3(const vector<T, 3>& v) : x(v[0]), y(v[1]), z(v[2]) {}
};

// Type aliases for common use cases
using point3i = point3<int>;
using point3f = point3<float>;
using point3d = point3<double>;

} // namespace euler