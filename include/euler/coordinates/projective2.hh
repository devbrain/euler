/**
 * @file projective2.hh
 * @brief 2D projective (homogeneous) coordinates for the Euler library
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/vector/vector.hh>
#include <cmath>
#include <limits>

namespace euler {

/**
 * @brief 2D projective (homogeneous) coordinates
 * @tparam T The scalar type (float or double)
 * 
 * Represents a point in 2D projective space using homogeneous coordinates (x, y, w).
 * When w≠0, represents the Cartesian point (x/w, y/w).
 * When w=0, represents a point at infinity (direction).
 */
template<typename T>
struct projective2 {
    static_assert(std::is_floating_point_v<T>, 
                  "projective2 requires floating point type");
    
    using value_type = T;
    
    /// X coordinate (homogeneous)
    T x;
    /// Y coordinate (homogeneous)
    T y;
    /// W coordinate (homogeneous weight)
    T w;
    
    /**
     * @brief Default constructor - initializes to origin (0, 0, 1)
     */
    constexpr projective2() : x{}, y{}, w{T(1)} {}
    
    /**
     * @brief Construct from homogeneous coordinates
     * @param x_ X coordinate
     * @param y_ Y coordinate
     * @param w_ W coordinate (default 1)
     */
    constexpr projective2(T x_, T y_, T w_ = T(1)) : x(x_), y(y_), w(w_) {}
    
    /**
     * @brief Implicit construction from Cartesian point
     * @param p Cartesian point
     * 
     * Creates projective point (p.x, p.y, 1)
     */
    constexpr projective2(const point2<T>& p) : x(p.x), y(p.y), w(T(1)) {}
    
    /**
     * @brief Convert from projective with different scalar type
     * @tparam U Source scalar type
     * @param p Source projective point
     */
    template<typename U>
    explicit constexpr projective2(const projective2<U>& p) 
        : x(static_cast<T>(p.x)), y(static_cast<T>(p.y)), w(static_cast<T>(p.w)) {}
    
    /**
     * @brief Explicit conversion to Cartesian point
     * @return Cartesian point (x/w, y/w)
     * 
     * If w=0 (point at infinity), returns point with infinity coordinates
     */
    explicit constexpr operator point2<T>() const {
        if (w == T(0)) {
            // Point at infinity
            return {std::numeric_limits<T>::infinity(), 
                    std::numeric_limits<T>::infinity()};
        }
        return {x/w, y/w};
    }
    
    /**
     * @brief Get Cartesian point representation
     * @return Cartesian point (x/w, y/w)
     * 
     * Convenience method equivalent to explicit cast
     */
    constexpr point2<T> point() const { 
        return static_cast<point2<T>>(*this); 
    }
    
    /**
     * @brief Check if this is a point at infinity
     * @return True if w=0
     */
    constexpr bool is_infinite() const { return w == T(0); }
    
    /**
     * @brief Get normalized projective coordinates
     * @return Projective point with w=1 (if possible)
     * 
     * If w=0, returns unchanged. Otherwise returns (x/w, y/w, 1)
     */
    constexpr projective2 normalized() const {
        if (w == T(0)) return *this;
        return {x/w, y/w, T(1)};
    }
    
    /**
     * @brief Convert to homogeneous vector
     * @return 3D vector (x, y, w)
     */
    constexpr vector<T, 3> vec() const { return {x, y, w}; }
    
    /**
     * @brief Access coordinates by index
     * @param i Index (0=x, 1=y, 2=w)
     * @return Reference to coordinate
     */
    constexpr T& operator[](size_t i) {
        EULER_CHECK_INDEX(i, 3);
        return (&x)[i];
    }
    
    /**
     * @brief Access coordinates by index (const)
     * @param i Index (0=x, 1=y, 2=w)
     * @return Const reference to coordinate
     */
    constexpr const T& operator[](size_t i) const {
        EULER_CHECK_INDEX(i, 3);
        return (&x)[i];
    }
    
    /**
     * @brief Construct from homogeneous vector
     * @param v 3D vector with homogeneous coordinates
     */
    explicit constexpr projective2(const vector<T, 3>& v) 
        : x(v[0]), y(v[1]), w(v[2]) {}
    
    /**
     * @brief Create projective point from Cartesian with specific w
     * @param p Cartesian point
     * @param w_ W coordinate
     * @return Projective point (p.x*w, p.y*w, w)
     */
    static constexpr projective2 from_cartesian(const point2<T>& p, T w_ = T(1)) {
        return {p.x * w_, p.y * w_, w_};
    }
    
    /**
     * @brief Create point at infinity from direction
     * @param dx X direction
     * @param dy Y direction
     * @return Projective point (dx, dy, 0)
     */
    static constexpr projective2 at_infinity(T dx, T dy) {
        return {dx, dy, T(0)};
    }
    
    /**
     * @brief Create point at infinity from direction vector
     * @param dir Direction vector
     * @return Projective point (dir.x, dir.y, 0)
     */
    static constexpr projective2 at_infinity(const vector<T, 2>& dir) {
        return {dir[0], dir[1], T(0)};
    }
};

// Type aliases
using proj2f = projective2<float>;
using proj2d = projective2<double>;

// Note: point2 -> projective2 conversion is handled by projective2's constructor

} // namespace euler