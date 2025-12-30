/**
 * @file projective3.hh
 * @brief 3D projective (homogeneous) coordinates for the Euler library
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/coordinates/point3.hh>
#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/vector/vector.hh>
#include <cmath>
#include <limits>

namespace euler {

/**
 * @brief 3D projective (homogeneous) coordinates
 * @tparam T The scalar type (float or double)
 * 
 * Represents a point in 3D projective space using homogeneous coordinates (x, y, z, w).
 * When w≠0, represents the Cartesian point (x/w, y/w, z/w).
 * When w=0, represents a point at infinity (direction).
 */
template<typename T>
struct projective3 {
    static_assert(std::is_floating_point_v<T>, 
                  "projective3 requires floating point type");
    
    using value_type = T;
    
    /// X coordinate (homogeneous)
    T x;
    /// Y coordinate (homogeneous)
    T y;
    /// Z coordinate (homogeneous)
    T z;
    /// W coordinate (homogeneous weight)
    T w;
    
    /**
     * @brief Default constructor - initializes to origin (0, 0, 0, 1)
     */
    constexpr projective3() : x{}, y{}, z{}, w{T(1)} {}
    
    /**
     * @brief Construct from homogeneous coordinates
     * @param x_ X coordinate
     * @param y_ Y coordinate
     * @param z_ Z coordinate
     * @param w_ W coordinate (default 1)
     */
    constexpr projective3(T x_, T y_, T z_, T w_ = T(1)) 
        : x(x_), y(y_), z(z_), w(w_) {}
    
    /**
     * @brief Implicit construction from Cartesian point
     * @param p Cartesian point
     * 
     * Creates projective point (p.x, p.y, p.z, 1)
     */
    constexpr projective3(const point3<T>& p) 
        : x(p.x), y(p.y), z(p.z), w(T(1)) {}
    
    /**
     * @brief Convert from projective with different scalar type
     * @tparam U Source scalar type
     * @param p Source projective point
     */
    template<typename U>
    explicit constexpr projective3(const projective3<U>& p) 
        : x(static_cast<T>(p.x)), y(static_cast<T>(p.y)), 
          z(static_cast<T>(p.z)), w(static_cast<T>(p.w)) {}
    
    /**
     * @brief Explicit conversion to Cartesian point
     * @return Cartesian point (x/w, y/w, z/w)
     * 
     * If w=0 (point at infinity), returns point with infinity coordinates
     */
    explicit constexpr operator point3<T>() const {
        if (w == T(0)) {
            // Point at infinity
            return {std::numeric_limits<T>::infinity(), 
                    std::numeric_limits<T>::infinity(),
                    std::numeric_limits<T>::infinity()};
        }
        return {x/w, y/w, z/w};
    }
    
    /**
     * @brief Get Cartesian point representation
     * @return Cartesian point (x/w, y/w, z/w)
     * 
     * Convenience method equivalent to explicit cast
     */
    constexpr point3<T> point() const { 
        return static_cast<point3<T>>(*this); 
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
     * If w=0, returns unchanged. Otherwise returns (x/w, y/w, z/w, 1)
     */
    constexpr projective3 normalized() const {
        if (w == T(0)) return *this;
        return {x/w, y/w, z/w, T(1)};
    }
    
    /**
     * @brief Convert to homogeneous vector
     * @return 4D vector (x, y, z, w)
     */
    constexpr vector<T, 4> vec() const { return {x, y, z, w}; }
    
    /**
     * @brief Access coordinates by index
     * @param i Index (0=x, 1=y, 2=z, 3=w)
     * @return Reference to coordinate
     */
    constexpr T& operator[](size_t i) {
        EULER_CHECK_INDEX(i, 4);
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        return w;
    }

    /**
     * @brief Access coordinates by index (const)
     * @param i Index (0=x, 1=y, 2=z, 3=w)
     * @return Const reference to coordinate
     */
    constexpr const T& operator[](size_t i) const {
        EULER_CHECK_INDEX(i, 4);
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        return w;
    }
    
    /**
     * @brief Construct from homogeneous vector
     * @param v 4D vector with homogeneous coordinates
     */
    explicit constexpr projective3(const vector<T, 4>& v) 
        : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}
    
    /**
     * @brief Create projective point from Cartesian with specific w
     * @param p Cartesian point
     * @param w_ W coordinate
     * @return Projective point (p.x*w, p.y*w, p.z*w, w)
     */
    static constexpr projective3 from_cartesian(const point3<T>& p, T w_ = T(1)) {
        return {p.x * w_, p.y * w_, p.z * w_, w_};
    }
    
    /**
     * @brief Create point at infinity from direction
     * @param dx X direction
     * @param dy Y direction
     * @param dz Z direction
     * @return Projective point (dx, dy, dz, 0)
     */
    static constexpr projective3 at_infinity(T dx, T dy, T dz) {
        return {dx, dy, dz, T(0)};
    }
    
    /**
     * @brief Create point at infinity from direction vector
     * @param dir Direction vector
     * @return Projective point (dir.x, dir.y, dir.z, 0)
     */
    static constexpr projective3 at_infinity(const vector<T, 3>& dir) {
        return {dir[0], dir[1], dir[2], T(0)};
    }
};

// Type aliases
using proj3f = projective3<float>;
using proj3d = projective3<double>;

// Note: point3 -> projective3 conversion is handled by projective3's constructor

} // namespace euler