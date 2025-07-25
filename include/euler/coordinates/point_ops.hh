/**
 * @file point_ops.hh
 * @brief Operators and utility functions for point types
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/projective3.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/core/approx_equal.hh>
#include <cmath>
#include <type_traits>

namespace euler {

// Point-vector operations (2D)

/**
 * @brief Translate point by vector
 * @param p Point to translate
 * @param v Translation vector
 * @return Translated point
 */
template<typename T>
constexpr point2<T> operator+(const point2<T>& p, const vector<T, 2>& v) {
    return {p.x + v[0], p.y + v[1]};
}

/**
 * @brief Translate point by vector (vector first)
 * @param v Translation vector
 * @param p Point to translate
 * @return Translated point
 */
template<typename T>
constexpr point2<T> operator+(const vector<T, 2>& v, const point2<T>& p) {
    return p + v;
}

/**
 * @brief Translate point by negative vector
 * @param p Point to translate
 * @param v Translation vector
 * @return Translated point
 */
template<typename T>
constexpr point2<T> operator-(const point2<T>& p, const vector<T, 2>& v) {
    return {p.x - v[0], p.y - v[1]};
}

/**
 * @brief Get displacement vector between points
 * @param a End point
 * @param b Start point
 * @return Displacement vector from b to a
 */
template<typename T>
constexpr vector<T, 2> operator-(const point2<T>& a, const point2<T>& b) {
    return {a.x - b.x, a.y - b.y};
}

// Point-point addition is explicitly deleted (undefined operation)
template<typename T>
point2<T> operator+(const point2<T>&, const point2<T>&) = delete;

// Point-vector operations (3D)

/**
 * @brief Translate point by vector
 * @param p Point to translate
 * @param v Translation vector
 * @return Translated point
 */
template<typename T>
constexpr point3<T> operator+(const point3<T>& p, const vector<T, 3>& v) {
    return {p.x + v[0], p.y + v[1], p.z + v[2]};
}

/**
 * @brief Translate point by vector (vector first)
 * @param v Translation vector
 * @param p Point to translate
 * @return Translated point
 */
template<typename T>
constexpr point3<T> operator+(const vector<T, 3>& v, const point3<T>& p) {
    return p + v;
}

/**
 * @brief Translate point by negative vector
 * @param p Point to translate
 * @param v Translation vector
 * @return Translated point
 */
template<typename T>
constexpr point3<T> operator-(const point3<T>& p, const vector<T, 3>& v) {
    return {p.x - v[0], p.y - v[1], p.z - v[2]};
}

/**
 * @brief Get displacement vector between points
 * @param a End point
 * @param b Start point
 * @return Displacement vector from b to a
 */
template<typename T>
constexpr vector<T, 3> operator-(const point3<T>& a, const point3<T>& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// Point-point addition is explicitly deleted (undefined operation)
template<typename T>
point3<T> operator+(const point3<T>&, const point3<T>&) = delete;

// Scalar multiplication (scaling from origin)

/**
 * @brief Scale point from origin
 * @param s Scale factor
 * @param p Point to scale
 * @return Scaled point
 */
template<typename T>
constexpr point2<T> operator*(T s, const point2<T>& p) {
    return {s * p.x, s * p.y};
}

/**
 * @brief Scale point from origin
 * @param p Point to scale
 * @param s Scale factor
 * @return Scaled point
 */
template<typename T>
constexpr point2<T> operator*(const point2<T>& p, T s) {
    return s * p;
}

/**
 * @brief Scale point from origin
 * @param s Scale factor
 * @param p Point to scale
 * @return Scaled point
 */
template<typename T>
constexpr point3<T> operator*(T s, const point3<T>& p) {
    return {s * p.x, s * p.y, s * p.z};
}

/**
 * @brief Scale point from origin
 * @param p Point to scale
 * @param s Scale factor
 * @return Scaled point
 */
template<typename T>
constexpr point3<T> operator*(const point3<T>& p, T s) {
    return s * p;
}

/**
 * @brief Divide point coordinates by scalar
 * @param p Point to divide
 * @param s Divisor
 * @return Point with divided coordinates
 */
template<typename T>
constexpr point2<T> operator/(const point2<T>& p, T s) {
    return {p.x / s, p.y / s};
}

/**
 * @brief Divide point coordinates by scalar
 * @param p Point to divide
 * @param s Divisor
 * @return Point with divided coordinates
 */
template<typename T>
constexpr point3<T> operator/(const point3<T>& p, T s) {
    return {p.x / s, p.y / s, p.z / s};
}

// Comparison operators

/**
 * @brief Check exact equality of points
 * @param a First point
 * @param b Second point
 * @return True if all coordinates are exactly equal
 */
template<typename T>
constexpr bool operator==(const point2<T>& a, const point2<T>& b) {
    return a.x == b.x && a.y == b.y;
}

/**
 * @brief Check inequality of points
 * @param a First point
 * @param b Second point
 * @return True if any coordinate differs
 */
template<typename T>
constexpr bool operator!=(const point2<T>& a, const point2<T>& b) {
    return !(a == b);
}

/**
 * @brief Check exact equality of points
 * @param a First point
 * @param b Second point
 * @return True if all coordinates are exactly equal
 */
template<typename T>
constexpr bool operator==(const point3<T>& a, const point3<T>& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

/**
 * @brief Check inequality of points
 * @param a First point
 * @param b Second point
 * @return True if any coordinate differs
 */
template<typename T>
constexpr bool operator!=(const point3<T>& a, const point3<T>& b) {
    return !(a == b);
}

// Approximate equality

/**
 * @brief Check approximate equality of 2D points
 * @param a First point
 * @param b Second point
 * @param epsilon Tolerance for each coordinate
 * @return True if all coordinates are within epsilon
 */
template<typename T>
constexpr bool approx_equal(const point2<T>& a, const point2<T>& b, 
                           T eps = constants<T>::epsilon) {
    return approx_equal(a.x, b.x, eps) && 
           approx_equal(a.y, b.y, eps);
}

/**
 * @brief Check approximate equality of 3D points
 * @param a First point
 * @param b Second point
 * @param epsilon Tolerance for each coordinate
 * @return True if all coordinates are within epsilon
 */
template<typename T>
constexpr bool approx_equal(const point3<T>& a, const point3<T>& b, 
                           T eps = constants<T>::epsilon) {
    return approx_equal(a.x, b.x, eps) && 
           approx_equal(a.y, b.y, eps) &&
           approx_equal(a.z, b.z, eps);
}

// Matrix transformations

/**
 * @brief Transform 2D point by 3x3 matrix
 * @param m Transformation matrix (homogeneous)
 * @param p Point to transform
 * @return Transformed point
 * 
 * Converts point to projective coordinates, transforms, and converts back
 */
template<typename T>
point2<T> transform(const matrix<T, 3, 3>& m, const point2<T>& p) {
    projective2<T> proj = p;  // Implicit conversion
    vector<T, 3> v = proj.vec();
    auto result = m * v;
    return projective2<T>{result[0], result[1], result[2]}.point();
}

/**
 * @brief Transform 2D point by 3x3 matrix (operator form)
 * @param m Transformation matrix (homogeneous)
 * @param p Point to transform
 * @return Transformed point
 */
template<typename T>
point2<T> operator*(const matrix<T, 3, 3>& m, const point2<T>& p) {
    return transform(m, p);
}

/**
 * @brief Transform 3D point by 4x4 matrix
 * @param m Transformation matrix (homogeneous)
 * @param p Point to transform
 * @return Transformed point
 * 
 * Converts point to projective coordinates, transforms, and converts back
 */
template<typename T>
point3<T> transform(const matrix<T, 4, 4>& m, const point3<T>& p) {
    projective3<T> proj = p;  // Implicit conversion
    vector<T, 4> v = proj.vec();
    auto result = m * v;
    return projective3<T>{result[0], result[1], result[2], result[3]}.point();
}

/**
 * @brief Transform 3D point by 4x4 matrix (operator form)
 * @param m Transformation matrix (homogeneous)
 * @param p Point to transform
 * @return Transformed point
 */
template<typename T>
point3<T> operator*(const matrix<T, 4, 4>& m, const point3<T>& p) {
    return transform(m, p);
}

// Geometric operations

/**
 * @brief Calculate distance between two 2D points
 * @param a First point
 * @param b Second point
 * @return Euclidean distance
 */
template<typename T>
T distance(const point2<T>& a, const point2<T>& b) {
    return length(b - a);
}

/**
 * @brief Calculate squared distance between two 2D points
 * @param a First point
 * @param b Second point
 * @return Squared Euclidean distance
 */
template<typename T>
T distance_squared(const point2<T>& a, const point2<T>& b) {
    auto d = b - a;
    return dot(d, d);
}

/**
 * @brief Calculate distance between two 3D points
 * @param a First point
 * @param b Second point
 * @return Euclidean distance
 */
template<typename T>
T distance(const point3<T>& a, const point3<T>& b) {
    return length(b - a);
}

/**
 * @brief Calculate squared distance between two 3D points
 * @param a First point
 * @param b Second point
 * @return Squared Euclidean distance
 */
template<typename T>
T distance_squared(const point3<T>& a, const point3<T>& b) {
    auto d = b - a;
    return dot(d, d);
}

/**
 * @brief Calculate midpoint between two 2D points
 * @param a First point
 * @param b Second point
 * @return Point at the center
 */
template<typename T>
point2<T> midpoint(const point2<T>& a, const point2<T>& b) {
    return {(a.x + b.x) / 2, (a.y + b.y) / 2};
}

/**
 * @brief Calculate midpoint between two 3D points
 * @param a First point
 * @param b Second point
 * @return Point at the center
 */
template<typename T>
point3<T> midpoint(const point3<T>& a, const point3<T>& b) {
    return {(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2};
}

/**
 * @brief Linear interpolation between two 2D points
 * @param a Start point (t=0)
 * @param b End point (t=1)
 * @param t Interpolation parameter [0,1]
 * @return Interpolated point
 */
template<typename T>
constexpr point2<T> lerp(const point2<T>& a, const point2<T>& b, T t) {
    vector<T, 2> diff = b - a;  // b - a returns a vector
    vector<T, 2> v = t * diff;  // Evaluate the expression
    return a + v;
}

/**
 * @brief Linear interpolation between two 3D points
 * @param a Start point (t=0)
 * @param b End point (t=1)
 * @param t Interpolation parameter [0,1]
 * @return Interpolated point
 */
template<typename T>
constexpr point3<T> lerp(const point3<T>& a, const point3<T>& b, T t) {
    vector<T, 3> diff = b - a;  // b - a returns a vector
    vector<T, 3> v = t * diff;  // Evaluate the expression
    return a + v;
}

/**
 * @brief Barycentric interpolation of three 2D points
 * @param a First point
 * @param b Second point
 * @param c Third point
 * @param u Weight for point a
 * @param v Weight for point b
 * @param w Weight for point c
 * @return Weighted combination of points
 * 
 * Note: For valid barycentric coordinates, u + v + w = 1
 */
template<typename T>
constexpr point2<T> barycentric(const point2<T>& a, const point2<T>& b, 
                                const point2<T>& c, T u, T v, T w) {
    return {
        u * a.x + v * b.x + w * c.x,
        u * a.y + v * b.y + w * c.y
    };
}

/**
 * @brief Barycentric interpolation of three 3D points
 * @param a First point
 * @param b Second point
 * @param c Third point
 * @param u Weight for point a
 * @param v Weight for point b
 * @param w Weight for point c
 * @return Weighted combination of points
 * 
 * Note: For valid barycentric coordinates, u + v + w = 1
 */
template<typename T>
constexpr point3<T> barycentric(const point3<T>& a, const point3<T>& b, 
                                const point3<T>& c, T u, T v, T w) {
    return {
        u * a.x + v * b.x + w * c.x,
        u * a.y + v * b.y + w * c.y,
        u * a.z + v * b.z + w * c.z
    };
}

// Rounding operations for float to int conversions

/**
 * @brief Round floating point coordinates to nearest integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point2i round(const point2f& p) {
    return {
        static_cast<int>(std::round(p.x)),
        static_cast<int>(std::round(p.y))
    };
}

/**
 * @brief Round floating point coordinates to nearest integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point2i round(const point2d& p) {
    return {
        static_cast<int>(std::round(p.x)),
        static_cast<int>(std::round(p.y))
    };
}

/**
 * @brief Floor floating point coordinates to integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point2i floor(const point2f& p) {
    return {
        static_cast<int>(std::floor(p.x)),
        static_cast<int>(std::floor(p.y))
    };
}

/**
 * @brief Ceiling floating point coordinates to integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point2i ceil(const point2f& p) {
    return {
        static_cast<int>(std::ceil(p.x)),
        static_cast<int>(std::ceil(p.y))
    };
}

/**
 * @brief Round floating point coordinates to nearest integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point3i round(const point3f& p) {
    return {
        static_cast<int>(std::round(p.x)),
        static_cast<int>(std::round(p.y)),
        static_cast<int>(std::round(p.z))
    };
}

/**
 * @brief Round floating point coordinates to nearest integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point3i round(const point3d& p) {
    return {
        static_cast<int>(std::round(p.x)),
        static_cast<int>(std::round(p.y)),
        static_cast<int>(std::round(p.z))
    };
}

/**
 * @brief Floor floating point coordinates to integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point3i floor(const point3f& p) {
    return {
        static_cast<int>(std::floor(p.x)),
        static_cast<int>(std::floor(p.y)),
        static_cast<int>(std::floor(p.z))
    };
}

/**
 * @brief Ceiling floating point coordinates to integers
 * @param p Point with floating point coordinates
 * @return Point with integer coordinates
 */
inline point3i ceil(const point3f& p) {
    return {
        static_cast<int>(std::ceil(p.x)),
        static_cast<int>(std::ceil(p.y)),
        static_cast<int>(std::ceil(p.z))
    };
}

} // namespace euler