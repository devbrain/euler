/**
 * @file dda_math.hh
 * @brief Math functions wrapper for DDA module using Euler math library
 * @ingroup DDAModule
 */
#pragma once

#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>
#include <euler/coordinates/point2.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/core/expression.hh>

namespace euler::dda {

// Import all math functions from euler namespace
using euler::abs;
using euler::sqrt;
using euler::sin;
using euler::cos;
using euler::tan;
using euler::asin;
using euler::acos;
using euler::atan;
using euler::atan2;
using euler::floor;
using euler::ceil;
using euler::round;
using euler::pow;
using euler::min;
using euler::max;
using euler::clamp;

// Import vector operations from euler namespace
using euler::length;
using euler::dot;
using euler::normalize;

// Helper functions for point2 (since euler vector ops work on vectors)
template<typename T>
inline T length(const point2<T>& p) {
    return length(vec2<T>(p.x, p.y));
}

template<typename T>
inline T length_squared(const point2<T>& p) {
    vec2<T> v(p.x, p.y);
    return dot(v, v);
}

// 2D cross product (returns scalar z-component)
template<typename T>
inline T cross_2d(const vec2<T>& a, const vec2<T>& b) {
    return a[0] * b[1] - a[1] * b[0];
}

template<typename T>
inline T cross_2d(const point2<T>& a, const point2<T>& b) {
    return a.x * b.y - a.y * b.x;
}

// Handle expression templates by evaluating them first
template<typename E1, typename E2, typename T = typename E1::value_type>
inline auto cross_2d(const expression<E1, T>& a, const expression<E2, T>& b) 
    -> std::enable_if_t<E1::static_rows == 2 && E2::static_rows == 2, T> {
    auto av = a.eval();
    auto bv = b.eval();
    return av[0] * bv[1] - av[1] * bv[0];
}

// Mixed cases with expression templates
template<typename E, typename T>
inline auto cross_2d(const expression<E, T>& a, const vec2<T>& b) 
    -> std::enable_if_t<E::static_rows == 2, T> {
    auto av = a.eval();
    return av[0] * b[1] - av[1] * b[0];
}

template<typename E, typename T>
inline auto cross_2d(const vec2<T>& a, const expression<E, T>& b)
    -> std::enable_if_t<E::static_rows == 2, T> {
    auto bv = b.eval();
    return a[0] * bv[1] - a[1] * bv[0];
}

template<typename E, typename T>
inline auto cross_2d(const expression<E, T>& a, const point2<T>& b)
    -> std::enable_if_t<E::static_rows == 2, T> {
    auto av = a.eval();
    return av[0] * b.y - av[1] * b.x;
}

template<typename E, typename T>
inline auto cross_2d(const point2<T>& a, const expression<E, T>& b)
    -> std::enable_if_t<E::static_rows == 2, T> {
    auto bv = b.eval();
    return a.x * bv[1] - a.y * bv[0];
}

// Perpendicular vector (rotate 90 degrees counter-clockwise)
template<typename T>
inline vec2<T> perp(const vec2<T>& v) {
    return vec2<T>(-v[1], v[0]);
}

// Note: distance and distance_squared are already defined in euler/coordinates/point_ops.hh

} // namespace euler::dda