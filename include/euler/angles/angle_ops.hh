#pragma once

#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_traits.hh>
#include <cmath>

namespace euler {

// Conversion functions
template<typename T>
constexpr radian<T> to_radians(const degree<T>& d) {
    return radian<T>(d);
}

template<typename T>
constexpr degree<T> to_degrees(const radian<T>& r) {
    return degree<T>(r);
}

// Identity conversions
template<typename T>
constexpr radian<T> to_radians(const radian<T>& r) {
    return r;
}

template<typename T>
constexpr degree<T> to_degrees(const degree<T>& d) {
    return d;
}

// Absolute value
template<typename T, typename Unit>
constexpr angle<T, Unit> abs(const angle<T, Unit>& a) {
    return angle<T, Unit>(std::abs(a.value()));
}

// Min/max functions
template<typename T, typename Unit>
constexpr angle<T, Unit> min(const angle<T, Unit>& a, const angle<T, Unit>& b) {
    return a.value() < b.value() ? a : b;
}

template<typename T, typename Unit>
constexpr angle<T, Unit> max(const angle<T, Unit>& a, const angle<T, Unit>& b) {
    return a.value() > b.value() ? a : b;
}

// Clamp function
template<typename T, typename Unit>
constexpr angle<T, Unit> clamp(const angle<T, Unit>& val, 
                               const angle<T, Unit>& low, 
                               const angle<T, Unit>& high) {
    return min(max(val, low), high);
}

// Common angle constants as functions (for template deduction)
template<typename T = float>
constexpr degree<T> degrees(T value) {
    return degree<T>(value);
}

template<typename T = float>
constexpr radian<T> radians(T value) {
    return radian<T>(value);
}

// Angle difference (shortest path)
template<typename T>
degree<T> angle_difference(const degree<T>& a, const degree<T>& b) {
    degree<T> diff = b - a;
    return wrap(diff);
}

template<typename T>
radian<T> angle_difference(const radian<T>& a, const radian<T>& b) {
    radian<T> diff = b - a;
    return wrap(diff);
}

// Note: Use the general approx_equal() function from core/approx_equal.hh
// for comparing angles. It handles angles automatically.

// Normalize angle to unit vector components
template<typename T>
struct angle_components {
    T cos;
    T sin;
};

template<typename T>
angle_components<T> angle_to_components(const radian<T>& angle) {
    return { std::cos(angle.value()), std::sin(angle.value()) };
}

template<typename T>
angle_components<T> angle_to_components(const degree<T>& angle) {
    return angle_to_components(to_radians(angle));
}

// Create angle from components (atan2)
// Default version returns radians
template<typename T>
radian<T> angle_from_components(T y, T x) {
    return radian<T>(std::atan2(y, x));
}

// Overload that allows specifying the angle type
template<typename AngleType, typename T,
         typename = std::enable_if_t<is_angle_v<AngleType>>>
AngleType angle_from_components(T y, T x) {
    if constexpr (std::is_same_v<AngleType, radian<T>>) {
        return radian<T>(std::atan2(y, x));
    } else if constexpr (std::is_same_v<AngleType, degree<T>>) {
        return to_degrees(radian<T>(std::atan2(y, x)));
    }
}


// Angle modulation
template<typename T, typename Unit>
angle<T, Unit> mod_angle(const angle<T, Unit>& a, const angle<T, Unit>& modulus) {
    return angle<T, Unit>(std::fmod(a.value(), modulus.value()));
}

// Sign of angle
template<typename T, typename Unit>
constexpr int sign(const angle<T, Unit>& a) {
    return (a.value() > T(0)) - (a.value() < T(0));
}

} // namespace euler