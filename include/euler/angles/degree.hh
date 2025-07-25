#pragma once

#include <euler/angles/angle.hh>

namespace euler {

// Degree type alias
template<typename T>
using degree = angle<T, degree_tag>;

// Common type aliases
using degreef = degree<float>;
using degreed = degree<double>;

// Literal operators for degrees
inline namespace literals {
    constexpr degreef operator""_deg(long double value) {
        return degreef(static_cast<float>(value));
    }
    
    constexpr degreef operator""_degf(long double value) {
        return degreef(static_cast<float>(value));
    }
    
    constexpr degreed operator""_degd(long double value) {
        return degreed(static_cast<double>(value));
    }
    
    // Integer literals
    constexpr degreef operator""_deg(unsigned long long value) {
        return degreef(static_cast<float>(value));
    }
    
    constexpr degreef operator""_degf(unsigned long long value) {
        return degreef(static_cast<float>(value));
    }
    
    constexpr degreed operator""_degd(unsigned long long value) {
        return degreed(static_cast<double>(value));
    }
}

// Angle wrapping specializations for degrees
template<typename T>
degree<T> wrap(const degree<T>& d) {
    T value = d.value();
    
    // Wrap to [-180, 180]
    while (value > T(180)) {
        value -= T(360);
    }
    while (value <= T(-180)) {
        value += T(360);
    }
    
    return degree<T>(value);
}

template<typename T>
degree<T> wrap_positive(const degree<T>& d) {
    T value = d.value();
    
    // Wrap to [0, 360)
    while (value >= T(360)) {
        value -= T(360);
    }
    while (value < T(0)) {
        value += T(360);
    }
    
    return degree<T>(value);
}

// Angle interpolation for degrees
template<typename T>
degree<T> lerp(const degree<T>& a, const degree<T>& b, T t) {
    // Handle wrap-around for shortest path
    T diff = b.value() - a.value();
    
    // Wrap difference to [-180, 180]
    while (diff > T(180)) {
        diff -= T(360);
    }
    while (diff <= T(-180)) {
        diff += T(360);
    }
    
    return degree<T>(a.value() + diff * t);
}

} // namespace euler