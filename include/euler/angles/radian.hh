#pragma once

#include <euler/angles/angle.hh>

namespace euler {

// Radian type alias
template<typename T>
using radian = angle<T, radian_tag>;

// Common type aliases
using radianf = radian<float>;
using radiand = radian<double>;

// Literal operators for radians
inline namespace literals {
    constexpr radianf operator""_rad(long double value) {
        return radianf(static_cast<float>(value));
    }
    
    constexpr radianf operator""_radf(long double value) {
        return radianf(static_cast<float>(value));
    }
    
    constexpr radiand operator""_radd(long double value) {
        return radiand(static_cast<double>(value));
    }
    
    // Integer literals
    constexpr radianf operator""_rad(unsigned long long value) {
        return radianf(static_cast<float>(value));
    }
    
    constexpr radianf operator""_radf(unsigned long long value) {
        return radianf(static_cast<float>(value));
    }
    
    constexpr radiand operator""_radd(unsigned long long value) {
        return radiand(static_cast<double>(value));
    }
}

// Angle wrapping specializations for radians
template<typename T>
radian<T> wrap(const radian<T>& r) {
    T value = r.value();
    const T two_pi_val = T(2) * constants<T>::pi;
    
    // Wrap to [-π, π]
    while (value > constants<T>::pi) {
        value -= two_pi_val;
    }
    while (value <= -constants<T>::pi) {
        value += two_pi_val;
    }
    
    return radian<T>(value);
}

template<typename T>
radian<T> wrap_positive(const radian<T>& r) {
    T value = r.value();
    const T two_pi_val = T(2) * constants<T>::pi;
    
    // Wrap to [0, 2π)
    while (value >= two_pi) {
        value -= two_pi_val;
    }
    while (value < T(0)) {
        value += two_pi_val;
    }
    
    return radian<T>(value);
}

// Angle interpolation for radians
template<typename T>
radian<T> lerp(const radian<T>& a, const radian<T>& b, T t) {
    // Handle wrap-around for shortest path
    T diff = b.value() - a.value();
    const T two_pi_val = T(2) * constants<T>::pi;
    
    // Wrap difference to [-π, π]
    while (diff > constants<T>::pi) {
        diff -= two_pi_val;
    }
    while (diff <= -constants<T>::pi) {
        diff += two_pi_val;
    }
    
    return radian<T>(a.value() + diff * t);
}

// Common angle constants
namespace angle_constants {
    // Pi radians (180 degrees)
    template<typename T = float>
    inline constexpr radian<T> pi_rad(constants<T>::pi);
    
    // Half pi radians (90 degrees)
    template<typename T = float>
    inline constexpr radian<T> half_pi_rad(constants<T>::half_pi);
    
    // Two pi radians (360 degrees)
    template<typename T = float>
    inline constexpr radian<T> two_pi_rad(T(2) * constants<T>::pi);
    
    // Quarter pi radians (45 degrees)
    template<typename T = float>
    inline constexpr radian<T> quarter_pi_rad(constants<T>::pi / T(4));
}

} // namespace euler