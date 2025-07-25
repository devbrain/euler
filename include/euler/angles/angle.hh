/**
 * @file angle.hh
 * @brief Type-safe angle representation with automatic unit conversion
 */
#pragma once

#include <euler/core/types.hh>
#include <type_traits>

namespace euler {

/**
 * @struct degree_tag
 * @brief Tag type to identify degree units
 */
struct degree_tag {};

/**
 * @struct radian_tag
 * @brief Tag type to identify radian units
 */
struct radian_tag {};

/**
 * @struct angle_conversion
 * @brief Traits for converting between angle units
 * @tparam FromUnit Source angle unit tag
 * @tparam ToUnit Target angle unit tag
 */
template<typename FromUnit, typename ToUnit>
struct angle_conversion;

/**
 * @struct angle_conversion<Unit, Unit>
 * @brief Specialization for same-unit conversion (no-op)
 */
template<typename Unit>
struct angle_conversion<Unit, Unit> {
    /**
     * @brief Convert value (identity function)
     * @param value The value to convert
     * @return The same value unchanged
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value;
    }
};

/**
 * @struct angle_conversion<degree_tag, radian_tag>
 * @brief Specialization for degree to radian conversion
 */
template<>
struct angle_conversion<degree_tag, radian_tag> {
    /**
     * @brief Convert degrees to radians
     * @param value Angle in degrees
     * @return Angle in radians
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value * constants<T>::deg_to_rad;
    }
};

/**
 * @struct angle_conversion<radian_tag, degree_tag>
 * @brief Specialization for radian to degree conversion
 */
template<>
struct angle_conversion<radian_tag, degree_tag> {
    /**
     * @brief Convert radians to degrees
     * @param value Angle in radians
     * @return Angle in degrees
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value * constants<T>::rad_to_deg;
    }
};

/**
 * @class angle
 * @brief Type-safe angle representation
 * 
 * This class provides a type-safe way to work with angles, preventing
 * accidental mixing of degrees and radians. Conversions between units
 * are handled automatically and efficiently.
 * 
 * @tparam T The numeric type (e.g., float, double)
 * @tparam Unit The angle unit tag (degree_tag or radian_tag)
 */
template<typename T, typename Unit>
class angle {
    static_assert(std::is_arithmetic_v<T>, 
                  "Angle value type must be arithmetic type");

public:
    using value_type = T;
    using unit_type = Unit;
    
    // Constructors
    constexpr angle() = default;
    constexpr explicit angle(T value) : value_(value) {}
    
    // Copy and move constructors
    constexpr angle(const angle&) = default;
    constexpr angle(angle&&) = default;
    
    // Assignment operators
    angle& operator=(const angle&) = default;
    angle& operator=(angle&&) = default;
    
    // Conversion constructor from different unit
    template<typename OtherUnit>
    constexpr angle(const angle<T, OtherUnit>& other) 
        : value_(angle_conversion<OtherUnit, Unit>::convert(other.value())) {}
    
    // Implicit conversion to value type for seamless usage
    constexpr operator T() const { return value_; }
    
    // Explicit value accessor
    constexpr T value() const { return value_; }
    
    // Arithmetic operators
    constexpr angle operator+() const { return *this; }
    constexpr angle operator-() const { return angle(-value_); }
    
    angle& operator+=(const angle& rhs) {
        value_ += rhs.value_;
        return *this;
    }
    
    angle& operator-=(const angle& rhs) {
        value_ -= rhs.value_;
        return *this;
    }
    
    angle& operator*=(T scale) {
        value_ *= scale;
        return *this;
    }
    
    angle& operator/=(T scale) {
        value_ /= scale;
        return *this;
    }
    
    // Increment/decrement operators
    angle& operator++() {  // Pre-increment
        ++value_;
        return *this;
    }
    
    angle operator++(int) {  // Post-increment
        angle temp(*this);
        ++value_;
        return temp;
    }
    
    angle& operator--() {  // Pre-decrement
        --value_;
        return *this;
    }
    
    angle operator--(int) {  // Post-decrement
        angle temp(*this);
        --value_;
        return temp;
    }
    
    // Comparison operators
    constexpr bool operator==(const angle& rhs) const {
        return value_ == rhs.value_;
    }
    
    constexpr bool operator!=(const angle& rhs) const {
        return value_ != rhs.value_;
    }
    
    constexpr bool operator<(const angle& rhs) const {
        return value_ < rhs.value_;
    }
    
    constexpr bool operator<=(const angle& rhs) const {
        return value_ <= rhs.value_;
    }
    
    constexpr bool operator>(const angle& rhs) const {
        return value_ > rhs.value_;
    }
    
    constexpr bool operator>=(const angle& rhs) const {
        return value_ >= rhs.value_;
    }

private:
    T value_{};
};

// Binary arithmetic operators
template<typename T, typename Unit>
constexpr angle<T, Unit> operator+(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(lhs.value() + rhs.value());
}

template<typename T, typename Unit>
constexpr angle<T, Unit> operator-(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(lhs.value() - rhs.value());
}

template<typename T, typename Unit>
constexpr angle<T, Unit> operator*(const angle<T, Unit>& lhs, T scale) {
    return angle<T, Unit>(lhs.value() * scale);
}

template<typename T, typename Unit>
constexpr angle<T, Unit> operator*(T scale, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(scale * rhs.value());
}

template<typename T, typename Unit>
constexpr angle<T, Unit> operator/(const angle<T, Unit>& lhs, T scale) {
    return angle<T, Unit>(lhs.value() / scale);
}

// Division of two angles yields a scalar ratio
template<typename T, typename Unit>
constexpr T operator/(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return lhs.value() / rhs.value();
}

// Angle wrapping utilities
template<typename T, typename Unit>
angle<T, Unit> wrap(const angle<T, Unit>& a);  // Wrap to [-π, π] or [-180°, 180°]

template<typename T, typename Unit>
angle<T, Unit> wrap_positive(const angle<T, Unit>& a);  // Wrap to [0, 2π] or [0°, 360°]

} // namespace euler