/**
 * @file angle.hh
 * @brief Type-safe angle representation with automatic unit conversion
 * @ingroup AnglesModule
 * 
 * @details
 * This header provides the core angle class template that ensures type safety
 * when working with angular measurements. It prevents accidental mixing of
 * degrees and radians through the type system while providing seamless
 * conversion when needed.
 * 
 * @section angle_usage Usage Example
 * @code{.cpp}
 * #include <euler/angles/angle.hh>
 * 
 * using namespace euler;
 * 
 * // Create angles with explicit units
 * angle<float, degree_tag> deg_angle(45.0f);
 * angle<float, radian_tag> rad_angle(constants<float>::pi / 4);
 * 
 * // Automatic conversion when needed
 * angle<float, radian_tag> converted = deg_angle;  // 45° → π/4 rad
 * 
 * // Arithmetic operations preserve units
 * auto sum = deg_angle + angle<float, degree_tag>(30.0f);  // 75°
 * auto scaled = deg_angle * 2.0f;  // 90°
 * 
 * // Get raw value when needed
 * float raw_degrees = deg_angle.value();  // 45.0f
 * @endcode
 * 
 * @section angle_design Design Rationale
 * - Zero-cost abstraction: no runtime overhead compared to raw floats
 * - Type safety: prevents unit mix-ups at compile time
 * - Intuitive API: supports all expected arithmetic operations
 * - Extensible: easy to add new angle units if needed
 */
#pragma once

#include <euler/core/types.hh>
#include <compare>
#include <euler/core/error.hh>
#include <type_traits>
#include <cmath>

namespace euler {

/**
 * @defgroup AnglesModule Angles Module
 * @brief Type-safe angle representation and operations
 * 
 * @details
 * The angles module provides a comprehensive system for working with angular
 * measurements in a type-safe manner. It includes:
 * - Type-safe angle representation with automatic unit conversion
 * - Specialized types for degrees and radians
 * - Trigonometric operations that work seamlessly with angle types
 * - Utilities for angle wrapping and normalization
 * @{
 */

/**
 * @struct degree_tag
 * @brief Tag type to identify degree units
 * @ingroup AnglesModule
 * 
 * @details
 * This empty struct serves as a compile-time tag to indicate that an angle
 * is measured in degrees. It's used as a template parameter for the angle
 * class to provide type safety.
 */
struct degree_tag {};

/**
 * @struct radian_tag
 * @brief Tag type to identify radian units
 * @ingroup AnglesModule
 * 
 * @details
 * This empty struct serves as a compile-time tag to indicate that an angle
 * is measured in radians. It's used as a template parameter for the angle
 * class to provide type safety.
 */
struct radian_tag {};

/**
 * @struct angle_conversion
 * @brief Traits for converting between angle units
 * @ingroup AnglesModule
 * @tparam FromUnit Source angle unit tag (degree_tag or radian_tag)
 * @tparam ToUnit Target angle unit tag (degree_tag or radian_tag)
 * 
 * @details
 * This traits class provides the conversion logic between different angle units.
 * It's specialized for each valid conversion pair. The primary template is
 * intentionally left undefined to catch invalid conversions at compile time.
 */
template<typename FromUnit, typename ToUnit>
struct angle_conversion;

/**
 * @struct angle_conversion<Unit, Unit>
 * @brief Specialization for same-unit conversion (no-op)
 * @ingroup AnglesModule
 * @tparam Unit The angle unit tag (same for source and target)
 * 
 * @details
 * This specialization handles the trivial case where no conversion is needed
 * because the source and target units are the same. It simply returns the
 * input value unchanged.
 */
template<typename Unit>
struct angle_conversion<Unit, Unit> {
    /**
     * @brief Convert value (identity function)
     * @tparam T The numeric type of the angle value
     * @param value The value to convert
     * @return The same value unchanged
     * 
     * @note This is a no-op conversion optimized away at compile time
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value;
    }
};

/**
 * @struct angle_conversion<degree_tag, radian_tag>
 * @brief Specialization for degree to radian conversion
 * @ingroup AnglesModule
 * 
 * @details
 * Converts angle values from degrees to radians using the conversion factor π/180.
 * The conversion is performed at compile time when possible.
 * 
 * @see constants<T>::deg_to_rad
 */
template<>
struct angle_conversion<degree_tag, radian_tag> {
    /**
     * @brief Convert degrees to radians
     * @tparam T The numeric type of the angle value
     * @param value Angle in degrees
     * @return Angle in radians (value × π/180)
     * 
     * @note The conversion factor is taken from constants<T>::deg_to_rad
     *       for maximum precision with the given numeric type
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value * constants<T>::deg_to_rad;
    }
};

/**
 * @struct angle_conversion<radian_tag, degree_tag>
 * @brief Specialization for radian to degree conversion
 * @ingroup AnglesModule
 * 
 * @details
 * Converts angle values from radians to degrees using the conversion factor 180/π.
 * The conversion is performed at compile time when possible.
 * 
 * @see constants<T>::rad_to_deg
 */
template<>
struct angle_conversion<radian_tag, degree_tag> {
    /**
     * @brief Convert radians to degrees
     * @tparam T The numeric type of the angle value
     * @param value Angle in radians
     * @return Angle in degrees (value × 180/π)
     * 
     * @note The conversion factor is taken from constants<T>::rad_to_deg
     *       for maximum precision with the given numeric type
     */
    template<typename T>
    static constexpr T convert(T value) {
        return value * constants<T>::rad_to_deg;
    }
};

/**
 * @class angle
 * @brief Type-safe angle representation
 * @ingroup AnglesModule
 * 
 * @details
 * This class provides a type-safe way to work with angles, preventing
 * accidental mixing of degrees and radians. Conversions between units
 * are handled automatically and efficiently at compile time.
 * 
 * The angle class supports:
 * - Implicit conversion between different angle units
 * - All standard arithmetic operations
 * - Comparison operations
 * - Zero runtime overhead (compiles to raw numeric operations)
 * 
 * @tparam T The numeric type (e.g., float, double)
 * @tparam Unit The angle unit tag (degree_tag or radian_tag)
 * 
 * @note The class uses explicit constructors to prevent accidental
 *       creation from raw numeric values without clear unit intent
 * 
 * @see degree<T> for a convenient degree angle type
 * @see radian<T> for a convenient radian angle type
 */
template<typename T, typename Unit>
class angle {
    static_assert(std::is_arithmetic_v<T>, 
                  "Angle value type must be arithmetic type");

public:
    /** @brief The underlying numeric type */
    using value_type = T;
    /** @brief The angle unit tag type */
    using unit_type = Unit;
    
    /**
     * @brief Default constructor
     * @details Initializes the angle to zero
     */
    constexpr angle() = default;
    
    /**
     * @brief Construct angle from numeric value
     * @param value The angle value in the specified unit
     * @note Constructor is explicit to prevent accidental conversions
     */
    constexpr explicit angle(T value) : value_(value) {}
    
    /** @brief Copy constructor */
    constexpr angle(const angle&) = default;
    /** @brief Move constructor */
    constexpr angle(angle&&) = default;
    
    /** @brief Copy assignment operator */
    angle& operator=(const angle&) = default;
    /** @brief Move assignment operator */
    angle& operator=(angle&&) = default;
    
    /**
     * @brief Conversion constructor from different unit
     * @tparam OtherUnit The source angle unit tag
     * @param other Angle in different units to convert from
     * 
     * @details
     * This constructor enables implicit conversion between different angle units.
     * For example, you can assign a degree angle to a radian angle variable and
     * the conversion happens automatically.
     * 
     * @note The conversion is performed at compile time when possible
     */
    template<typename OtherUnit>
    constexpr angle(const angle<T, OtherUnit>& other) 
        : value_(angle_conversion<OtherUnit, Unit>::convert(other.value())) {}
    
    /**
     * @brief Implicit conversion to underlying numeric type
     * @return The raw angle value in this angle's units
     * 
     * @details
     * This conversion operator allows angles to be used seamlessly with
     * functions expecting raw numeric values, such as standard math functions.
     * 
     * @warning Be careful when using this with functions that expect specific
     *          units - the returned value is in this angle's units
     */
    constexpr operator T() const { return value_; }
    
    /**
     * @brief Get the raw angle value
     * @return The angle value in this angle's units
     * 
     * @details
     * Explicit accessor for cases where implicit conversion might be ambiguous
     * or when you want to be explicit about extracting the raw value.
     */
    constexpr T value() const { return value_; }
    
    /**
     * @brief Unary plus operator
     * @return Copy of this angle
     */
    constexpr angle operator+() const { return *this; }
    
    /**
     * @brief Unary minus operator
     * @return Negated angle
     */
    constexpr angle operator-() const { return angle(-value_); }
    
    /**
     * @brief Compound addition operator
     * @param rhs Angle to add (must be in same units)
     * @return Reference to this angle
     */
    angle& operator+=(const angle& rhs) {
        value_ += rhs.value_;
        return *this;
    }
    
    /**
     * @brief Compound subtraction operator
     * @param rhs Angle to subtract (must be in same units)
     * @return Reference to this angle
     */
    angle& operator-=(const angle& rhs) {
        value_ -= rhs.value_;
        return *this;
    }
    
    /**
     * @brief Compound multiplication operator
     * @param scale Scalar value to multiply by
     * @return Reference to this angle
     */
    angle& operator*=(T scale) {
        value_ *= scale;
        return *this;
    }
    
    /**
     * @brief Compound division operator
     * @param scale Scalar value to divide by
     * @return Reference to this angle
     * @warning Division by zero results in undefined behavior
     */
    angle& operator/=(T scale) {
        EULER_CHECK(std::abs(scale) > constants<T>::epsilon, 
                    error_code::invalid_argument,
                    "Division by zero or near-zero value in angle: ", scale);
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
    
    // C++20 Three-way comparison operator
    constexpr auto operator<=>(const angle& rhs) const = default;

    // Equality comparison (can use default in C++20)
    constexpr bool operator==(const angle& rhs) const = default;

private:
    T value_{};
};

/**
 * @brief Add two angles
 * @ingroup AnglesModule
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Sum of the angles
 * @note Both angles must be in the same units
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> operator+(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(lhs.value() + rhs.value());
}

/**
 * @brief Subtract two angles
 * @ingroup AnglesModule
 * @param lhs Left operand
 * @param rhs Right operand
 * @return Difference of the angles
 * @note Both angles must be in the same units
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> operator-(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(lhs.value() - rhs.value());
}

/**
 * @brief Multiply angle by scalar
 * @ingroup AnglesModule
 * @param lhs The angle
 * @param scale The scalar multiplier
 * @return Scaled angle
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> operator*(const angle<T, Unit>& lhs, T scale) {
    return angle<T, Unit>(lhs.value() * scale);
}

/**
 * @brief Multiply scalar by angle
 * @ingroup AnglesModule
 * @param scale The scalar multiplier
 * @param rhs The angle
 * @return Scaled angle
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> operator*(T scale, const angle<T, Unit>& rhs) {
    return angle<T, Unit>(scale * rhs.value());
}

/**
 * @brief Divide angle by scalar
 * @ingroup AnglesModule
 * @param lhs The angle
 * @param scale The scalar divisor
 * @return Scaled angle
 */
template<typename T, typename Unit>
constexpr angle<T, Unit> operator/(const angle<T, Unit>& lhs, T scale) {
    return angle<T, Unit>(lhs.value() / scale);
}

/**
 * @brief Divide two angles to get dimensionless ratio
 * @ingroup AnglesModule
 * @param lhs Numerator angle
 * @param rhs Denominator angle
 * @return Dimensionless ratio
 * @note Both angles must be in the same units
 */
template<typename T, typename Unit>
constexpr T operator/(const angle<T, Unit>& lhs, const angle<T, Unit>& rhs) {
    return lhs.value() / rhs.value();
}

/**
 * @brief Wrap angle to [-π, π] radians or [-180°, 180°] degrees
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a The angle to wrap
 * @return Wrapped angle in range [-π, π] or [-180°, 180°]
 * 
 * @details
 * This function normalizes an angle to the principal value range.
 * Useful for comparing angles or ensuring canonical representation.
 * 
 * @see wrap_positive() for [0, 2π] range
 */
template<typename T, typename Unit>
angle<T, Unit> wrap(const angle<T, Unit>& a);

/**
 * @brief Wrap angle to [0, 2π] radians or [0°, 360°] degrees  
 * @ingroup AnglesModule
 * @tparam T The numeric type
 * @tparam Unit The angle unit tag
 * @param a The angle to wrap
 * @return Wrapped angle in range [0, 2π] or [0°, 360°]
 * 
 * @details
 * This function normalizes an angle to the positive range.
 * Useful when negative angles are not desired.
 * 
 * @see wrap() for [-π, π] range
 */
template<typename T, typename Unit>
angle<T, Unit> wrap_positive(const angle<T, Unit>& a);

/** @} */ // end of AnglesModule

} // namespace euler