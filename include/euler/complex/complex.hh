/**
 * @file complex.hh
 * @brief Complex number implementation with angle-aware operations
 * @ingroup ComplexModule
 * 
 * @details
 * This header provides a comprehensive complex number class that integrates
 * seamlessly with the Euler library's angle types. It includes:
 * - Complex number representation with real and imaginary parts
 * - Polar form construction with type-safe angles
 * - Full arithmetic operations
 * - Conversion to/from std::complex
 * - User-defined literals for imaginary numbers
 * 
 * The implementation is designed to be a drop-in replacement for std::complex
 * with additional features for scientific computing and graphics.
 * 
 * @section complex_usage Usage Example
 * @code{.cpp}
 * #include <euler/complex/complex.hh>
 * 
 * using namespace euler;
 * using namespace euler::literals;
 * 
 * // Create complex numbers
 * complex<float> z1(3.0f, 4.0f);      // 3 + 4i
 * auto z2 = 2.0f + 3.0_i;             // 2 + 3i using literal
 * 
 * // Polar form with type-safe angles
 * auto z3 = complex<float>::polar(5.0f, 45.0_deg);
 * auto z4 = complex<float>::polar(2.0f, pi_rad<float>);
 * 
 * // Operations
 * auto sum = z1 + z2;                 // 5 + 7i
 * auto product = z1 * z2;             // -6 + 17i
 * 
 * // Polar access
 * float magnitude = z1.abs();         // 5.0
 * auto phase = z1.arg();              // atan2(4, 3) radians
 * auto phase_deg = z1.arg_deg();      // Same angle in degrees
 * @endcode
 */
#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/degree.hh>
#include <cmath>
#include <complex>

namespace euler {

/**
 * @defgroup ComplexModule Complex Numbers Module
 * @brief Complex number representation and operations
 * 
 * @details
 * The complex numbers module provides a feature-rich implementation of complex
 * arithmetic that integrates with Euler's type-safe angle system. Key features:
 * - Zero-overhead complex number representation
 * - Seamless integration with angle types for polar form
 * - Expression template support for efficient computations
 * - Interoperability with std::complex
 * @{
 */

/**
 * @class complex
 * @brief Complex number with real and imaginary parts
 * @ingroup ComplexModule
 * @tparam T The numeric type for components (float, double, etc.)
 * 
 * @details
 * This class represents complex numbers with separate real and imaginary
 * components. It provides:
 * - Cartesian form: z = real + imag*i
 * - Polar form: z = magnitude * exp(i * phase)
 * - Full arithmetic operations
 * - Integration with angle types for type-safe phase handling
 * 
 * The class is designed to have zero overhead compared to using two
 * separate floating-point values.
 * 
 * @note This class can be used as a drop-in replacement for std::complex
 *       in most scenarios, with added benefits of angle type safety.
 */
template<typename T>
class complex {
    static_assert(std::is_arithmetic_v<T>, 
                  "Complex value type must be arithmetic type");
public:
    /** @brief The underlying numeric type for real and imaginary parts */
    using value_type = T;
    
    /**
     * @brief Default constructor
     * @details Initializes both real and imaginary parts to zero
     */
    constexpr complex() = default;
    
    /**
     * @brief Construct from real part only
     * @param real The real part
     * @details Imaginary part is initialized to zero
     */
    constexpr complex(T real) : real_(real), imag_(0) {}
    
    /**
     * @brief Construct from real and imaginary parts
     * @param real The real part
     * @param imag The imaginary part
     */
    constexpr complex(T real, T imag) : real_(real), imag_(imag) {}
    
    /** @brief Copy constructor */
    constexpr complex(const complex&) = default;
    /** @brief Move constructor */
    constexpr complex(complex&&) = default;
    
    /** @brief Copy assignment operator */
    complex& operator=(const complex&) = default;
    /** @brief Move assignment operator */
    complex& operator=(complex&&) = default;
    
    /**
     * @brief Create complex number from polar form with type-safe angle
     * @tparam Unit The angle unit tag (degree_tag or radian_tag)
     * @param magnitude The magnitude (absolute value)
     * @param phase The phase angle
     * @return Complex number z = magnitude * exp(i * phase)
     * 
     * @details
     * This factory function creates a complex number from its polar
     * representation. The angle can be in any unit (degrees or radians)
     * and is automatically converted as needed.
     * 
     * @example
     * @code
     * auto z1 = complex<float>::polar(2.0f, 45.0_deg);   // √2 + √2i
     * auto z2 = complex<float>::polar(1.0f, pi_rad<float>); // -1 + 0i
     * @endcode
     */
    template<typename Unit>
    static complex polar(T magnitude, const angle<T, Unit>& phase) {
        radian<T> phase_rad(phase);
        return complex(
            magnitude * std::cos(phase_rad.value()),
            magnitude * std::sin(phase_rad.value())
        );
    }
    
    /**
     * @brief Create complex number from polar form with raw radians
     * @param magnitude The magnitude (absolute value)
     * @param phase_radians The phase angle in radians
     * @return Complex number z = magnitude * exp(i * phase_radians)
     * 
     * @details
     * This overload accepts the phase directly in radians for cases
     * where type safety is not needed or when interfacing with C APIs.
     */
    static complex polar(T magnitude, T phase_radians) {
        return complex(
            magnitude * std::cos(phase_radians),
            magnitude * std::sin(phase_radians)
        );
    }
    
    /**
     * @brief Get real part (const)
     * @return The real component
     */
    constexpr T real() const { return real_; }
    
    /**
     * @brief Get imaginary part (const)
     * @return The imaginary component
     */
    constexpr T imag() const { return imag_; }
    
    /**
     * @brief Get real part (mutable)
     * @return Reference to the real component
     */
    T& real() { return real_; }
    
    /**
     * @brief Get imaginary part (mutable)
     * @return Reference to the imaginary component
     */
    T& imag() { return imag_; }
    
    /**
     * @brief Get magnitude (absolute value)
     * @return |z| = sqrt(real² + imag²)
     * 
     * @details
     * Computes the magnitude of the complex number, which is the
     * distance from the origin in the complex plane.
     */
    T abs() const { 
        return std::sqrt(real_ * real_ + imag_ * imag_); 
    }
    
    /**
     * @brief Get squared magnitude (norm)
     * @return |z|² = real² + imag²
     * 
     * @details
     * Returns the squared magnitude, which avoids the square root
     * computation. Useful for comparisons and when the actual
     * magnitude is not needed.
     */
    constexpr T norm() const { 
        return real_ * real_ + imag_ * imag_; 
    }
    
    /**
     * @brief Get phase angle in radians
     * @return Phase angle in range [-π, π]
     * 
     * @details
     * Computes the argument (phase angle) of the complex number
     * using atan2, which correctly handles all quadrants.
     */
    radian<T> arg() const { 
        return radian<T>(std::atan2(imag_, real_)); 
    }
    
    /**
     * @brief Get phase angle in degrees
     * @return Phase angle in range [-180°, 180°]
     * 
     * @details
     * Convenience method that returns the phase angle in degrees
     * instead of radians.
     */
    degree<T> arg_deg() const { 
        return degree<T>(arg()); 
    }
    
    /**
     * @brief Compound addition operator
     * @param rhs Complex number to add
     * @return Reference to this complex number
     * @details Performs component-wise addition: (a+bi) + (c+di) = (a+c) + (b+d)i
     */
    complex& operator+=(const complex& rhs) {
        real_ += rhs.real_;
        imag_ += rhs.imag_;
        return *this;
    }
    
    /**
     * @brief Compound subtraction operator
     * @param rhs Complex number to subtract
     * @return Reference to this complex number
     * @details Performs component-wise subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
     */
    complex& operator-=(const complex& rhs) {
        real_ -= rhs.real_;
        imag_ -= rhs.imag_;
        return *this;
    }
    
    /**
     * @brief Compound multiplication operator
     * @param rhs Complex number to multiply by
     * @return Reference to this complex number
     * 
     * @details
     * Performs complex multiplication using the formula:
     * (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
     * 
     * @note Uses temporary variables to handle self-multiplication correctly
     */
    complex& operator*=(const complex& rhs) {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        T new_real = real_ * rhs.real_ - imag_ * rhs.imag_;
        T new_imag = real_ * rhs.imag_ + imag_ * rhs.real_;
        real_ = new_real;
        imag_ = new_imag;
        return *this;
    }
    
    /**
     * @brief Compound division operator
     * @param rhs Complex number to divide by
     * @return Reference to this complex number
     * 
     * @details
     * Performs complex division using the formula:
     * (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
     * 
     * @warning Division by zero (rhs = 0 + 0i) results in undefined behavior
     */
    complex& operator/=(const complex& rhs) {
        // (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
        T denominator = rhs.real_ * rhs.real_ + rhs.imag_ * rhs.imag_;
        EULER_CHECK(std::abs(denominator) > constants<T>::epsilon, 
                    error_code::invalid_argument,
                    "Division by zero complex number: ", rhs.real_, " + ", rhs.imag_, "i");
        T new_real = (real_ * rhs.real_ + imag_ * rhs.imag_) / denominator;
        T new_imag = (imag_ * rhs.real_ - real_ * rhs.imag_) / denominator;
        real_ = new_real;
        imag_ = new_imag;
        return *this;
    }
    
    /**
     * @brief Compound scalar multiplication
     * @param scalar Value to multiply by
     * @return Reference to this complex number
     * @details Scales both real and imaginary parts: (a+bi) * s = (as) + (bs)i
     */
    complex& operator*=(T ascalar) {
        real_ *= ascalar;
        imag_ *= ascalar;
        return *this;
    }
    
    /**
     * @brief Compound scalar division
     * @param scalar Value to divide by
     * @return Reference to this complex number
     * @details Scales both real and imaginary parts: (a+bi) / s = (a/s) + (b/s)i
     * @warning Division by zero results in undefined behavior
     */
    complex& operator/=(T ascalar) {
        EULER_CHECK(std::abs(ascalar) > constants<T>::epsilon,
                    error_code::invalid_argument,
                    "Division by zero or near-zero value in complex: ", ascalar);
        real_ /= ascalar;
        imag_ /= ascalar;
        return *this;
    }
    
    /**
     * @brief Unary plus operator
     * @return Copy of this complex number
     */
    constexpr complex operator+() const { return *this; }
    
    /**
     * @brief Unary minus operator (negation)
     * @return Negated complex number
     * @details Returns -(a+bi) = (-a) + (-b)i
     */
    constexpr complex operator-() const { return complex(-real_, -imag_); }
    
    /**
     * @brief Conversion constructor from std::complex
     * @tparam U The numeric type of the source complex number
     * @param c The std::complex to convert from
     * 
     * @details
     * Enables implicit conversion from std::complex with potentially
     * different precision. This allows seamless interoperability with
     * the standard library.
     */
    template<typename U>
    complex(const std::complex<U>& c) 
        : real_(static_cast<T>(c.real()))
        , imag_(static_cast<T>(c.imag())) {}
    
    /**
     * @brief Conversion operator to std::complex
     * @return std::complex with same value
     * 
     * @details
     * Enables implicit conversion to std::complex for compatibility
     * with standard library functions and third-party libraries.
     */
    operator std::complex<T>() const {
        return std::complex<T>(real_, imag_);
    }
    
    /**
     * @brief Equality comparison operator
     * @param rhs Complex number to compare with
     * @return true if both real and imaginary parts are equal
     * 
     * @note For approximate equality with tolerance, use approx_equal()
     *       from core/approx_equal.hh
     */
    constexpr bool operator==(const complex& rhs) const {
        return real_ == rhs.real_ && imag_ == rhs.imag_;
    }
    
    /**
     * @brief Inequality comparison operator
     * @param rhs Complex number to compare with
     * @return true if real or imaginary parts differ
     */
    constexpr bool operator!=(const complex& rhs) const {
        return !(*this == rhs);
    }
    
private:
    T real_{};  ///< Real component
    T imag_{};  ///< Imaginary component
};

/**
 * @brief Single-precision complex number
 * @ingroup ComplexModule
 * 
 * @details
 * Convenient type alias for `complex<float>`. This is the most commonly
 * used complex type for graphics and real-time applications.
 */
using complexf = complex<float>;

/**
 * @brief Double-precision complex number
 * @ingroup ComplexModule
 * 
 * @details
 * Convenient type alias for `complex<double>`. Use this when higher
 * precision is needed for scientific or engineering calculations.
 */
using complexd = complex<double>;

/**
 * @brief User-defined literals for imaginary numbers
 * @ingroup ComplexModule
 * 
 * @details
 * This inline namespace provides user-defined literal operators for
 * creating imaginary numbers with intuitive syntax. The literals are:
 * - `_i` - Creates a `complexf` with zero real part
 * - `_if` - Explicitly creates a `complexf` 
 * - `_id` - Explicitly creates a `complexd`
 * 
 * @example
 * @code
 * using namespace euler::literals;
 * 
 * auto z1 = 3.0f + 4.0_i;    // complexf(3, 4)
 * auto z2 = 1.0 - 2.5_i;      // complexf(1, -2.5)
 * auto z3 = 5.0_i;            // complexf(0, 5)
 * auto z4 = 3.14_id;          // complexd(0, 3.14)
 * @endcode
 */
inline namespace literals {
    /**
     * @brief Create imaginary number (single precision)
     * @param value The imaginary value
     * @return complexf with zero real part
     * @note Example: `auto z = 3.14_i;`
     */
    constexpr complexf operator""_i(long double value) {
        return complexf(0, static_cast<float>(value));
    }
    
    /**
     * @brief Explicitly create imaginary number (single precision)
     * @param value The imaginary value
     * @return complexf with zero real part
     * @note Example: `auto z = 3.14_if;`
     */
    constexpr complexf operator""_if(long double value) {
        return complexf(0, static_cast<float>(value));
    }
    
    /**
     * @brief Create imaginary number (double precision)
     * @param value The imaginary value
     * @return complexd with zero real part
     * @note Example: `auto z = 3.141592653589793_id;`
     */
    constexpr complexd operator""_id(long double value) {
        return complexd(0, static_cast<double>(value));
    }
    
    /**
     * @brief Create imaginary number from integer literal
     * @param value The imaginary value
     * @return complexf with zero real part
     * @note Example: `auto z = 5_i;`
     */
    constexpr complexf operator""_i(unsigned long long value) {
        return complexf(0, static_cast<float>(value));
    }
}

/**
 * @brief Add two complex numbers
 * @ingroup ComplexModule
 * @param lhs First complex number
 * @param rhs Second complex number
 * @return Sum of the complex numbers
 * @details Component-wise addition: (a+bi) + (c+di) = (a+c) + (b+d)i
 */
template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
    return complex<T>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}

/**
 * @brief Subtract two complex numbers
 * @ingroup ComplexModule
 * @param lhs First complex number
 * @param rhs Second complex number
 * @return Difference of the complex numbers
 * @details Component-wise subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
 */
template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
    return complex<T>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
}

/**
 * @brief Multiply two complex numbers
 * @ingroup ComplexModule
 * @param lhs First complex number
 * @param rhs Second complex number
 * @return Product of the complex numbers
 * @details Uses the formula: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
 */
template<typename T>
complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
    complex<T> result = lhs;
    result *= rhs;
    return result;
}

/**
 * @brief Divide two complex numbers
 * @ingroup ComplexModule
 * @param lhs Numerator complex number
 * @param rhs Denominator complex number
 * @return Quotient of the complex numbers
 * @details Uses complex division formula with denominator normalization
 * @warning Division by zero (0+0i) results in undefined behavior
 */
template<typename T>
complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
    complex<T> result = lhs;
    result /= rhs;
    return result;
}

/**
 * @defgroup ComplexMixedOps Mixed Real-Complex Operations
 * @ingroup ComplexModule
 * @brief Operations between real numbers and complex numbers
 * 
 * @details
 * These operators allow seamless arithmetic between real numbers and
 * complex numbers by treating real numbers as complex numbers with
 * zero imaginary part.
 * @{
 */
/**
 * @brief Add complex and real number
 * @param lhs Complex number
 * @param rhs Real number
 * @return Sum as complex number
 */
template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() + rhs, lhs.imag());
}

/**
 * @brief Add real and complex number
 * @param lhs Real number
 * @param rhs Complex number
 * @return Sum as complex number
 */
template<typename T>
constexpr complex<T> operator+(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs + rhs.real(), rhs.imag());
}

/**
 * @brief Subtract real from complex number
 * @param lhs Complex number
 * @param rhs Real number
 * @return Difference as complex number
 */
template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() - rhs, lhs.imag());
}

/**
 * @brief Subtract complex from real number
 * @param lhs Real number
 * @param rhs Complex number
 * @return Difference as complex number
 */
template<typename T>
constexpr complex<T> operator-(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs - rhs.real(), -rhs.imag());
}

/**
 * @brief Multiply complex by real number
 * @param lhs Complex number
 * @param rhs Real scalar
 * @return Scaled complex number
 */
template<typename T>
complex<T> operator*(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() * rhs, lhs.imag() * rhs);
}

/**
 * @brief Multiply real by complex number
 * @param lhs Real scalar
 * @param rhs Complex number
 * @return Scaled complex number
 */
template<typename T>
complex<T> operator*(T lhs, const complex<T>& rhs) {
    return complex<T>(lhs * rhs.real(), lhs * rhs.imag());
}

/**
 * @brief Divide complex by real number
 * @param lhs Complex number
 * @param rhs Real divisor
 * @return Quotient as complex number
 * @warning Division by zero results in undefined behavior
 */
template<typename T>
complex<T> operator/(const complex<T>& lhs, T rhs) {
    return complex<T>(lhs.real() / rhs, lhs.imag() / rhs);
}

/**
 * @brief Divide real by complex number
 * @param lhs Real numerator
 * @param rhs Complex divisor
 * @return Quotient as complex number
 * 
 * @details
 * Uses the formula: r / (a+bi) = r * (a-bi) / (a²+b²)
 * 
 * @warning Division by zero (0+0i) results in undefined behavior
 */
template<typename T>
complex<T> operator/(T lhs, const complex<T>& rhs) {
    // lhs / (a + bi) = lhs * (a - bi) / (a² + b²)
    T denominator = rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
    return complex<T>(
        lhs * rhs.real() / denominator,
        -lhs * rhs.imag() / denominator
    );
}

/** @} */ // end of ComplexMixedOps

/** @} */ // end of ComplexModule

} // namespace euler