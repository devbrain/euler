/**
 * @file complex_ops.hh
 * @brief Additional operations and utilities for complex numbers
 * @ingroup ComplexModule
 * 
 * @details
 * This header provides additional operations for complex numbers that don't
 * fit naturally as member functions. It includes:
 * - Complex conjugate operations
 * - Polar form construction functions
 * - Magnitude and phase extraction
 * - Real and imaginary part extraction
 * - Expression template support for efficient computations
 * - Stream output formatting
 * 
 * All functions are designed to work seamlessly with expression templates
 * for optimal performance in complex mathematical expressions.
 * 
 * @section complex_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/complex/complex_ops.hh>
 * 
 * using namespace euler;
 * using namespace euler::literals;
 * 
 * // Basic operations
 * auto z = 3.0f + 4.0_i;
 * auto z_conj = conj(z);              // 3 - 4i
 * float mag = abs(z);                 // 5.0
 * float norm_val = norm(z);           // 25.0
 * auto phase = arg(z);                // atan2(4, 3) radians
 * 
 * // Polar construction
 * auto z1 = polar(2.0f, 45.0_deg);   // √2 + √2i
 * auto z2 = polar(1.0f, pi_rad<float>); // -1 + 0i
 * 
 * // Part extraction
 * float re = real(z);                // 3.0
 * float im = imag(z);                // 4.0
 * 
 * // Stream output
 * std::cout << z;                    // Prints: 3+4i
 * @endcode
 */
#pragma once

#include <euler/complex/complex.hh>
#include <euler/core/expression.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/degree.hh>
#include <cmath>

namespace euler {

/**
 * @brief Complex conjugate
 * @ingroup ComplexModule
 * @param z Complex number
 * @return Conjugate of z
 * 
 * @details
 * Returns the complex conjugate, which reflects the complex number
 * across the real axis: conj(a + bi) = a - bi
 * 
 * Properties:
 * - conj(conj(z)) = z
 * - conj(z1 + z2) = conj(z1) + conj(z2)
 * - conj(z1 * z2) = conj(z1) * conj(z2)
 * - z * conj(z) = |z|²
 */
template<typename T>
inline constexpr complex<T> conj(const complex<T>& z) {
    return complex<T>(z.real(), -z.imag());
}

/**
 * @brief Create complex number from polar coordinates (radians)
 * @ingroup ComplexModule
 * @param magnitude The magnitude (radius)
 * @param phase_radians The phase angle in radians
 * @return Complex number z = magnitude * exp(i * phase_radians)
 * 
 * @details
 * This free function provides a convenient way to create complex
 * numbers from polar form without explicitly using the class name.
 * 
 * @see complex::polar for the static member function
 */
template<typename T>
inline complex<T> polar(T magnitude, T phase_radians) {
    return complex<T>::polar(magnitude, phase_radians);
}

/**
 * @brief Create complex number from polar coordinates (type-safe angle)
 * @ingroup ComplexModule
 * @tparam Unit The angle unit tag (degree_tag or radian_tag)
 * @param magnitude The magnitude (radius)
 * @param phase The phase angle
 * @return Complex number z = magnitude * exp(i * phase)
 * 
 * @details
 * This overload accepts type-safe angles, automatically converting
 * between degrees and radians as needed.
 * 
 * @example
 * @code
 * auto z1 = polar(2.0f, 45.0_deg);    // √2 + √2i
 * auto z2 = polar(1.0f, half_pi_rad<float>); // 0 + 1i
 * @endcode
 */
template<typename T, typename Unit>
inline complex<T> polar(T magnitude, const angle<T, Unit>& phase) {
    return complex<T>::polar(magnitude, phase);
}

/**
 * @brief Absolute value (magnitude) of complex number
 * @ingroup ComplexModule
 * @param z Complex number
 * @return |z| = sqrt(real² + imag²)
 * 
 * @details
 * Computes the magnitude of a complex number, which represents
 * the distance from the origin in the complex plane.
 * 
 * @note This free function enables ADL (Argument-Dependent Lookup)
 *       and expression template support
 */
template<typename T>
inline T abs(const complex<T>& z) {
    return z.abs();
}

/**
 * @brief Squared magnitude (norm) of complex number
 * @ingroup ComplexModule
 * @param z Complex number
 * @return |z|² = real² + imag²
 * 
 * @details
 * Returns the squared magnitude without computing the square root.
 * This is more efficient when only relative magnitudes are needed
 * or when the actual magnitude is not required.
 * 
 * @note norm(z) = z * conj(z) = real² + imag²
 */
template<typename T>
inline T norm(const complex<T>& z) {
    return z.norm();
}

/**
 * @brief Phase angle (argument) of complex number
 * @ingroup ComplexModule
 * @param z Complex number
 * @return Phase angle in radians, range [-π, π]
 * 
 * @details
 * Computes the angle from the positive real axis to the complex
 * number in the complex plane. Uses atan2 for correct quadrant.
 * 
 * @note arg(0 + 0i) returns 0
 * @see complex::arg_deg() for angle in degrees
 */
template<typename T>
inline radian<T> arg(const complex<T>& z) {
    return z.arg();
}

/**
 * @defgroup ComplexExprTemplates Complex Expression Templates
 * @ingroup ComplexModule
 * @brief Expression template support for complex operations
 * 
 * @details
 * These overloads enable complex operations to work with expression
 * templates, allowing for efficient evaluation of complex mathematical
 * expressions without temporary objects.
 * @{
 */

/**
 * @brief Complex conjugate for expression templates
 * @param expr Expression yielding complex values
 * @return Expression computing conjugate element-wise
 */
template<typename Derived, typename T>
inline auto conj(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return conj(x); });
}

/**
 * @brief Absolute value for complex expression templates
 * @param expr Expression yielding complex values
 * @return Expression computing magnitude element-wise
 */
template<typename Derived, typename T>
inline auto abs(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return abs(x); });
}

/**
 * @brief Norm for complex expression templates
 * @param expr Expression yielding complex values
 * @return Expression computing squared magnitude element-wise
 */
template<typename Derived, typename T>
inline auto norm(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return norm(x); });
}

/**
 * @brief Argument for complex expression templates
 * @param expr Expression yielding complex values
 * @return Expression computing phase angle element-wise
 */
template<typename Derived, typename T>
inline auto arg(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return arg(x); });
}

/** @} */ // end of ComplexExprTemplates

/**
 * @brief Extract real part of complex number
 * @ingroup ComplexModule
 * @param z Complex number
 * @return Real component
 * 
 * @details
 * This free function provides a uniform interface for extracting
 * the real part, consistent with std::real.
 */
template<typename T>
inline T real(const complex<T>& z) {
    return z.real();
}

/**
 * @brief Extract imaginary part of complex number
 * @ingroup ComplexModule
 * @param z Complex number
 * @return Imaginary component
 * 
 * @details
 * This free function provides a uniform interface for extracting
 * the imaginary part, consistent with std::imag.
 */
template<typename T>
inline T imag(const complex<T>& z) {
    return z.imag();
}

/**
 * @brief Extract real parts from complex expression
 * @ingroup ComplexExprTemplates
 * @param expr Expression yielding complex values
 * @return Expression computing real parts element-wise
 */
template<typename Derived, typename T>
inline auto real(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return real(x); });
}

/**
 * @brief Extract imaginary parts from complex expression
 * @ingroup ComplexExprTemplates
 * @param expr Expression yielding complex values
 * @return Expression computing imaginary parts element-wise
 */
template<typename Derived, typename T>
inline auto imag(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return imag(x); });
}

/**
 * @brief Stream output operator for complex numbers
 * @ingroup ComplexModule
 * @param os Output stream
 * @param z Complex number to output
 * @return Reference to the output stream
 * 
 * @details
 * Formats complex numbers in the form "a+bi" or "a-bi".
 * Special cases:
 * - Pure real: "5" becomes "5+0i"
 * - Pure imaginary: "3i" becomes "0+3i"
 * - Negative imaginary: "2-3i" (no extra + sign)
 * 
 * @example
 * @code
 * complex<float> z(3, -4);
 * std::cout << z;  // Outputs: 3-4i
 * @endcode
 */
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const complex<T>& z) {
    os << z.real();
    if (z.imag() >= 0) {
        os << "+" << z.imag() << "i";
    } else {
        os << z.imag() << "i";
    }
    return os;
}

} // namespace euler