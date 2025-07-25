#pragma once

#include <euler/complex/complex.hh>
#include <euler/core/expression.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/degree.hh>
#include <cmath>

namespace euler {

// Basic operations
template<typename T>
inline constexpr complex<T> conj(const complex<T>& z) {
    return complex<T>(z.real(), -z.imag());
}

// Polar construction helpers
template<typename T>
inline complex<T> polar(T magnitude, T phase_radians) {
    return complex<T>::polar(magnitude, phase_radians);
}

template<typename T, typename Unit>
inline complex<T> polar(T magnitude, const angle<T, Unit>& phase) {
    return complex<T>::polar(magnitude, phase);
}

// Absolute value for expression templates
template<typename T>
inline T abs(const complex<T>& z) {
    return z.abs();
}

// Norm (squared magnitude)
template<typename T>
inline T norm(const complex<T>& z) {
    return z.norm();
}

// Argument (phase angle)
template<typename T>
inline radian<T> arg(const complex<T>& z) {
    return z.arg();
}

// Expression template support for complex operations
template<typename Derived, typename T>
inline auto conj(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return conj(x); });
}

template<typename Derived, typename T>
inline auto abs(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return abs(x); });
}

template<typename Derived, typename T>
inline auto norm(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return norm(x); });
}

template<typename Derived, typename T>
inline auto arg(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return arg(x); });
}

// Real and imaginary part extraction
template<typename T>
inline T real(const complex<T>& z) {
    return z.real();
}

template<typename T>
inline T imag(const complex<T>& z) {
    return z.imag();
}

// Expression template support for real/imag
template<typename Derived, typename T>
inline auto real(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return real(x); });
}

template<typename Derived, typename T>
inline auto imag(const expression<Derived, complex<T>>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return imag(x); });
}

// Stream output operator
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