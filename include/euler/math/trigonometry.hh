#pragma once

#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <euler/core/expression.hh>
#include <euler/core/error.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_traits.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/complex/complex.hh>
#include <cmath>

namespace euler {

// ============================================================================
// Basic Trigonometric Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Sine function
// ----------------------------------------------------------------------------

// Scalar version (assumes radians)
template<typename T>
inline auto sin(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::sin(x);
}

// Radian angle version
template<typename T>
inline T sin(const radian<T>& angle) {
    return std::sin(angle.value());
}

// Degree angle version (converts to radians internally)
template<typename T>
inline T sin(const degree<T>& angle) {
    return std::sin(angle.value() * constants<T>::deg_to_rad);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> sin(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::sin(v[i]);
    }
    return result;
}

// Vector of angles version
template<typename T, size_t N, typename Unit>
inline vector<T, N> sin(const vector<angle<T, Unit>, N>& angles) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = sin(angles[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> sin(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::sin(m(i, j));
        }
    }
    return result;
}

// Complex version
// sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
template<typename T>
inline complex<T> sin(const complex<T>& z) {
    T real_part = std::sin(z.real()) * std::cosh(z.imag());
    T imag_part = std::cos(z.real()) * std::sinh(z.imag());
    return complex<T>(real_part, imag_part);
}

// Expression template version
template<typename Derived, typename T>
inline auto sin(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::sin(x); });
}

// ----------------------------------------------------------------------------
// Cosine function
// ----------------------------------------------------------------------------

// Scalar version (assumes radians)
template<typename T>
inline auto cos(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::cos(x);
}

// Radian angle version
template<typename T>
inline T cos(const radian<T>& angle) {
    return std::cos(angle.value());
}

// Degree angle version (converts to radians internally)
template<typename T>
inline T cos(const degree<T>& angle) {
    return std::cos(angle.value() * constants<T>::deg_to_rad);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> cos(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::cos(v[i]);
    }
    return result;
}

// Vector of angles version
template<typename T, size_t N, typename Unit>
inline vector<T, N> cos(const vector<angle<T, Unit>, N>& angles) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = cos(angles[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> cos(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::cos(m(i, j));
        }
    }
    return result;
}

// Complex version
// cos(a+bi) = cos(a)cosh(b) - i sin(a)sinh(b)
template<typename T>
inline complex<T> cos(const complex<T>& z) {
    T real_part = std::cos(z.real()) * std::cosh(z.imag());
    T imag_part = -std::sin(z.real()) * std::sinh(z.imag());
    return complex<T>(real_part, imag_part);
}

// Expression template version
template<typename Derived, typename T>
inline auto cos(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::cos(x); });
}

// ----------------------------------------------------------------------------
// Tangent function
// ----------------------------------------------------------------------------

// Scalar version (assumes radians)
template<typename T>
inline auto tan(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::tan(x);
}

// Radian angle version
template<typename T>
inline T tan(const radian<T>& angle) {
    return std::tan(angle.value());
}

// Degree angle version (converts to radians internally)
template<typename T>
inline T tan(const degree<T>& angle) {
    return std::tan(angle.value() * constants<T>::deg_to_rad);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> tan(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::tan(v[i]);
    }
    return result;
}

// Vector of angles version
template<typename T, size_t N, typename Unit>
inline vector<T, N> tan(const vector<angle<T, Unit>, N>& angles) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = tan(angles[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> tan(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::tan(m(i, j));
        }
    }
    return result;
}

// Complex version
// tan(z) = sin(z) / cos(z)
template<typename T>
inline complex<T> tan(const complex<T>& z) {
    return sin(z) / cos(z);
}

// Expression template version
template<typename Derived, typename T>
inline auto tan(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::tan(x); });
}

// ----------------------------------------------------------------------------
// Combined sin/cos for efficiency
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline std::pair<T, T> sincos(T x) {
    return {std::sin(x), std::cos(x)};
}

// Radian angle version
template<typename T>
inline std::pair<T, T> sincos(const radian<T>& angle) {
    T value = angle.value();
    return {std::sin(value), std::cos(value)};
}

// Degree angle version
template<typename T>
inline std::pair<T, T> sincos(const degree<T>& angle) {
    T radians = angle.value() * constants<T>::deg_to_rad;
    return {std::sin(radians), std::cos(radians)};
}

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Arcsine function
// ----------------------------------------------------------------------------

// Scalar version (returns radians)
template<typename T>
inline auto asin(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, radian<T>> {
    EULER_CHECK(x >= T(-1) && x <= T(1), error_code::invalid_argument, "asin: argument ", x, " outside [-1, 1]");
    return radian<T>(std::asin(x));
}

// Vector version (pointwise, returns vector of radians)
template<typename T, size_t N>
inline vector<radian<T>, N> asin(const vector<T, N>& v) {
    vector<radian<T>, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] >= T(-1) && v[i] <= T(1), error_code::invalid_argument, "asin: argument at index ", i, ": ", v[i], " outside [-1, 1]");
        result[i] = asin(v[i]);
    }
    return result;
}

// Matrix version (pointwise, returns matrix of radians)
template<typename T, size_t M, size_t N>
inline matrix<radian<T>, M, N> asin(const matrix<T, M, N>& m) {
    matrix<radian<T>, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) >= T(-1) && m(i, j) <= T(1), error_code::invalid_argument, "asin: argument at (", i, ",", j, "): ", m(i, j), " outside [-1, 1]");
            result(i, j) = asin(m(i, j));
        }
    }
    return result;
}

// Degree version
template<typename T>
inline degree<T> asin_deg(T x) {
    return degree<T>(asin(x));
}

// ----------------------------------------------------------------------------
// Arccosine function
// ----------------------------------------------------------------------------

// Scalar version (returns radians)
template<typename T>
inline auto acos(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, radian<T>> {
    EULER_CHECK(x >= T(-1) && x <= T(1), error_code::invalid_argument, "acos: argument ", x, " outside [-1, 1]");
    return radian<T>(std::acos(x));
}

// Vector version (pointwise, returns vector of radians)
template<typename T, size_t N>
inline vector<radian<T>, N> acos(const vector<T, N>& v) {
    vector<radian<T>, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] >= T(-1) && v[i] <= T(1), error_code::invalid_argument, "acos: argument at index ", i, ": ", v[i], " outside [-1, 1]");
        result[i] = acos(v[i]);
    }
    return result;
}

// Matrix version (pointwise, returns matrix of radians)
template<typename T, size_t M, size_t N>
inline matrix<radian<T>, M, N> acos(const matrix<T, M, N>& m) {
    matrix<radian<T>, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) >= T(-1) && m(i, j) <= T(1), error_code::invalid_argument, "acos: argument at (", i, ",", j, "): ", m(i, j), " outside [-1, 1]");
            result(i, j) = acos(m(i, j));
        }
    }
    return result;
}

// Degree version
template<typename T>
inline degree<T> acos_deg(T x) {
    return degree<T>(acos(x));
}

// ----------------------------------------------------------------------------
// Arctangent function
// ----------------------------------------------------------------------------

// Scalar version (returns radians)
template<typename T>
inline auto atan(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, radian<T>> {
    return radian<T>(std::atan(x));
}

// Vector version (pointwise, returns vector of radians)
template<typename T, size_t N>
inline vector<radian<T>, N> atan(const vector<T, N>& v) {
    vector<radian<T>, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = atan(v[i]);
    }
    return result;
}

// Matrix version (pointwise, returns matrix of radians)
template<typename T, size_t M, size_t N>
inline matrix<radian<T>, M, N> atan(const matrix<T, M, N>& m) {
    matrix<radian<T>, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = atan(m(i, j));
        }
    }
    return result;
}

// Degree version
template<typename T>
inline degree<T> atan_deg(T x) {
    return degree<T>(atan(x));
}

// ----------------------------------------------------------------------------
// Two-argument arctangent function
// ----------------------------------------------------------------------------

// Scalar version (returns radians)
template<typename T>
inline auto atan2(T y, T x) -> std::enable_if_t<std::is_arithmetic_v<T>, radian<T>> {
    return radian<T>(std::atan2(y, x));
}

// Vector version (pointwise, returns vector of radians)
template<typename T, size_t N>
inline vector<radian<T>, N> atan2(const vector<T, N>& y, const vector<T, N>& x) {
    vector<radian<T>, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = atan2(y[i], x[i]);
    }
    return result;
}

// Degree version
template<typename T>
inline degree<T> atan2_deg(T y, T x) {
    return degree<T>(atan2(y, x));
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Hyperbolic sine
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto sinh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::sinh(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> sinh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::sinh(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> sinh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::sinh(m(i, j));
        }
    }
    return result;
}

// Complex version
// sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
template<typename T>
inline complex<T> sinh(const complex<T>& z) {
    T real_part = std::sinh(z.real()) * std::cos(z.imag());
    T imag_part = std::cosh(z.real()) * std::sin(z.imag());
    return complex<T>(real_part, imag_part);
}

// Expression template version
template<typename Derived, typename T>
inline auto sinh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::sinh(x); });
}

// ----------------------------------------------------------------------------
// Hyperbolic cosine
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto cosh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::cosh(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> cosh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::cosh(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> cosh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::cosh(m(i, j));
        }
    }
    return result;
}

// Complex version
// cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
template<typename T>
inline complex<T> cosh(const complex<T>& z) {
    T real_part = std::cosh(z.real()) * std::cos(z.imag());
    T imag_part = std::sinh(z.real()) * std::sin(z.imag());
    return complex<T>(real_part, imag_part);
}

// Expression template version
template<typename Derived, typename T>
inline auto cosh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::cosh(x); });
}

// ----------------------------------------------------------------------------
// Hyperbolic tangent
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto tanh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::tanh(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> tanh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::tanh(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> tanh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::tanh(m(i, j));
        }
    }
    return result;
}

// Complex version
// tanh(z) = sinh(z) / cosh(z)
template<typename T>
inline complex<T> tanh(const complex<T>& z) {
    return sinh(z) / cosh(z);
}

// Expression template version
template<typename Derived, typename T>
inline auto tanh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::tanh(x); });
}

// ----------------------------------------------------------------------------
// Inverse hyperbolic functions
// ----------------------------------------------------------------------------

// Inverse hyperbolic sine
template<typename T>
inline auto asinh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::asinh(x);
}

// Inverse hyperbolic cosine
template<typename T>
inline auto acosh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x >= T(1), error_code::invalid_argument, "acosh: argument ", x, " must be >= 1");
    return std::acosh(x);
}

// Inverse hyperbolic tangent
template<typename T>
inline auto atanh(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x > T(-1) && x < T(1), error_code::invalid_argument, "atanh: argument ", x, " must be in (-1, 1)");
    return std::atanh(x);
}

// Vector versions for inverse hyperbolic functions
template<typename T, size_t N>
inline vector<T, N> asinh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::asinh(v[i]);
    }
    return result;
}

template<typename T, size_t N>
inline vector<T, N> acosh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] >= T(1), error_code::invalid_argument, "acosh: argument at index ", i, ": ", v[i], " must be >= 1");
        result[i] = std::acosh(v[i]);
    }
    return result;
}

template<typename T, size_t N>
inline vector<T, N> atanh(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] > T(-1) && v[i] < T(1), error_code::invalid_argument, "atanh: argument at index ", i, ": ", v[i], " must be in (-1, 1)");
        result[i] = std::atanh(v[i]);
    }
    return result;
}

// Matrix versions for inverse hyperbolic functions
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> asinh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::asinh(m(i, j));
        }
    }
    return result;
}

template<typename T, size_t M, size_t N>
inline matrix<T, M, N> acosh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) >= T(1), error_code::invalid_argument, "acosh: argument at (", i, ",", j, "): ", m(i, j), " must be >= 1");
            result(i, j) = std::acosh(m(i, j));
        }
    }
    return result;
}

template<typename T, size_t M, size_t N>
inline matrix<T, M, N> atanh(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) > T(-1) && m(i, j) < T(1), error_code::invalid_argument, "atanh: argument at (", i, ",", j, "): ", m(i, j), " must be in (-1, 1)");
            result(i, j) = std::atanh(m(i, j));
        }
    }
    return result;
}

// Expression template versions for inverse hyperbolic functions
template<typename Derived, typename T>
inline auto asinh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::asinh(x); });
}

template<typename Derived, typename T>
inline auto acosh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::acosh(x); });
}

template<typename Derived, typename T>
inline auto atanh(const expression<Derived, T>& expr) {
    return make_unary_expression(expr, [](const auto& x) { return std::atanh(x); });
}

// ============================================================================
// Conversion Functions
// ============================================================================

// Convert degrees to radians (raw values)
template<typename T>
inline T to_radians_raw(T degrees) {
    return degrees * constants<T>::deg_to_rad;
}

// Convert radians to degrees (raw values)
template<typename T>
inline T to_degrees_raw(T radians) {
    return radians * constants<T>::rad_to_deg;
}

// These overloads are already handled by angle conversion constructors,
// but we provide them for completeness
template<typename T>
inline radian<T> to_radians_angle(const degree<T>& deg) {
    return radian<T>(deg);
}

template<typename T>
inline degree<T> to_degrees_angle(const radian<T>& rad) {
    return degree<T>(rad);
}

} // namespace euler