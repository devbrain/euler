#pragma once

#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <euler/core/expression.hh>
#include <euler/core/error.hh>
#include <euler/complex/complex.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_expr.hh>
#include <cmath>
#include <type_traits>

namespace euler {

// ============================================================================
// Power and Exponential Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Square root function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto sqrt(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x >= T(0), error_code::invalid_argument, "sqrt: negative argument ", x);
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::sqrt(static_cast<double>(x)));
    } else {
        return std::sqrt(x);
    }
}

// Complex version
template<typename T>
inline complex<T> sqrt(const complex<T>& z) {
    T r = z.abs();
    T arg = z.arg().value() / T(2);
    return complex<T>::polar(std::sqrt(r), arg);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> sqrt(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] >= T(0), error_code::invalid_argument, "sqrt: negative argument at index ", i, ": ", v[i]);
        result[i] = std::sqrt(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> sqrt(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) >= T(0), error_code::invalid_argument, "sqrt: negative argument at (", i, ",", j, "): ", m(i, j));
            result(i, j) = std::sqrt(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto sqrt(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::sqrt(x); });
}

// ----------------------------------------------------------------------------
// Cube root function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto cbrt(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::cbrt(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> cbrt(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::cbrt(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> cbrt(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::cbrt(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto cbrt(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::cbrt(x); });
}

// ----------------------------------------------------------------------------
// Power function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T1, typename T2>
inline auto pow(T1 base, T2 exponent) -> std::enable_if_t<
    std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>,
    decltype(std::pow(base, exponent))
> {
    // Check for negative base with non-integer exponent
    if constexpr (std::is_floating_point_v<T2>) {
        EULER_CHECK(base >= T1(0) || std::floor(exponent) == exponent, 
                    error_code::invalid_argument, 
                    "pow: negative base ", base, " with non-integer exponent ", exponent);
    }
    return std::pow(base, exponent);
}

// Complex power - base^exponent = exp(exponent * log(base))
template<typename T>
inline complex<T> pow(const complex<T>& base, const complex<T>& exponent) {
    // Handle special case of zero base
    if (base.real() == T(0) && base.imag() == T(0)) {
        if (exponent.real() == T(0) && exponent.imag() == T(0)) {
            return complex<T>(T(1), T(0)); // 0^0 = 1 by convention
        }
        return complex<T>(T(0), T(0)); // 0^z = 0 for z != 0
    }
    // General case: base^exp = exp(exp * log(base))
    return exp(exponent * log(base));
}

// Complex base, real exponent
template<typename T>
inline complex<T> pow(const complex<T>& base, T exponent) {
    return pow(base, complex<T>(exponent, T(0)));
}

// Real base, complex exponent
template<typename T>
inline complex<T> pow(T base, const complex<T>& exponent) {
    return pow(complex<T>(base, T(0)), exponent);
}

// Vector version (pointwise)
template<typename T, size_t N, typename U>
inline auto pow(const vector<T, N>& v, U exponent) -> vector<decltype(std::pow(T{}, U{})), N> {
    using result_type = decltype(std::pow(T{}, U{}));
    vector<result_type, N> result;
    for (size_t i = 0; i < N; ++i) {
        // Check for negative base with non-integer exponent
        if constexpr (std::is_floating_point_v<U>) {
            EULER_CHECK(v[i] >= T(0) || std::floor(exponent) == exponent, 
                        error_code::invalid_argument, 
                        "pow: negative base at index ", i, ": ", v[i], " with non-integer exponent ", exponent);
        }
        result[i] = std::pow(v[i], exponent);
    }
    return result;
}

// Scalar base with vector exponent (pointwise)
template<typename T, size_t N, typename U>
inline auto pow(U base, const vector<T, N>& exponent) -> std::enable_if_t<
    std::is_arithmetic_v<U>,
    vector<decltype(std::pow(U{}, T{})), N>
> {
    using result_type = decltype(std::pow(U{}, T{}));
    vector<result_type, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::pow(base, exponent[i]);
    }
    return result;
}

// Vector base and vector exponent (pointwise)
template<typename T, size_t N>
inline vector<T, N> pow(const vector<T, N>& base, const vector<T, N>& exponent) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        // Check for negative base with non-integer exponent
        if constexpr (std::is_floating_point_v<T>) {
            EULER_CHECK(base[i] >= T(0) || std::floor(exponent[i]) == exponent[i], 
                        error_code::invalid_argument, 
                        "pow: negative base at index ", i, ": ", base[i], " with non-integer exponent ", exponent[i]);
        }
        result[i] = std::pow(base[i], exponent[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N, typename U>
inline auto pow(const matrix<T, M, N>& m, U exponent) -> matrix<decltype(std::pow(T{}, U{})), M, N> {
    using result_type = decltype(std::pow(T{}, U{}));
    matrix<result_type, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::pow(m(i, j), exponent);
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T, typename U>
inline auto pow(const expression<Derived, T>& expr, U exponent) {
    return make_unary_expression(expr.self(), [exponent](const auto& x) { return std::pow(x, exponent); });
}

// ----------------------------------------------------------------------------
// Exponential function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto exp(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::exp(x);
}

// Complex version - e^(a+bi) = e^a * (cos(b) + i*sin(b))
template<typename T>
inline complex<T> exp(const complex<T>& z) {
    T exp_real = std::exp(z.real());
    T cos_imag = std::cos(z.imag());
    T sin_imag = std::sin(z.imag());
    return complex<T>(exp_real * cos_imag, exp_real * sin_imag);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> exp(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::exp(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> exp(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::exp(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto exp(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::exp(x); });
}

// ----------------------------------------------------------------------------
// Natural logarithm function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto log(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x > T(0), error_code::invalid_argument, "log: non-positive argument ", x);
    return std::log(x);
}

// Complex version - log(z) = log(|z|) + i*arg(z)
template<typename T>
inline complex<T> log(const complex<T>& z) {
    return complex<T>(std::log(z.abs()), z.arg().value());
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> log(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] > T(0), error_code::invalid_argument, "log: non-positive argument at index ", i, ": ", v[i]);
        result[i] = std::log(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> log(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) > T(0), error_code::invalid_argument, "log: non-positive argument at (", i, ",", j, "): ", m(i, j));
            result(i, j) = std::log(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto log(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::log(x); });
}

// ----------------------------------------------------------------------------
// Base 2 logarithm function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto log2(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x > T(0), error_code::invalid_argument, "log2: non-positive argument ", x);
    return std::log2(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> log2(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] > T(0), error_code::invalid_argument, "log2: non-positive argument at index ", i, ": ", v[i]);
        result[i] = std::log2(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> log2(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::log2(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto log2(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::log2(x); });
}

// ----------------------------------------------------------------------------
// Base 10 logarithm function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto log10(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x > T(0), error_code::invalid_argument, "log10: non-positive argument ", x);
    return std::log10(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> log10(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] > T(0), error_code::invalid_argument, "log10: non-positive argument at index ", i, ": ", v[i]);
        result[i] = std::log10(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> log10(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::log10(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto log10(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::log10(x); });
}

// ============================================================================
// Rounding and Remainder Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Floor function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto floor(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::floor(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> floor(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::floor(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> floor(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::floor(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto floor(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::floor(x); });
}

// ----------------------------------------------------------------------------
// Ceiling function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto ceil(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::ceil(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> ceil(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::ceil(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> ceil(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::ceil(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto ceil(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::ceil(x); });
}

// ----------------------------------------------------------------------------
// Round function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto round(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::round(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> round(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::round(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> round(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::round(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto round(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::round(x); });
}

// ----------------------------------------------------------------------------
// Truncate function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto trunc(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::trunc(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> trunc(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::trunc(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> trunc(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::trunc(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto trunc(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::trunc(x); });
}

// ----------------------------------------------------------------------------
// Fractional part function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto fract(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return x - std::floor(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> fract(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = v[i] - std::floor(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> fract(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = m(i, j) - std::floor(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto fract(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { 
        using std::floor;
        return x - floor(x);
    });
}

// ----------------------------------------------------------------------------
// Modulo function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto mod(T x, T y) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(y != T(0), error_code::invalid_argument, "mod: division by zero");
    return x - y * std::floor(x / y);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> mod(const vector<T, N>& v, T divisor) {
    EULER_CHECK(divisor != T(0), error_code::invalid_argument, "mod: division by zero");
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = v[i] - divisor * std::floor(v[i] / divisor);
    }
    return result;
}

// Vector-vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> mod(const vector<T, N>& v1, const vector<T, N>& v2) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v2[i] != T(0), error_code::invalid_argument, "mod: division by zero at index ", i);
        result[i] = v1[i] - v2[i] * std::floor(v1[i] / v2[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> mod(const matrix<T, M, N>& m, T divisor) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = m(i, j) - divisor * std::floor(m(i, j) / divisor);
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T, typename U>
inline auto mod(const expression<Derived, T>& expr, U divisor) {
    return make_unary_expression(expr.self(), [divisor](const auto& x) { 
        using std::floor;
        return x - divisor * floor(x / divisor);
    });
}

// ----------------------------------------------------------------------------
// Floating-point remainder function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto fmod(T x, T y) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(y != T(0), error_code::invalid_argument, "fmod: division by zero");
    return std::fmod(x, y);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> fmod(const vector<T, N>& v, T divisor) {
    EULER_CHECK(divisor != T(0), error_code::invalid_argument, "fmod: division by zero");
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::fmod(v[i], divisor);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> fmod(const matrix<T, M, N>& m, T divisor) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::fmod(m(i, j), divisor);
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T, typename U>
inline auto fmod(const expression<Derived, T>& expr, U divisor) {
    return make_unary_expression(expr.self(), [divisor](const auto& x) { return std::fmod(x, divisor); });
}

// ============================================================================
// Sign and Step Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Sign function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto sign(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return T((T(0) < x) - (x < T(0)));
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> sign(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = sign(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> sign(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = sign(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto sign(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) {
        using value_type = std::decay_t<decltype(x)>;
        return (x > value_type(0)) - (x < value_type(0));
    });
}

// ----------------------------------------------------------------------------
// Step function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto step(T edge, T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return x < edge ? T(0) : T(1);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> step(T edge, const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = step(edge, v[i]);
    }
    return result;
}

// Vector edge version (pointwise)
template<typename T, size_t N>
inline vector<T, N> step(const vector<T, N>& edge, const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = step(edge[i], v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> step(T edge, const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = step(edge, m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T, typename U>
inline auto step(U edge, const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [edge](const auto& x) {
        using value_type = std::decay_t<decltype(x)>;
        return x < edge ? value_type(0) : value_type(1);
    });
}

// ============================================================================
// Additional utility functions
// ============================================================================

// Mix function (alias for lerp) - already implemented in vector_ops.hh
template<typename T>
inline auto mix(T a, T b, T t) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return a * (T(1) - t) + b * t;
}

// Clamp function
template<typename T>
inline auto clamp(T x, T min_val, T max_val) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return x < min_val ? min_val : (x > max_val ? max_val : x);
}

// Saturate function (clamp to [0, 1])
template<typename T>
inline auto saturate(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return clamp(x, T(0), T(1));
}

// Vector version
template<typename T, size_t N>
inline vector<T, N> saturate(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = saturate(v[i]);
    }
    return result;
}

// Matrix version
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> saturate(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = saturate(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto saturate(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) {
        using std::min;
        using std::max;
        using value_type = std::decay_t<decltype(x)>;
        return min(max(x, value_type(0)), value_type(1));
    });
}

// Reciprocal function
template<typename T>
inline auto rcp(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    EULER_CHECK(x != T(0), error_code::invalid_argument, "rcp: division by zero");
    return T(1) / x;
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> rcp(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        EULER_CHECK(v[i] != T(0), error_code::invalid_argument, "rcp: division by zero at index ", i);
        result[i] = T(1) / v[i];
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> rcp(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EULER_CHECK(m(i, j) != T(0), error_code::invalid_argument, "rcp: division by zero at (", i, ",", j, ")");
            result(i, j) = T(1) / m(i, j);
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto rcp(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) {
        using value_type = std::decay_t<decltype(x)>;
        return value_type(1) / x;
    });
}

// ============================================================================
// Special functions for numerical stability
// ============================================================================

// log(1 + x) for small x
template<typename T>
inline auto log1p(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::log1p(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> log1p(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::log1p(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> log1p(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::log1p(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto log1p(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::log1p(x); });
}

// exp(x) - 1 for small x
template<typename T>
inline auto expm1(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::expm1(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> expm1(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::expm1(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> expm1(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::expm1(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto expm1(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::expm1(x); });
}

// ============================================================================
// Absolute value function
// ============================================================================

// Scalar version
template<typename T>
inline auto abs(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return std::abs(x);
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> abs(const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::abs(v[i]);
    }
    return result;
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> abs(const matrix<T, M, N>& m) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = std::abs(m(i, j));
        }
    }
    return result;
}

// Expression template version
template<typename Derived, typename T>
inline auto abs(const expression<Derived, T>& expr) {
    return make_unary_expression(expr.self(), [](const auto& x) { return std::abs(x); });
}

// ============================================================================
// Minimum and Maximum functions
// ============================================================================

// ----------------------------------------------------------------------------
// Minimum function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto min(T a, T b) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return a < b ? a : b;
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> min(const vector<T, N>& a, const vector<T, N>& b) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = min(a[i], b[i]);
    }
    return result;
}

// Vector-scalar version
template<typename T, size_t N>
inline vector<T, N> min(const vector<T, N>& v, T value) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = min(v[i], value);
    }
    return result;
}

// Scalar-vector version
template<typename T, size_t N>
inline vector<T, N> min(T value, const vector<T, N>& v) {
    return min(v, value);
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> min(const matrix<T, M, N>& a, const matrix<T, M, N>& b) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = min(a(i, j), b(i, j));
        }
    }
    return result;
}

// Matrix-scalar version
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> min(const matrix<T, M, N>& m, T value) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = min(m(i, j), value);
        }
    }
    return result;
}

// Scalar-matrix version
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> min(T value, const matrix<T, M, N>& m) {
    return min(m, value);
}

// Expression template version (binary)
template<typename Derived1, typename T1, typename Derived2, typename T2>
inline auto min(const expression<Derived1, T1>& expr1, const expression<Derived2, T2>& expr2) {
    return make_binary_expression(expr1.self(), expr2.self(), 
                                  [](const auto& a, const auto& b) { return min(a, b); });
}

// Expression-scalar version
template<typename Derived, typename T, typename U>
inline auto min(const expression<Derived, T>& expr, U value) {
    return make_unary_expression(expr.self(), 
                                 [value](const auto& x) { return min(x, value); });
}

// Scalar-expression version
template<typename U, typename Derived, typename T>
inline auto min(U value, const expression<Derived, T>& expr) {
    return min(expr, value);
}

// ----------------------------------------------------------------------------
// Maximum function
// ----------------------------------------------------------------------------

// Scalar version
template<typename T>
inline auto max(T a, T b) -> std::enable_if_t<std::is_arithmetic_v<T>, T> {
    return a > b ? a : b;
}

// Vector version (pointwise)
template<typename T, size_t N>
inline vector<T, N> max(const vector<T, N>& a, const vector<T, N>& b) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = max(a[i], b[i]);
    }
    return result;
}

// Vector-scalar version
template<typename T, size_t N>
inline vector<T, N> max(const vector<T, N>& v, T value) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = max(v[i], value);
    }
    return result;
}

// Scalar-vector version
template<typename T, size_t N>
inline vector<T, N> max(T value, const vector<T, N>& v) {
    return max(v, value);
}

// Matrix version (pointwise)
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> max(const matrix<T, M, N>& a, const matrix<T, M, N>& b) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = max(a(i, j), b(i, j));
        }
    }
    return result;
}

// Matrix-scalar version
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> max(const matrix<T, M, N>& m, T value) {
    matrix<T, M, N> result;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = max(m(i, j), value);
        }
    }
    return result;
}

// Scalar-matrix version
template<typename T, size_t M, size_t N>
inline matrix<T, M, N> max(T value, const matrix<T, M, N>& m) {
    return max(m, value);
}

// Expression template version (binary)
template<typename Derived1, typename T1, typename Derived2, typename T2>
inline auto max(const expression<Derived1, T1>& expr1, const expression<Derived2, T2>& expr2) {
    return make_binary_expression(expr1.self(), expr2.self(), 
                                  [](const auto& a, const auto& b) { return max(a, b); });
}

// Expression-scalar version
template<typename Derived, typename T, typename U>
inline auto max(const expression<Derived, T>& expr, U value) {
    return make_unary_expression(expr.self(), 
                                 [value](const auto& x) { return max(x, value); });
}

// Scalar-expression version
template<typename U, typename Derived, typename T>
inline auto max(U value, const expression<Derived, T>& expr) {
    return max(expr, value);
}

// ============================================================================
// Operator overloads to resolve ambiguities
// ============================================================================

// Specialization for scalar / vector to resolve ambiguity
template<typename T, size_t N>
inline vector<T, N> operator/(T s, const vector<T, N>& v) {
    vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = s / v[i];
    }
    return result;
}

// Specialization for scalar ^ vector (power) to resolve ambiguity
template<typename T, size_t N>
inline vector<T, N> operator^(T base, const vector<T, N>& exponent) {
    return pow(base, exponent);
}

// Specialization for vector ^ scalar (power) to resolve ambiguity
template<typename T, size_t N>
inline vector<T, N> operator^(const vector<T, N>& base, T exponent) {
    return pow(base, exponent);
}

} // namespace euler