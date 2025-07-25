#include <euler/math/basic.hh>
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;

TEST_CASE("Basic math functions - Power and exponential") {
    SUBCASE("Square root") {
        // Scalar
        CHECK(approx_equal(sqrt(4.0f), 2.0f));
        CHECK(approx_equal(sqrt(9.0f), 3.0f));
        CHECK(approx_equal(sqrt(0.25f), 0.5f));
        
        // Vector
        vec3f v(4.0f, 9.0f, 16.0f);
        vec3f result = sqrt(v);
        CHECK(approx_equal(result, vec3f(2.0f, 3.0f, 4.0f)));
        
        // Matrix
        matrix<float, 2, 2> m;
        m(0, 0) = 1.0f; m(0, 1) = 4.0f;
        m(1, 0) = 9.0f; m(1, 1) = 16.0f;
        
        auto m_sqrt = sqrt(m);
        CHECK(approx_equal(m_sqrt(0, 0), 1.0f));
        CHECK(approx_equal(m_sqrt(0, 1), 2.0f));
        CHECK(approx_equal(m_sqrt(1, 0), 3.0f));
        CHECK(approx_equal(m_sqrt(1, 1), 4.0f));
    }
    
    SUBCASE("Cube root") {
        // Scalar
        CHECK(approx_equal(cbrt(8.0f), 2.0f));
        CHECK(approx_equal(cbrt(27.0f), 3.0f));
        CHECK(approx_equal(cbrt(-8.0f), -2.0f));
        
        // Vector
        vec3f v(8.0f, 27.0f, 64.0f);
        vec3f result = cbrt(v);
        CHECK(approx_equal(result, vec3f(2.0f, 3.0f, 4.0f)));
    }
    
    SUBCASE("Power function") {
        // Scalar
        CHECK(approx_equal(pow(2.0f, 3.0f), 8.0f));
        CHECK(approx_equal(pow(3.0f, 2.0f), 9.0f));
        CHECK(approx_equal(pow(4.0f, 0.5f), 2.0f));
        
        // Vector with scalar exponent
        vec3f v(2.0f, 3.0f, 4.0f);
        vec3f result = pow(v, 2.0f);
        CHECK(approx_equal(result, vec3f(4.0f, 9.0f, 16.0f)));
        
        // Vector with vector exponent
        vec3f base(2.0f, 3.0f, 4.0f);
        vec3f exponent(3.0f, 2.0f, 0.5f);
        vec3f result2 = pow(base, exponent);
        CHECK(approx_equal(result2, vec3f(8.0f, 9.0f, 2.0f)));
    }
    
    SUBCASE("Exponential function") {
        // Scalar
        CHECK(approx_equal(exp(0.0f), 1.0f));
        CHECK(approx_equal(exp(1.0f), constants<float>::e));
        CHECK(approx_equal(exp(2.0f), constants<float>::e * constants<float>::e, 1e-5f));
        
        // Vector
        vec3f v(0.0f, 1.0f, 2.0f);
        vec3f result = exp(v);
        CHECK(approx_equal(result[0], 1.0f));
        CHECK(approx_equal(result[1], constants<float>::e));
        CHECK(approx_equal(result[2], std::exp(2.0f)));
    }
    
    SUBCASE("Natural logarithm") {
        // Scalar
        CHECK(approx_equal(log(1.0f), 0.0f));
        CHECK(approx_equal(log(constants<float>::e), 1.0f));
        CHECK(approx_equal(log(constants<float>::e * constants<float>::e), 2.0f));
        
        // Vector
        vec3f v(1.0f, constants<float>::e, 10.0f);
        vec3f result = log(v);
        CHECK(approx_equal(result[0], 0.0f));
        CHECK(approx_equal(result[1], 1.0f));
        CHECK(approx_equal(result[2], std::log(10.0f)));
    }
    
    SUBCASE("Base 2 and base 10 logarithm") {
        // log2
        CHECK(approx_equal(log2(1.0f), 0.0f));
        CHECK(approx_equal(log2(2.0f), 1.0f));
        CHECK(approx_equal(log2(8.0f), 3.0f));
        
        // log10
        CHECK(approx_equal(log10(1.0f), 0.0f));
        CHECK(approx_equal(log10(10.0f), 1.0f));
        CHECK(approx_equal(log10(100.0f), 2.0f));
        
        // Vector versions
        vec3f v(1.0f, 10.0f, 100.0f);
        vec3f log10_result = log10(v);
        CHECK(approx_equal(log10_result, vec3f(0.0f, 1.0f, 2.0f)));
    }
}

TEST_CASE("Basic math functions - Rounding") {
    SUBCASE("Floor function") {
        // Scalar
        CHECK(approx_equal(floor(3.7f), 3.0f));
        CHECK(approx_equal(floor(-3.7f), -4.0f));
        CHECK(approx_equal(floor(3.0f), 3.0f));
        
        // Vector
        vec3f v(3.7f, -3.7f, 5.0f);
        vec3f result = floor(v);
        CHECK(approx_equal(result, vec3f(3.0f, -4.0f, 5.0f)));
    }
    
    SUBCASE("Ceiling function") {
        // Scalar
        CHECK(approx_equal(ceil(3.2f), 4.0f));
        CHECK(approx_equal(ceil(-3.2f), -3.0f));
        CHECK(approx_equal(ceil(3.0f), 3.0f));
        
        // Vector
        vec3f v(3.2f, -3.2f, 5.0f);
        vec3f result = ceil(v);
        CHECK(approx_equal(result, vec3f(4.0f, -3.0f, 5.0f)));
    }
    
    SUBCASE("Round function") {
        // Scalar
        CHECK(approx_equal(round(3.2f), 3.0f));
        CHECK(approx_equal(round(3.7f), 4.0f));
        CHECK(approx_equal(round(-3.2f), -3.0f));
        CHECK(approx_equal(round(-3.7f), -4.0f));
        
        // Vector
        vec3f v(3.2f, 3.7f, -3.5f);
        vec3f result = round(v);
        CHECK(approx_equal(result, vec3f(3.0f, 4.0f, -4.0f)));
    }
    
    SUBCASE("Truncate function") {
        // Scalar
        CHECK(approx_equal(trunc(3.7f), 3.0f));
        CHECK(approx_equal(trunc(-3.7f), -3.0f));
        CHECK(approx_equal(trunc(3.0f), 3.0f));
        
        // Vector
        vec3f v(3.7f, -3.7f, 5.0f);
        vec3f result = trunc(v);
        CHECK(approx_equal(result, vec3f(3.0f, -3.0f, 5.0f)));
    }
    
    SUBCASE("Fractional part") {
        // Scalar
        CHECK(approx_equal(fract(3.7f), 0.7f, 1e-6f));
        CHECK(approx_equal(fract(5.0f), 0.0f));
        CHECK(approx_equal(fract(-1.3f), 0.7f, 1e-6f)); // fract(-1.3) = -1.3 - floor(-1.3) = -1.3 - (-2) = 0.7
        
        // Vector
        vec3f v(3.7f, 5.0f, 1.25f);
        vec3f result = fract(v);
        CHECK(approx_equal(result[0], 0.7f, 1e-6f));
        CHECK(approx_equal(result[1], 0.0f));
        CHECK(approx_equal(result[2], 0.25f));
    }
    
    SUBCASE("Modulo functions") {
        // mod (GLSL-style)
        CHECK(approx_equal(mod(7.0f, 3.0f), 1.0f));
        CHECK(approx_equal(mod(-7.0f, 3.0f), 2.0f)); // mod(-7, 3) = -7 - 3*floor(-7/3) = -7 - 3*(-3) = 2
        
        // fmod (C-style)
        CHECK(approx_equal(fmod(7.0f, 3.0f), 1.0f));
        CHECK(approx_equal(fmod(-7.0f, 3.0f), -1.0f));
        
        // Vector versions
        vec3f v(7.0f, -7.0f, 10.0f);
        vec3f mod_result = mod(v, 3.0f);
        CHECK(approx_equal(mod_result[0], 1.0f));
        CHECK(approx_equal(mod_result[1], 2.0f));
        CHECK(approx_equal(mod_result[2], 1.0f));
    }
}

TEST_CASE("Basic math functions - Sign and step") {
    SUBCASE("Sign function") {
        // Scalar
        CHECK(approx_equal(sign(5.0f), 1.0f));
        CHECK(approx_equal(sign(-5.0f), -1.0f));
        CHECK(approx_equal(sign(0.0f), 0.0f));
        
        // Vector
        vec3f v(5.0f, -5.0f, 0.0f);
        vec3f result = sign(v);
        CHECK(approx_equal(result, vec3f(1.0f, -1.0f, 0.0f)));
    }
    
    SUBCASE("Step function") {
        // Scalar
        CHECK(approx_equal(step(3.0f, 2.0f), 0.0f));
        CHECK(approx_equal(step(3.0f, 4.0f), 1.0f));
        CHECK(approx_equal(step(3.0f, 3.0f), 1.0f));
        
        // Vector with scalar edge
        vec3f v(2.0f, 3.0f, 4.0f);
        vec3f result = step(3.0f, v);
        CHECK(approx_equal(result, vec3f(0.0f, 1.0f, 1.0f)));
        
        // Vector with vector edge
        vec3f edges(1.0f, 3.0f, 5.0f);
        vec3f values(2.0f, 2.0f, 2.0f);
        vec3f result2 = step(edges, values);
        CHECK(approx_equal(result2, vec3f(1.0f, 0.0f, 0.0f)));
    }
}

TEST_CASE("Basic math functions - Utility functions") {
    SUBCASE("Mix function") {
        CHECK(approx_equal(mix(0.0f, 10.0f, 0.5f), 5.0f));
        CHECK(approx_equal(mix(0.0f, 10.0f, 0.0f), 0.0f));
        CHECK(approx_equal(mix(0.0f, 10.0f, 1.0f), 10.0f));
    }
    
    SUBCASE("Saturate function") {
        CHECK(approx_equal(saturate(-0.5f), 0.0f));
        CHECK(approx_equal(saturate(0.5f), 0.5f));
        CHECK(approx_equal(saturate(1.5f), 1.0f));
        
        // Vector
        vec3f v(-0.5f, 0.5f, 1.5f);
        vec3f result = saturate(v);
        CHECK(approx_equal(result, vec3f(0.0f, 0.5f, 1.0f)));
    }
    
    SUBCASE("Reciprocal function") {
        CHECK(approx_equal(rcp(2.0f), 0.5f));
        CHECK(approx_equal(rcp(4.0f), 0.25f));
        CHECK(approx_equal(rcp(0.5f), 2.0f));
    }
}

TEST_CASE("Basic math functions - Special functions") {
    SUBCASE("log1p for numerical stability") {
        float small_x = 1e-7f;
        CHECK(approx_equal(log1p(small_x), small_x, 1e-12f)); // For small x, log(1+x) ≈ x
        CHECK(approx_equal(log1p(0.0f), 0.0f));
        CHECK(approx_equal(log1p(constants<float>::e - 1.0f), 1.0f, 1e-6f));
    }
    
    SUBCASE("expm1 for numerical stability") {
        float small_x = 1e-7f;
        CHECK(approx_equal(expm1(small_x), small_x, 1e-12f)); // For small x, exp(x)-1 ≈ x
        CHECK(approx_equal(expm1(0.0f), 0.0f));
        CHECK(approx_equal(expm1(1.0f), constants<float>::e - 1.0f, 1e-6f));
    }
}

TEST_CASE("Basic math functions - Expression templates") {
    SUBCASE("Chained operations") {
        vec3f v(1.0f, 4.0f, 9.0f);
        
        // sqrt(v) + log(v) * 2
        auto expr = sqrt(v) + log(v) * 2.0f;
        vec3f result = expr;
        
        vec3f expected;
        for (unsigned int i = 0; i < 3; ++i) {
            expected[i] = std::sqrt(v[i]) + std::log(v[i]) * 2.0f;
        }
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Complex expression with multiple functions") {
        vec3f a(0.5f, 1.0f, 1.5f);
        vec3f b(2.0f, 3.0f, 4.0f);
        
        // pow(a, 2) + exp(b * 0.1f) - floor(a * 10)
        auto expr = pow(a, 2.0f) + exp(b * 0.1f) - floor(a * 10.0f);
        vec3f result = expr;
        
        vec3f expected;
        for (unsigned int i = 0; i < 3; ++i) {
            expected[i] = std::pow(a[i], 2.0f) + std::exp(b[i] * 0.1f) - std::floor(a[i] * 10.0f);
        }
        CHECK(approx_equal(result, expected, 1e-5f));
    }
}

TEST_CASE("Basic math functions - Matrix operations") {
    SUBCASE("Pointwise operations on matrices") {
        matrix<float, 2, 3> m;
        m(0, 0) = 1.0f; m(0, 1) = 4.0f; m(0, 2) = 9.0f;
        m(1, 0) = 0.5f; m(1, 1) = 2.0f; m(1, 2) = 8.0f;
        
        // sqrt
        auto sqrt_m = sqrt(m);
        CHECK(approx_equal(sqrt_m(0, 0), 1.0f));
        CHECK(approx_equal(sqrt_m(0, 1), 2.0f));
        CHECK(approx_equal(sqrt_m(0, 2), 3.0f));
        CHECK(approx_equal(sqrt_m(1, 0), std::sqrt(0.5f)));
        CHECK(approx_equal(sqrt_m(1, 1), std::sqrt(2.0f)));
        CHECK(approx_equal(sqrt_m(1, 2), std::sqrt(8.0f)));
        
        // log
        auto log_m = log(m);
        CHECK(approx_equal(log_m(0, 0), 0.0f));
        CHECK(approx_equal(log_m(0, 1), std::log(4.0f)));
        CHECK(approx_equal(log_m(0, 2), std::log(9.0f)));
        
        // sign
        matrix<float, 2, 2> m2;
        m2(0, 0) = 5.0f;  m2(0, 1) = -3.0f;
        m2(1, 0) = 0.0f;  m2(1, 1) = -7.0f;
        
        auto sign_m = sign(m2);
        CHECK(approx_equal(sign_m(0, 0), 1.0f));
        CHECK(approx_equal(sign_m(0, 1), -1.0f));
        CHECK(approx_equal(sign_m(1, 0), 0.0f));
        CHECK(approx_equal(sign_m(1, 1), -1.0f));
    }
}

TEST_CASE("Basic math functions - Absolute value") {
    SUBCASE("Scalar absolute value") {
        CHECK(approx_equal(abs(5.0f), 5.0f));
        CHECK(approx_equal(abs(-3.5f), 3.5f));
        CHECK(approx_equal(abs(0.0f), 0.0f));
        CHECK(approx_equal(abs(-42.7f), 42.7f));
        
        // Double precision
        CHECK(approx_equal(abs(5.0), 5.0));
        CHECK(approx_equal(abs(-3.5), 3.5));
    }
    
    SUBCASE("Vector absolute value") {
        vec3f v(-1.0f, 2.5f, -3.7f);
        vec3f result = abs(v);
        CHECK(approx_equal(result, vec3f(1.0f, 2.5f, 3.7f)));
        
        vec4f v2(-5.0f, 0.0f, 3.0f, -7.2f);
        vec4f result2 = abs(v2);
        CHECK(approx_equal(result2, vec4f(5.0f, 0.0f, 3.0f, 7.2f)));
    }
    
    SUBCASE("Matrix absolute value") {
        matrix<float, 2, 3> m;
        m(0, 0) = -1.0f; m(0, 1) = 2.5f; m(0, 2) = -3.0f;
        m(1, 0) = 4.0f; m(1, 1) = -5.5f; m(1, 2) = 0.0f;
        
        auto result = abs(m);
        CHECK(approx_equal(result(0, 0), 1.0f));
        CHECK(approx_equal(result(0, 1), 2.5f));
        CHECK(approx_equal(result(0, 2), 3.0f));
        CHECK(approx_equal(result(1, 0), 4.0f));
        CHECK(approx_equal(result(1, 1), 5.5f));
        CHECK(approx_equal(result(1, 2), 0.0f));
    }
    
    SUBCASE("Expression template absolute value") {
        vec3f a(-1.0f, 2.0f, -3.0f);
        vec3f b(4.0f, -5.0f, 6.0f);
        
        auto expr = abs(a + b);
        vec3f result = expr;
        CHECK(approx_equal(result, vec3f(3.0f, 3.0f, 3.0f)));
    }
}

TEST_CASE("Basic math functions - Minimum") {
    SUBCASE("Scalar minimum") {
        CHECK(approx_equal(min(3.0f, 5.0f), 3.0f));
        CHECK(approx_equal(min(7.0f, 2.0f), 2.0f));
        CHECK(approx_equal(min(-1.0f, -3.0f), -3.0f));
        CHECK(approx_equal(min(4.5f, 4.5f), 4.5f));
    }
    
    SUBCASE("Vector minimum") {
        vec3f a(1.0f, 5.0f, 3.0f);
        vec3f b(2.0f, 3.0f, 4.0f);
        vec3f result = min(a, b);
        CHECK(approx_equal(result, vec3f(1.0f, 3.0f, 3.0f)));
    }
    
    SUBCASE("Vector-scalar minimum") {
        vec3f v(1.0f, 5.0f, 3.0f);
        vec3f result = min(v, 2.5f);
        CHECK(approx_equal(result, vec3f(1.0f, 2.5f, 2.5f)));
        
        vec3f result2 = min(4.0f, v);
        CHECK(approx_equal(result2, vec3f(1.0f, 4.0f, 3.0f)));
    }
    
    SUBCASE("Matrix minimum") {
        matrix<float, 2, 2> a, b;
        a(0, 0) = 1.0f; a(0, 1) = 5.0f;
        a(1, 0) = 3.0f; a(1, 1) = 2.0f;
        
        b(0, 0) = 2.0f; b(0, 1) = 3.0f;
        b(1, 0) = 4.0f; b(1, 1) = 1.0f;
        
        auto result = min(a, b);
        CHECK(approx_equal(result(0, 0), 1.0f));
        CHECK(approx_equal(result(0, 1), 3.0f));
        CHECK(approx_equal(result(1, 0), 3.0f));
        CHECK(approx_equal(result(1, 1), 1.0f));
    }
    
    SUBCASE("Matrix-scalar minimum") {
        matrix<float, 2, 2> m;
        m(0, 0) = 1.0f; m(0, 1) = 5.0f;
        m(1, 0) = 3.0f; m(1, 1) = 2.0f;
        
        auto result = min(m, 2.5f);
        CHECK(approx_equal(result(0, 0), 1.0f));
        CHECK(approx_equal(result(0, 1), 2.5f));
        CHECK(approx_equal(result(1, 0), 2.5f));
        CHECK(approx_equal(result(1, 1), 2.0f));
    }
    
    SUBCASE("Expression template minimum") {
        vec3f a(1.0f, 5.0f, 3.0f);
        vec3f b(2.0f, 3.0f, 4.0f);
        vec3f c(0.5f, 6.0f, 2.0f);
        
        // Evaluate expressions first to avoid ambiguity
        vec3f ab = a + b;  // (3.0f, 8.0f, 7.0f)
        vec3f c2 = c * 2.0f;  // (1.0f, 12.0f, 4.0f)
        vec3f result = min(ab, c2);
        CHECK(approx_equal(result, vec3f(1.0f, 8.0f, 4.0f)));
    }
}

TEST_CASE("Basic math functions - Maximum") {
    SUBCASE("Scalar maximum") {
        CHECK(approx_equal(max(3.0f, 5.0f), 5.0f));
        CHECK(approx_equal(max(7.0f, 2.0f), 7.0f));
        CHECK(approx_equal(max(-1.0f, -3.0f), -1.0f));
        CHECK(approx_equal(max(4.5f, 4.5f), 4.5f));
    }
    
    SUBCASE("Vector maximum") {
        vec3f a(1.0f, 5.0f, 3.0f);
        vec3f b(2.0f, 3.0f, 4.0f);
        vec3f result = max(a, b);
        CHECK(approx_equal(result, vec3f(2.0f, 5.0f, 4.0f)));
    }
    
    SUBCASE("Vector-scalar maximum") {
        vec3f v(1.0f, 5.0f, 3.0f);
        vec3f result = max(v, 2.5f);
        CHECK(approx_equal(result, vec3f(2.5f, 5.0f, 3.0f)));
        
        vec3f result2 = max(4.0f, v);
        CHECK(approx_equal(result2, vec3f(4.0f, 5.0f, 4.0f)));
    }
    
    SUBCASE("Matrix maximum") {
        matrix<float, 2, 2> a, b;
        a(0, 0) = 1.0f; a(0, 1) = 5.0f;
        a(1, 0) = 3.0f; a(1, 1) = 2.0f;
        
        b(0, 0) = 2.0f; b(0, 1) = 3.0f;
        b(1, 0) = 4.0f; b(1, 1) = 1.0f;
        
        auto result = max(a, b);
        CHECK(approx_equal(result(0, 0), 2.0f));
        CHECK(approx_equal(result(0, 1), 5.0f));
        CHECK(approx_equal(result(1, 0), 4.0f));
        CHECK(approx_equal(result(1, 1), 2.0f));
    }
    
    SUBCASE("Matrix-scalar maximum") {
        matrix<float, 2, 2> m;
        m(0, 0) = 1.0f; m(0, 1) = 5.0f;
        m(1, 0) = 3.0f; m(1, 1) = 2.0f;
        
        auto result = max(m, 2.5f);
        CHECK(approx_equal(result(0, 0), 2.5f));
        CHECK(approx_equal(result(0, 1), 5.0f));
        CHECK(approx_equal(result(1, 0), 3.0f));
        CHECK(approx_equal(result(1, 1), 2.5f));
    }
    
    SUBCASE("Expression template maximum") {
        vec3f a(1.0f, 5.0f, 3.0f);
        vec3f b(2.0f, 3.0f, 4.0f);
        vec3f c(0.5f, 6.0f, 2.0f);
        
        // Evaluate expressions first to avoid ambiguity
        vec3f ab = a + b;  // (3.0f, 8.0f, 7.0f)
        vec3f c2 = c * 2.0f;  // (1.0f, 12.0f, 4.0f)
        vec3f result = max(ab, c2);
        CHECK(approx_equal(result, vec3f(3.0f, 12.0f, 7.0f)));
    }
}