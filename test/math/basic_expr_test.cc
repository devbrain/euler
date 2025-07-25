#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;

TEST_CASE("Math functions expression templates") {
    SUBCASE("Basic expression composition") {
        vec3f v(1.0f, 4.0f, 9.0f);
        
        // Expression should not be evaluated yet
        auto expr = sqrt(v) + log(v);
        static_assert(!std::is_same_v<decltype(expr), vec3f>, 
                      "Should be an expression, not evaluated");
        
        // Evaluation happens here
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sqrt(v[i]) + std::log(v[i]);
        }
        
        // Debug output
        INFO("v = " << v);
        INFO("result = " << result);
        INFO("expected = " << expected);
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Complex nested expressions") {
        vec3f a(0.5f, 1.0f, 2.0f);
        vec3f b(2.0f, 4.0f, 8.0f);
        
        // exp(log(a) + log(b)) = a * b
        auto expr = exp(log(a) + log(b));
        vec3f result = expr;
        
        vec3f expected = a * b;
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Mixed math and vector operations") {
        vec3f v1(1.0f, 2.0f, 3.0f);
        vec3f v2(0.5f, 1.0f, 1.5f);
        
        // sqrt(dot(v1, v2)) * normalize(v1) + pow(v2, 2)
        float dot_val = dot(v1, v2);
        auto expr = sqrt(dot_val) * normalize(v1) + pow(v2, 2.0f);
        vec3f result = expr;
        
        // Manual calculation
        vec3f norm_v1 = normalize(v1);
        vec3f pow_v2 = pow(v2, 2.0f);
        vec3f expected = std::sqrt(dot_val) * norm_v1 + pow_v2;
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Expression with trigonometric and basic functions") {
        vec3f angles(0.5f, 1.0f, 1.5f);
        
        // log(1 + sin(angles)^2) + sqrt(cos(angles)^2)
        auto expr = log(1.0f + pow(sin(angles), 2.0f)) + sqrt(pow(cos(angles), 2.0f));
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            float sin_val = std::sin(angles[i]);
            float cos_val = std::cos(angles[i]);
            expected[i] = std::log(1.0f + sin_val * sin_val) + std::sqrt(cos_val * cos_val);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Expression with rounding functions") {
        vec3f v(3.2f, 4.7f, -2.8f);
        
        // floor(v) + fract(v) should equal v
        auto expr = floor(v) + fract(v);
        vec3f result = expr;
        
        CHECK(approx_equal(result, v, 1e-6f));
    }
    
    SUBCASE("Expression with min/max/clamp") {
        vec3f a(0.5f, 1.5f, 2.5f);
        vec3f b(1.0f, 1.0f, 1.0f);
        
        // saturate(sin(a)) + step(0.5f, cos(b))
        auto expr = saturate(sin(a)) + step(0.5f, cos(b));
        vec3f result = expr;
        
        // Manual calculation
        vec3f sin_a = sin(a);
        vec3f cos_b = cos(b);
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            float sat = sin_a[i] < 0.0f ? 0.0f : (sin_a[i] > 1.0f ? 1.0f : sin_a[i]);
            float stp = cos_b[i] < 0.5f ? 0.0f : 1.0f;
            expected[i] = sat + stp;
        }
        
        CHECK(approx_equal(result, expected));
    }
}

TEST_CASE("Math functions matrix expressions") {
    SUBCASE("Matrix expression templates") {
        matrix<float, 2, 2> m;
        m(0, 0) = 1.0f; m(0, 1) = 4.0f;
        m(1, 0) = 9.0f; m(1, 1) = 16.0f;
        
        // sqrt(m) * 2 + log(m)
        auto expr = sqrt(m) * 2.0f + log(m);
        matrix<float, 2, 2> result = expr;
        
        // Manual calculation
        matrix<float, 2, 2> expected;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                expected(i, j) = std::sqrt(m(i, j)) * 2.0f + std::log(m(i, j));
            }
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Matrix operations with functions") {
        matrix<float, 2, 2> A;
        A(0, 0) = 2.0f; A(0, 1) = 1.0f;
        A(1, 0) = 1.0f; A(1, 1) = 2.0f;
        
        matrix<float, 2, 2> B;
        B(0, 0) = 1.0f; B(0, 1) = 0.0f;
        B(1, 0) = 0.0f; B(1, 1) = 1.0f;
        
        // exp(A * B * 0.1f) - floor(A + B)
        auto expr = exp(A * B * 0.1f) - floor(A + B);
        matrix<float, 2, 2> result = expr;
        
        // Manual calculation
        matrix<float, 2, 2> AB = A * B;
        matrix<float, 2, 2> AB_scaled = AB * 0.1f;
        matrix<float, 2, 2> exp_AB = exp(AB_scaled);
        matrix<float, 2, 2> AplusB = A + B;
        matrix<float, 2, 2> floor_AplusB = floor(AplusB);
        matrix<float, 2, 2> expected = exp_AB - floor_AplusB;
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
}

TEST_CASE("Math functions performance patterns") {
    SUBCASE("Single evaluation of complex expression") {
        vec4f a(0.1f, 0.2f, 0.3f, 0.4f);
        vec4f b(1.0f, 2.0f, 3.0f, 4.0f);
        vec4f c(0.5f, 0.5f, 0.5f, 0.5f);
        
        // Store intermediate results to avoid dangling references
        vec4f pow_a = pow(a, 2.0f);
        vec4f b_scaled = b * 0.1f;
        vec4f exp_b = exp(b_scaled);
        vec4f sqrt_c = sqrt(c);
        vec4f b_plus_1 = b + 1.0f;
        vec4f log_b_plus_1 = log(b_plus_1);
        vec4f a_times_10 = a * 10.0f;
        vec4f floor_a = floor(a_times_10);
        vec4f b_times_half = b * 0.5f;
        vec4f ceil_b = ceil(b_times_half);
        
        // This complex expression should compile to a single loop
        auto expr = pow_a * exp_b + 
                   sqrt_c * log_b_plus_1 - 
                   floor_a / ceil_b;
        
        // Should still be an expression
        static_assert(!std::is_same_v<decltype(expr), vec4f>, 
                      "Should be an expression");
        
        // Single evaluation
        vec4f result = expr;
        
        // Verify correctness
        vec4f expected;
        for (size_t i = 0; i < 4; ++i) {
            expected[i] = std::pow(a[i], 2.0f) * std::exp(b[i] * 0.1f) + 
                         std::sqrt(c[i]) * std::log(b[i] + 1.0f) - 
                         std::floor(a[i] * 10.0f) / std::ceil(b[i] * 0.5f);
        }
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Expression reuse") {
        vec3f v(1.0f, 2.0f, 3.0f);
        
        // Create expression once
        auto expr = sqrt(v) + log(v + 1.0f);
        
        // Evaluate multiple times
        vec3f result1 = expr;
        vec3f result2 = expr;
        
        // Results should be identical
        CHECK(approx_equal(result1, result2));
        
        // And correct
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sqrt(v[i]) + std::log(v[i] + 1.0f);
        }
        CHECK(approx_equal(result1, expected));
    }
}

TEST_CASE("Math functions with scalar broadcasting") {
    SUBCASE("Scalar-first operations") {
        vec3f v(2.0f, 4.0f, 8.0f);
        
        // 1.0 / v should work via scalar broadcasting
        auto expr = 1.0f / v;
        vec3f result = expr;
        CHECK(approx_equal(result, vec3f(0.5f, 0.25f, 0.125f)));
        
        // pow(2, v) - scalar base, vector exponent
        auto expr2 = 2.0f ^ v;  // Using ^ operator for pow
        vec3f result2 = expr2;
        CHECK(approx_equal(result2, vec3f(4.0f, 16.0f, 256.0f)));
    }
    
    SUBCASE("Mixed scalar operations in expressions") {
        vec3f v(1.0f, 2.0f, 3.0f);
        float s = 2.0f;
        
        // Complex expression mixing scalars and vectors
        auto expr = pow(s, v) + exp(v * 0.5f) - log(s + v);
        vec3f result = expr;
        
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::pow(s, v[i]) + std::exp(v[i] * 0.5f) - std::log(s + v[i]);
        }
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
}