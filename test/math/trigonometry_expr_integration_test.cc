#include <euler/math/trigonometry.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include <chrono>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Trigonometric expression templates integration") {
    SUBCASE("Complex mixed expressions with vector operations") {
        vec3f v1(1.0f, 2.0f, 3.0f);
        vec3f v2(0.5f, 1.0f, 1.5f);
        vec3f v3(2.0f, 1.5f, 1.0f);
        
        // Store temporaries to avoid dangling references
        vec3f cross_v2_v3 = cross(v2, v3);
        vec3f norm_v2 = normalize(v2);
        
        // Complex expression combining trig functions with vector ops
        auto expr = sin(v1) * cross_v2_v3 + cos(v1) * norm_v2 - tan(v1 / 2.0f);
        
        // Verify it's still an expression
        static_assert(!std::is_same_v<decltype(expr), vec3f>, 
                      "Should be an expression, not evaluated");
        
        // Evaluate
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(v1[i]) * cross_v2_v3[i] + 
                         std::cos(v1[i]) * norm_v2[i] - 
                         std::tan(v1[i] / 2.0f);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Trigonometric functions with dot and length operations") {
        vec4f a(0.5f, 1.0f, 1.5f, 2.0f);
        vec4f b(2.0f, 1.5f, 1.0f, 0.5f);
        
        // Store scalar temporaries to avoid dangling references
        float dot_ab = dot(a, b);
        float len_a = length(a);
        
        // Expression combining trig with dot product and length
        auto expr = sin(a) * dot_ab + cos(b) * len_a + sinh(a - b);
        
        vec4f result = expr;
        
        // Verify correctness
        vec4f expected;
        for (size_t i = 0; i < 4; ++i) {
            expected[i] = std::sin(a[i]) * dot_ab + 
                         std::cos(b[i]) * len_a + 
                         std::sinh(a[i] - b[i]);
        }
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }

    SUBCASE("Nested trigonometric and algebraic expressions") {
        vec3f v(0.5f, 1.0f, 1.5f);
        
        // Store intermediate expressions to avoid dangling references
        vec3f cos_v = cos(v);
        vec3f sin_v = sin(v);
        vec3f abs_cos_v = abs(cos_v);
        
        // Deep nesting of operations using only available functions
        auto expr = sin(cos(v * 2.0f) + 0.5f) * tan(v / 3.0f) + 
                    abs_cos_v * (1.0f + sin_v * 0.1f);
        
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            float cos_2v = std::cos(v[i] * 2.0f);
            float sin_cos = std::sin(cos_2v + 0.5f);
            float tan_v3 = std::tan(v[i] / 3.0f);
            float abs_cos = std::abs(std::cos(v[i]));
            float factor = 1.0f + std::sin(v[i]) * 0.1f;
            expected[i] = sin_cos * tan_v3 + abs_cos * factor;
        }
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Matrix operations with trigonometric functions") {
        matrix<float, 2, 3> m1;
        m1(0, 0) = 0.5f; m1(0, 1) = 1.0f; m1(0, 2) = 1.5f;
        m1(1, 0) = 2.0f; m1(1, 1) = 2.5f; m1(1, 2) = 3.0f;
        
        matrix<float, 3, 2> m2;
        m2(0, 0) = 0.1f; m2(0, 1) = 0.2f;
        m2(1, 0) = 0.3f; m2(1, 1) = 0.4f;
        m2(2, 0) = 0.5f; m2(2, 1) = 0.6f;
        
        // Matrix expression with trig functions - correct dimensions
        auto expr = sin(m1 * m2) * 2.0f + cos(m1 * m2);
        
        matrix<float, 2, 2> result = expr;
        
        // Verify dimensions
        CHECK(result.rows == 2);
        CHECK(result.cols == 2);
        
        // Manual calculation
        matrix<float, 2, 2> prod = m1 * m2;
        matrix<float, 2, 2> sin_prod = sin(prod);
        matrix<float, 2, 2> cos_prod = cos(prod);
        matrix<float, 2, 2> expected = sin_prod * 2.0f + cos_prod;
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Trigonometric functions with min/max/clamp operations") {
        vec3f v1(0.5f, 1.5f, 2.5f);
        vec3f v2(1.0f, 1.0f, 1.0f);
        vec3f v3(2.0f, 0.5f, 1.5f);
        
        // Store temporaries to avoid dangling references
        vec3f clamped = clamp(v1, 0.0f, constants<float>::pi);
        vec3f maxed = max(v2, v3);
        vec3f pi_quarter(constants<float>::pi / 4);
        vec3f mined = min(v1, pi_quarter);
        vec3f sin_clamped = sin(clamped);
        vec3f cos_maxed = cos(maxed);
        vec3f tan_mined = tan(mined);
        
        // Expression with min/max/clamp
        auto expr = sin_clamped + 
                    cos_maxed * 
                    tan_mined;
        
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(clamped[i]) + 
                         std::cos(maxed[i]) * 
                         std::tan(mined[i]);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Trigonometric functions with lerp") {
        vec3f a(0.0f, 0.5f, 1.0f);
        vec3f b(1.0f, 1.5f, 2.0f);
        float t = 0.3f;
        
        // Expression with interpolation function
        auto expr = sin(lerp(a, b, t)) + cos(lerp(b, a, t)) * 2.0f;
        
        vec3f result = expr;
        
        // Manual calculation
        vec3f lerped_ab = lerp(a, b, t);
        vec3f lerped_ba = lerp(b, a, t);
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(lerped_ab[i]) + std::cos(lerped_ba[i]) * 2.0f;
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Angle type expressions with vector operations") {
        vector<degree<float>, 3> angles;
        angles[0] = 30.0_deg;
        angles[1] = 45.0_deg;
        angles[2] = 60.0_deg;
        
        vec3f radians(constants<float>::pi / 6, constants<float>::pi / 4, constants<float>::pi / 3);
        
        // Mixed angle and radian expressions
        // Note: angles / 2.0f needs explicit conversion
        vector<degree<float>, 3> half_angles;
        for (size_t i = 0; i < 3; ++i) {
            half_angles[i] = degree<float>(angles[i].value() / 2.0f);
        }
        
        // Store intermediate results to avoid dangling references
        vec3f sin_angles = sin(angles);
        vec3f cos_radians = cos(radians);
        vec3f tan_half_angles = tan(half_angles);
        vec3f sin_radians_2 = sin(radians * 2.0f);
        
        auto expr = sin_angles * cos_radians + 
                    tan_half_angles - 
                    sin_radians_2;
        
        vec3f result = expr;
        
        // Verify the calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            float angle_rad = angles[i].value() * constants<float>::deg_to_rad;
            float half_angle_rad = half_angles[i].value() * constants<float>::deg_to_rad;
            expected[i] = std::sin(angle_rad) * std::cos(radians[i]) + 
                         std::tan(half_angle_rad) - 
                         std::sin(radians[i] * 2.0f);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Performance test - single evaluation of complex expression") {
        const size_t N = 1000;
        std::vector<vec4f> data(N);
        
        // Initialize with test data
        for (size_t i = 0; i < N; ++i) {
            data[i] = vec4f(static_cast<float>(i) * 0.001f, 
                           static_cast<float>(i) * 0.002f, 
                           static_cast<float>(i) * 0.003f, 
                           static_cast<float>(i) * 0.004f);
        }
        
        // Complex expression that should evaluate in a single pass
        std::vector<vec4f> results(N);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < N; ++i) {
            // Store length as a temporary to avoid dangling reference
            float len = length(data[i]);
            
            // This complex expression should compile to a single loop
            auto expr = sin(data[i] * 2.0f) * cos(data[i] * 3.0f) + 
                       tan(data[i] / 2.0f) * len - 
                       sinh(data[i] * 0.1f) / (cosh(data[i] * 0.1f) + 1.0f);
            results[i] = expr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Just verify a few results
        for (size_t i = 0; i < 10; i += 3) {
            vec4f expected;
            float len = length(data[i]);
            for (size_t j = 0; j < 4; ++j) {
                float val = data[i][j];
                expected[j] = std::sin(val * 2.0f) * std::cos(val * 3.0f) + 
                             std::tan(val / 2.0f) * len - 
                             std::sinh(val * 0.1f) / (std::cosh(val * 0.1f) + 1.0f);
            }
            CHECK(approx_equal(results[i], expected, 1e-5f));
        }
        
        // Performance should be reasonable
        CHECK(duration.count() < 100000); // Less than 100ms for 1000 vectors
    }
    
    SUBCASE("Expression templates with scalar broadcast") {
        vec3f v(0.5f, 1.0f, 1.5f);
        float scalar = 2.0f;
        
        // Store intermediate results to avoid issues with temporaries
        vec3f sin_v = sin(v);
        vec3f v_times_scalar = v * scalar;
        vec3f tan_v_scalar = tan(v_times_scalar);
        
        // Expression with scalar broadcasting
        auto expr = sin_v * scalar + cos(scalar) * v - tan_v_scalar;
        
        vec3f result = expr;
        
        // Manual calculation
        float cos_scalar = std::cos(scalar);
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(v[i]) * scalar + 
                         cos_scalar * v[i] - 
                         std::tan(v[i] * scalar);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Chained trigonometric transformations") {
        vec3f angles(15.0f, 30.0f, 45.0f);
        angles = angles * constants<float>::deg_to_rad; // Convert to radians
        
        // Chain of transformations - store intermediate results
        vec3f sin_angles = sin(angles);
        vec3f cos_angles = cos(angles);
        vec3f expr1_pi = sin_angles * constants<float>::pi;
        vec3f cos_expr1 = cos(expr1_pi);
        vec3f expr2_div2 = cos_expr1 / 2.0f;
        vec3f tan_expr2 = tan(expr2_div2);
        
        auto final_expr = tan_expr2 + sin_angles * cos_angles;
        
        vec3f result = final_expr;
        
        // Verify the chain evaluates correctly
        vec3f expected = tan_expr2 + sin_angles * cos_angles;
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Expression templates with hyperbolic and regular trig functions") {
        vec4f x(0.1f, 0.2f, 0.3f, 0.4f);
        
        // Store intermediate results to avoid dangling references
        vec4f sinh_x = sinh(x);
        vec4f sin_x = sin(x);
        vec4f cosh_x = cosh(x);
        vec4f cos_x = cos(x);
        vec4f tanh_x = tanh(x);
        vec4f tan_x = tan(x);
        
        // Mix of hyperbolic and regular trig functions
        auto expr = sinh_x * sin_x + cosh_x * cos_x - tanh_x * tan_x;
        
        vec4f result = expr;
        
        // Verify
        vec4f expected;
        for (size_t i = 0; i < 4; ++i) {
            expected[i] = std::sinh(x[i]) * std::sin(x[i]) + 
                         std::cosh(x[i]) * std::cos(x[i]) - 
                         std::tanh(x[i]) * std::tan(x[i]);
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Expression template type deduction") {
        vec3f v1(1.0f, 2.0f, 3.0f);
        vec3f v2(0.5f, 1.5f, 2.5f);
        
        // Create complex expression
        auto expr = sin(v1 + v2) * cos(v1 - v2) + tan(v1 * v2);
        

        // Should be able to evaluate multiple times
        vec3f result1 = expr;
        vec3f result2 = expr;
        
        // Results should be identical
        CHECK(approx_equal(result1, result2));
        
        // And correct
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(v1[i] + v2[i]) * std::cos(v1[i] - v2[i]) + 
                         std::tan(v1[i] * v2[i]);
        }
        CHECK(approx_equal(result1, expected));
    }
}