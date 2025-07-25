#include <euler/math/trigonometry.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Trigonometric expression templates") {
    SUBCASE("Basic expression template operations") {
        vec3f v1(0.0f, constants<float>::pi / 4, constants<float>::pi / 2);
        vec3f v2(constants<float>::pi / 2, constants<float>::pi / 4, 0.0f);
        
        // Create expression without immediate evaluation
        auto expr = sin(v1 + v2);
        
        // Check that it's an expression type, not a vector
        static_assert(!std::is_same_v<decltype(expr), vec3f>, 
                      "sin(v1 + v2) should return an expression, not a vector");
        
        // Evaluate the expression
        vec3f result = expr;
        
        // Verify results
        for (size_t i = 0; i < 3; ++i) {
            CHECK(approx_equal(result[i], std::sin(v1[i] + v2[i])));
        }
    }
    
    SUBCASE("Complex expression templates") {
        vec3f angles(0.0f, constants<float>::pi / 6, constants<float>::pi / 3);
        
        // Store intermediate results to avoid dangling references
        vec3f sin_angles = sin(angles);
        vec3f cos_angles = cos(angles);
        
        // Complex expression with multiple operations
        auto expr = sin_angles * 2.0f + cos_angles * 3.0f;
        
        // Verify it's still an expression
        static_assert(!std::is_same_v<decltype(expr), vec3f>, 
                      "Complex expression should not evaluate immediately");
        
        // Evaluate
        vec3f result = expr;
        
        // Manual computation for verification
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(angles[i]) * 2.0f + std::cos(angles[i]) * 3.0f;
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Nested trigonometric expressions") {
        vec3f v(0.5f, 1.0f, 1.5f);
        
        // Nested operations
        auto expr = sin(cos(v));
        vec3f result = expr;
        
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(std::cos(v[i]));
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Mixed trigonometric and algebraic expressions") {
        vec3f a(1.0f, 2.0f, 3.0f);
        vec3f b(0.5f, 0.5f, 0.5f);
        
        // Store intermediate results to avoid dangling references
        vec3f sin_a = sin(a);
        vec3f cos_a = cos(a);
        vec3f sin_b = sin(b);
        vec3f cos_b = cos(b);
        vec3f a_plus_b = a + b;
        vec3f sin_a_plus_b = sin(a_plus_b);
        
        // sin(a) * cos(b) + cos(a) * sin(b) = sin(a + b)
        auto expr1 = sin_a * cos_b + cos_a * sin_b;
        
        vec3f result1 = expr1;
        vec3f result2 = sin_a_plus_b;
        
        CHECK(approx_equal(result1, result2, 1e-5f));
    }
    
    SUBCASE("Expression templates with angle types") {
        vector<degree<float>, 3> angles;
        angles[0] = 30.0_deg;
        angles[1] = 45.0_deg;
        angles[2] = 60.0_deg;
        vec3f radians(constants<float>::pi / 6, constants<float>::pi / 4, constants<float>::pi / 3);
        
        // Store intermediate results to avoid dangling references
        vec3f sin_angles = sin(angles);
        vec3f cos_radians = cos(radians);
        
        // Create expression from angle vector
        auto expr = sin_angles + cos_radians;
        vec3f result = expr;
        
        // Verify
        vec3f expected;
        expected[0] = std::sin(30.0f * constants<float>::deg_to_rad) + std::cos(constants<float>::pi / 6);
        expected[1] = std::sin(45.0f * constants<float>::deg_to_rad) + std::cos(constants<float>::pi / 4);
        expected[2] = std::sin(60.0f * constants<float>::deg_to_rad) + std::cos(constants<float>::pi / 3);
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Hyperbolic expression templates") {
        vec3f v(0.0f, 0.5f, 1.0f);
        
        // Store intermediate results to avoid dangling references
        vec3f sinh_v = sinh(v);
        vec3f cosh_v = cosh(v);
        
        // sinh^2(x) - cosh^2(x) = -1
        auto expr = sinh_v * sinh_v - cosh_v * cosh_v;
        vec3f result = expr;
        
        vec3f expected(-1.0f, -1.0f, -1.0f);
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Expression template with other vector operations") {
        vec3f v1(1.0f, 2.0f, 3.0f);
        vec3f v2(0.5f, 0.5f, 0.5f);
        
        // Store scalar temporaries to avoid dangling references
        float len_v2 = length(v2);
        float dot_v1_v2 = dot(v1, v2);
        vec3f sin_v1 = sin(v1);
        vec3f cos_v1 = cos(v1);
        
        // Combine trigonometric functions with other vector operations
        auto expr = sin_v1 * len_v2 + cos_v1 * dot_v1_v2;
        vec3f result = expr;
        
        // Manual calculation
        vec3f expected;
        for (size_t i = 0; i < 3; ++i) {
            expected[i] = std::sin(v1[i]) * len_v2 + std::cos(v1[i]) * dot_v1_v2;
        }
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Expression templates avoid intermediate allocations") {
        // This test verifies that expression templates work correctly
        // by checking that complex expressions produce correct results
        vec4f angles(0.0f, constants<float>::pi / 4, constants<float>::pi / 2, constants<float>::pi);
        
        // Store intermediate results to avoid dangling references
        vec4f sin_angles = sin(angles);
        vec4f cos_angles = cos(angles);
        vec4f half_angles = angles / 2.0f;
        vec4f tan_half_angles = tan(half_angles);
        
        // Complex expression that would require multiple temporaries without expression templates
        auto expr = sin_angles * cos_angles * 2.0f + tan_half_angles;
        
        // The expression should not have evaluated yet
        static_assert(!std::is_same_v<decltype(expr), vec4f>, 
                      "Expression should be lazy");
        
        // Now evaluate
        vec4f result = expr;
        
        // Verify each component
        for (size_t i = 0; i < 4; ++i) {
            float angle = angles[i];
            float expected = std::sin(angle) * std::cos(angle) * 2.0f + std::tan(angle / 2.0f);
            CHECK(approx_equal(result[i], expected));
        }
    }
}

TEST_CASE("Performance characteristics of expression templates") {
    SUBCASE("Single pass evaluation") {
        // Expression templates should evaluate in a single pass
        const size_t N = 100;
        std::vector<vec3f> angles(N);
        
        // Initialize with random angles
        for (size_t i = 0; i < N; ++i) {
            angles[i] = vec3f(static_cast<float>(i) * 0.01f, static_cast<float>(i) * 0.02f, static_cast<float>(i) * 0.03f);
        }
        
        // Process all vectors with a complex expression
        std::vector<vec3f> results(N);
        for (size_t i = 0; i < N; ++i) {
            // Store intermediate results to avoid dangling references
            vec3f sin_angle = sin(angles[i]);
            vec3f angles_3 = angles[i] * 3.0f;
            vec3f cos_angles_3 = cos(angles_3);
            vec3f angles_half = angles[i] / 2.0f;
            vec3f tan_angles_half = tan(angles_half);
            
            // This expression should compile to a single loop over the 3 components
            auto expr = sin_angle * 2.0f + cos_angles_3 - tan_angles_half;
            results[i] = expr;
        }
        
        // Verify a few results
        for (size_t i = 0; i < 10; ++i) {
            vec3f expected;
            for (size_t j = 0; j < 3; ++j) {
                float a = angles[i][j];
                expected[j] = std::sin(a) * 2.0f + std::cos(a * 3.0f) - std::tan(a / 2.0f);
            }
            CHECK(approx_equal(results[i], expected));
        }
    }
}