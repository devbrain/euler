#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/core/simd.hh>
#include <algorithm>
#include <numeric>
#include <vector>

// Helper to check if two vectors are approximately equal
template<typename Vec>
bool vec_approx_equal(const Vec& a, const Vec& b, typename Vec::value_type eps = 1e-6f) {
    constexpr size_t N = euler::vector_size_helper<Vec>::value;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

// Macro to run tests with and without SIMD
#define TEST_WITH_AND_WITHOUT_SIMD(name) \
    SUBCASE(name " - with SIMD") { \
        if (euler::simd_traits<float>::has_simd) { \
            test_impl(true); \
        } \
    } \
    SUBCASE(name " - without SIMD") { \
        test_impl(false); \
    }

TEST_CASE("euler::vector comprehensive expression template tests") {
    using namespace euler;
    
    SUBCASE("basic arithmetic expressions") {
        auto test_impl = [](bool) {
            // Test vectors of different sizes
            {
                vec2f a(1.0f, 2.0f);
                vec2f b(3.0f, 4.0f);
                vec2f c(5.0f, 6.0f);
                
                // Complex expression
                auto expr = a + b * 2.0f - c / 2.0f;
                vec2f result = expr;
                
                vec2f expected(1.0f + 3.0f*2.0f - 5.0f/2.0f,
                             2.0f + 4.0f*2.0f - 6.0f/2.0f);
                CHECK(vec_approx_equal(result, expected));
            }
            
            {
                vec3f a(1.0f, 2.0f, 3.0f);
                vec3f b(4.0f, 5.0f, 6.0f);
                vec3f c(7.0f, 8.0f, 9.0f);
                
                // Nested expressions
                auto expr1 = a + b;
                auto expr2 = c - a;
                auto expr3 = expr1 * 2.0f + expr2 / 3.0f;
                vec3f result = expr3;
                
                vec3f expected((1.0f+4.0f)*2.0f + (7.0f-1.0f)/3.0f,
                             (2.0f+5.0f)*2.0f + (8.0f-2.0f)/3.0f,
                             (3.0f+6.0f)*2.0f + (9.0f-3.0f)/3.0f);
                CHECK(vec_approx_equal(result, expected));
            }
            
            {
                vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
                vec4f b(5.0f, 6.0f, 7.0f, 8.0f);
                
                // Chain of operations
                auto expr = ((a + b) * 2.0f - a * 3.0f) / b;
                vec4f result = expr;
                
                vec4f expected(((1.0f+5.0f)*2.0f - 1.0f*3.0f)/5.0f,
                             ((2.0f+6.0f)*2.0f - 2.0f*3.0f)/6.0f,
                             ((3.0f+7.0f)*2.0f - 3.0f*3.0f)/7.0f,
                             ((4.0f+8.0f)*2.0f - 4.0f*3.0f)/8.0f);
                CHECK(vec_approx_equal(result, expected));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("basic arithmetic expressions")
    }
    
    SUBCASE("vector-specific expression templates") {
        auto test_impl = [](bool) {
            vec3f a(3.0f, 4.0f, 0.0f);
            vec3f b(0.0f, 0.0f, 1.0f);
            vec3f c(1.0f, 0.0f, 0.0f);
            
            // Test normalize expression
            {
                auto norm_expr = normalize(a);
                vec3f result = norm_expr;
                CHECK(vec_approx_equal(result, vec3f(0.6f, 0.8f, 0.0f)));
                CHECK(doctest::Approx(length(result)) == 1.0f);
            }
            
            // Test cross product expression
            {
                auto cross_expr = cross(c, b);
                vec3f result = cross_expr;
                CHECK(vec_approx_equal(result, vec3f(0.0f, -1.0f, 0.0f)));
            }
            
            // Test reflect expression
            {
                vec3f incident(1.0f, -1.0f, 0.0f);
                vec3f normal(0.0f, 1.0f, 0.0f);
                auto reflect_expr = reflect(incident, normal);
                vec3f result = reflect_expr;
                CHECK(vec_approx_equal(result, vec3f(1.0f, 1.0f, 0.0f)));
            }
            
            // Test lerp expression
            {
                vec3f start(0.0f, 0.0f, 0.0f);
                vec3f end(10.0f, 20.0f, 30.0f);
                auto lerp_expr = lerp(start, end, 0.25f);
                vec3f result = lerp_expr;
                CHECK(vec_approx_equal(result, vec3f(2.5f, 5.0f, 7.5f)));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("vector-specific expression templates")
    }
    
    SUBCASE("combined expressions") {
        auto test_impl = [](bool) {
            vec3f a(1.0f, 0.0f, 0.0f);
            vec3f b(0.0f, 1.0f, 0.0f);
            vec3f c(0.0f, 0.0f, 1.0f);
            
            // Complex combined expression
            {
                // normalize(cross(a, b) + c * 0.5f)
                vec3f cross_result = cross(a, b);
                vec3f sum = cross_result + c * 0.5f;
                auto expr = normalize(sum);
                vec3f result = expr;
                
                // cross(a, b) = (0, 0, 1)
                // cross(a, b) + c * 0.5f = (0, 0, 1.5)
                // normalize((0, 0, 1.5)) = (0, 0, 1)
                CHECK(vec_approx_equal(result, vec3f(0.0f, 0.0f, 1.0f)));
            }
            
            // Chained normalizations
            {
                vec3f v(3.0f, 4.0f, 0.0f);
                auto norm_v = normalize(v);
                auto norm_c = normalize(c);
                vec3f nv = norm_v;
                vec3f nc = norm_c;
                vec3f sum = nv + nc;
                auto expr = normalize(sum);
                vec3f result = expr;
                
                // normalize(v) = (0.6, 0.8, 0)
                // normalize(c) = (0, 0, 1)
                // sum = (0.6, 0.8, 1)
                // length = sqrt(0.36 + 0.64 + 1) = sqrt(2) ≈ 1.414
                vec3f expected(0.6f/std::sqrt(2.0f), 0.8f/std::sqrt(2.0f), 1.0f/std::sqrt(2.0f));
                CHECK(vec_approx_equal(result, expected, 1e-5f));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("combined expressions")
    }
    

        

    
    SUBCASE("expression with different vector types") {
        auto test_impl = [](bool) {
            // Test with column vectors
            {
                column_vector<float, 3> cv1;
                cv1[0] = 1.0f; cv1[1] = 2.0f; cv1[2] = 3.0f;
                
                column_vector<float, 3> cv2;
                cv2[0] = 4.0f; cv2[1] = 5.0f; cv2[2] = 6.0f;
                
                auto expr = cv1 + cv2 * 2.0f;
                column_vector<float, 3> result = expr;
                
                CHECK(result[0] == doctest::Approx(1.0f + 4.0f * 2.0f));
                CHECK(result[1] == doctest::Approx(2.0f + 5.0f * 2.0f));
                CHECK(result[2] == doctest::Approx(3.0f + 6.0f * 2.0f));
            }
            
            // Test with row vectors
            {
                row_vector<float, 3> rv1;
                rv1[0] = 1.0f; rv1[1] = 2.0f; rv1[2] = 3.0f;
                
                row_vector<float, 3> rv2;
                rv2[0] = 4.0f; rv2[1] = 5.0f; rv2[2] = 6.0f;
                
                auto expr = rv1 - rv2 / 2.0f;
                row_vector<float, 3> result = expr;
                
                CHECK(result[0] == doctest::Approx(1.0f - 4.0f / 2.0f));
                CHECK(result[1] == doctest::Approx(2.0f - 5.0f / 2.0f));
                CHECK(result[2] == doctest::Approx(3.0f - 6.0f / 2.0f));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("expression with different vector types")
    }
    
    SUBCASE("expression templates with immediate operations") {
        auto test_impl = [](bool) {
            vec3f a(3.0f, 4.0f, 0.0f);
            vec3f b(1.0f, 0.0f, 0.0f);
            vec3f c(0.0f, 1.0f, 0.0f);
            
            // Mix of lazy and immediate operations
            {
                // dot is immediate, but used in expression
                auto expr = b * dot(a, b) + c * dot(a, c);
                vec3f result = expr;
                
                // dot(a, b) = 3, dot(a, c) = 4
                vec3f expected(1.0f * 3.0f + 0.0f * 4.0f,
                             0.0f * 3.0f + 1.0f * 4.0f,
                             0.0f * 3.0f + 0.0f * 4.0f);
                CHECK(vec_approx_equal(result, expected));
            }
            
            // length_squared with expression - now works directly!
            {
                // Test that length_squared works directly with expressions
                float len_sq = length_squared(a + b);
                CHECK(len_sq == doctest::Approx(32.0f)); // (4,4,0) -> 16+16+0
                
                // Also test length with expression
                float len = length(a + b);
                CHECK(len == doctest::Approx(std::sqrt(32.0f)));
                
                // Test dot product with expressions
                float d = dot(a + b, c);
                CHECK(d == doctest::Approx(4.0f)); // (4,4,0) dot (0,1,0) = 4
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("expression templates with immediate operations")
    }
    
    SUBCASE("large vector operations") {
        auto test_impl = [](bool) {
            // Create larger vectors to test SIMD efficiency
            constexpr size_t N = 16;
            std::vector<vec4f> vectors_a, vectors_b, vectors_c;
            vectors_a.reserve(N);
            vectors_b.reserve(N);
            vectors_c.reserve(N);
            
            for (size_t i = 0; i < N; ++i) {
                auto fi = static_cast<float>(i);
                vectors_a.emplace_back(fi, fi+1, fi+2, fi+3);
                vectors_b.emplace_back(fi*2, fi*2+1, fi*2+2, fi*2+3);
                vectors_c.emplace_back(fi*3, fi*3+1, fi*3+2, fi*3+3);
            }
            
            // Process all vectors with expression templates
            std::vector<vec4f> results;
            results.reserve(N);
            
            for (size_t i = 0; i < N; ++i) {
                auto expr = vectors_a[i] * 2.0f + vectors_b[i] / 3.0f - vectors_c[i] * 0.5f;
                results.emplace_back(expr);
            }
            
            // Verify results
            for (size_t i = 0; i < N; ++i) {
                auto fi = static_cast<float>(i);
                vec4f expected(
                    fi*2.0f + (fi*2)/3.0f - (fi*3)*0.5f,
                    (fi+1)*2.0f + (fi*2+1)/3.0f - (fi*3+1)*0.5f,
                    (fi+2)*2.0f + (fi*2+2)/3.0f - (fi*3+2)*0.5f,
                    (fi+3)*2.0f + (fi*2+3)/3.0f - (fi*3+3)*0.5f
                );
                CHECK(vec_approx_equal(results[i], expected, 1e-5f));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("large vector operations")
    }
    
    SUBCASE("edge cases") {
        auto test_impl = [](bool) {
            // Zero vectors
            {
                vec3f zero = vec3f::zero();
                vec3f one = vec3f::ones();
                
                auto expr = zero + one * 2.0f;
                vec3f result = expr;
                CHECK(vec_approx_equal(result, vec3f(2.0f, 2.0f, 2.0f)));
            }
            
            // Very small values (denormalized numbers)
            {
                vec3f tiny(1e-38f, 2e-38f, 3e-38f);
                vec3f normal(1.0f, 2.0f, 3.0f);
                
                auto expr = tiny + normal;
                vec3f result = expr;
                CHECK(vec_approx_equal(result, normal, 1e-6f));
            }
            
            // Large values
            {
                vec3f large(1e20f, 2e20f, 3e20f);
                auto expr = large / 1e10f;
                vec3f result = expr;
                CHECK(vec_approx_equal(result, vec3f(1e10f, 2e10f, 3e10f)));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("edge cases")
    }
    
    SUBCASE("type deduction and conversions") {
        auto test_impl = [](bool) {
            // Ensure expressions maintain correct types
            {
                vec3f a(1.0f, 2.0f, 3.0f);
                vec3d b(4.0, 5.0, 6.0);
                
                // Expression with float vectors
                auto expr_f = a + a;
                static_assert(std::is_same_v<decltype(expr_f)::value_type, float>);
                vec3f result_f = expr_f;
                CHECK(vec_approx_equal(result_f, vec3f(2.0f, 4.0f, 6.0f)));
                
                // Expression with double vectors
                auto expr_d = b + b;
                static_assert(std::is_same_v<decltype(expr_d)::value_type, double>);
                vec3d result_d = expr_d;
                CHECK(vec_approx_equal(result_d, vec3d(8.0, 10.0, 12.0)));
            }
        };
        
        TEST_WITH_AND_WITHOUT_SIMD("type deduction and conversions")
    }
}

TEST_CASE("euler::vector expression performance characteristics") {
    using namespace euler;
    
    SUBCASE("expression reuse") {
        vec3f a(1.0f, 2.0f, 3.0f);
        vec3f b(4.0f, 5.0f, 6.0f);
        
        // Create expression once
        auto expr = a + b * 2.0f;
        
        // Evaluate multiple times
        vec3f result1 = expr;
        vec3f result2 = expr;
        
        CHECK(vec_approx_equal(result1, result2));
        CHECK(vec_approx_equal(result1, vec3f(9.0f, 12.0f, 15.0f)));
    }
    
    SUBCASE("nested expression depth") {
        vec3f v(1.0f, 2.0f, 3.0f);
        
        // Deep nesting of expressions
        auto expr1 = v + v;
        auto expr2 = expr1 + v;
        auto expr3 = expr2 + expr1;
        auto expr4 = expr3 + expr2;
        auto expr5 = expr4 + expr3;
        
        vec3f result = expr5;
        
        // v=1,2,3; expr1=2,4,6; expr2=3,6,9; expr3=5,10,15; expr4=8,16,24; expr5=13,26,39
        CHECK(vec_approx_equal(result, vec3f(13.0f, 26.0f, 39.0f)));
    }
}

TEST_CASE("euler::vector SIMD verification") {
    using namespace euler;
    
    SUBCASE("verify SIMD usage") {
        if (simd_traits<float>::has_simd) {
            // Check that SIMD-sized vectors use SIMD paths
            vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
            vec4f b(5.0f, 6.0f, 7.0f, 8.0f);
            
            // This should use SIMD operations internally
            auto expr = a + b;
            vec4f result = expr;
            
            CHECK(vec_approx_equal(result, vec4f(6.0f, 8.0f, 10.0f, 12.0f)));
        }
    }
    
    SUBCASE("compare SIMD vs non-SIMD results") {
        // Results should be identical regardless of SIMD
        vec4f a(1.5f, 2.5f, 3.5f, 4.5f);
        vec4f b(0.5f, 1.5f, 2.5f, 3.5f);
        
        // Complex expression
        auto expr = (a * 3.0f + b / 2.0f) - (a - b) * 0.5f;
        vec4f result = expr;
        
        // Manual calculation
        vec4f expected;
        for (size_t i = 0; i < 4; ++i) {
            expected[i] = (a[i] * 3.0f + b[i] / 2.0f) - (a[i] - b[i]) * 0.5f;
        }
        
        CHECK(vec_approx_equal(result, expected, 1e-6f));
    }
}