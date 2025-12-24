/**
 * @file test_vector_ops.cc
 * @brief Unit tests for direct SIMD vector operations
 */

#include <doctest/doctest.h>
#include <euler/direct/vector_ops.hh>
#include <euler/vector/vector.hh>
#include <euler/core/approx_equal.hh>
#include <random>
#include <vector>
#include <chrono>

using namespace euler;
using namespace euler::direct;

// Test configuration
constexpr float FLOAT_TOL = 1e-6f;
constexpr double DOUBLE_TOL = 1e-12;

// Helper to generate random vectors
template<typename T>
class RandomVectorGenerator {
public:
    RandomVectorGenerator(T min_val = -10, T max_val = 10) 
        : gen(std::random_device{}()), dist(min_val, max_val) {}
    
    template<size_t N>
    vector<T, N> generate() {
        vector<T, N> v;
        for (size_t i = 0; i < N; ++i) {
            v[i] = dist(gen);
        }
        return v;
    }
    
private:
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
};

// =============================================================================
// Binary Operations Tests
// =============================================================================

TEST_CASE("Direct vector addition") {
    RandomVectorGenerator<float> rng_f;
    RandomVectorGenerator<double> rng_d;
    
    SUBCASE("Basic addition - vec3<float>") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        vec3<float> result;
        
        add(a, b, result);
        
        CHECK(result[0] == doctest::Approx(5.0f));
        CHECK(result[1] == doctest::Approx(7.0f));
        CHECK(result[2] == doctest::Approx(9.0f));
    }
    
    SUBCASE("Addition with aliasing - result = a + a") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> expected(2.0f, 4.0f, 6.0f);
        
        add(a, a, a);  // a = a + a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Addition with aliasing - result = a + b where result is a") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        vec3<float> expected(5.0f, 7.0f, 9.0f);
        
        add(a, b, a);  // a = a + b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Addition with aliasing - result = a + b where result is b") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        vec3<float> expected(5.0f, 7.0f, 9.0f);
        
        add(a, b, b);  // b = a + b
        
        CHECK(approx_equal(b, expected, FLOAT_TOL));
    }
    
    SUBCASE("Random addition tests - various sizes") {
        // Test different vector sizes
        for (int test = 0; test < 10; ++test) {
            // vec2
            {
                auto a = rng_f.generate<2>();
                auto b = rng_f.generate<2>();
                vec2<float> result;
                vec2<float> expected;
                
                for (size_t i = 0; i < 2; ++i) {
                    expected[i] = a[i] + b[i];
                }
                
                add(a, b, result);
                CHECK(approx_equal(result, expected, FLOAT_TOL));
            }
            
            // vec4
            {
                auto a = rng_f.generate<4>();
                auto b = rng_f.generate<4>();
                vec4<float> result;
                vec4<float> expected;
                
                for (size_t i = 0; i < 4; ++i) {
                    expected[i] = a[i] + b[i];
                }
                
                add(a, b, result);
                CHECK(approx_equal(result, expected, FLOAT_TOL));
            }
            
            // Large vector
            {
                auto a = rng_f.generate<16>();
                auto b = rng_f.generate<16>();
                vector<float, 16> result;
                vector<float, 16> expected;
                
                for (size_t i = 0; i < 16; ++i) {
                    expected[i] = a[i] + b[i];
                }
                
                add(a, b, result);
                CHECK(approx_equal(result, expected, FLOAT_TOL));
            }
        }
    }
    
    SUBCASE("Addition - double precision") {
        vec3<double> a(1.0, 2.0, 3.0);
        vec3<double> b(4.0, 5.0, 6.0);
        vec3<double> result;
        
        add(a, b, result);
        
        CHECK(result[0] == doctest::Approx(5.0));
        CHECK(result[1] == doctest::Approx(7.0));
        CHECK(result[2] == doctest::Approx(9.0));
    }
}

TEST_CASE("Direct vector subtraction") {
    SUBCASE("Basic subtraction") {
        vec3<float> a(5.0f, 7.0f, 9.0f);
        vec3<float> b(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        sub(a, b, result);
        
        CHECK(result[0] == doctest::Approx(4.0f));
        CHECK(result[1] == doctest::Approx(5.0f));
        CHECK(result[2] == doctest::Approx(6.0f));
    }
    
    SUBCASE("Subtraction with aliasing - result = a - a") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> expected(0.0f, 0.0f, 0.0f);
        
        sub(a, a, a);  // a = a - a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Subtraction with aliasing - result = a - b where result is a") {
        vec3<float> a(5.0f, 7.0f, 9.0f);
        vec3<float> b(1.0f, 2.0f, 3.0f);
        vec3<float> expected(4.0f, 5.0f, 6.0f);
        
        sub(a, b, a);  // a = a - b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

TEST_CASE("Direct vector multiplication") {
    SUBCASE("Basic element-wise multiplication") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> result;
        
        mul(a, b, result);
        
        CHECK(result[0] == doctest::Approx(10.0f));
        CHECK(result[1] == doctest::Approx(18.0f));
        CHECK(result[2] == doctest::Approx(28.0f));
    }
    
    SUBCASE("Multiplication with aliasing - result = a * a") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> expected(4.0f, 9.0f, 16.0f);
        
        mul(a, a, a);  // a = a * a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

TEST_CASE("Direct vector division") {
    SUBCASE("Basic element-wise division") {
        vec3<float> a(10.0f, 18.0f, 28.0f);
        vec3<float> b(2.0f, 3.0f, 4.0f);
        vec3<float> result;
        
        div(a, b, result);
        
        CHECK(result[0] == doctest::Approx(5.0f));
        CHECK(result[1] == doctest::Approx(6.0f));
        CHECK(result[2] == doctest::Approx(7.0f));
    }
    
    SUBCASE("Division with aliasing - result = a / b where result is a") {
        vec3<float> a(10.0f, 18.0f, 28.0f);
        vec3<float> b(2.0f, 3.0f, 4.0f);
        vec3<float> expected(5.0f, 6.0f, 7.0f);
        
        div(a, b, a);  // a = a / b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

// =============================================================================
// Scalar Broadcasting Tests
// =============================================================================

TEST_CASE("Scalar broadcasting operations") {
    SUBCASE("Scalar + Vector") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        add(5.0f, v, result);
        
        CHECK(result[0] == doctest::Approx(6.0f));
        CHECK(result[1] == doctest::Approx(7.0f));
        CHECK(result[2] == doctest::Approx(8.0f));
    }
    
    SUBCASE("Vector + Scalar") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        add(v, 5.0f, result);
        
        CHECK(result[0] == doctest::Approx(6.0f));
        CHECK(result[1] == doctest::Approx(7.0f));
        CHECK(result[2] == doctest::Approx(8.0f));
    }
    
    SUBCASE("Scalar + Vector with aliasing") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> expected(6.0f, 7.0f, 8.0f);
        
        add(5.0f, v, v);  // v = 5 + v
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("Scalar - Vector") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        sub(10.0f, v, result);
        
        CHECK(result[0] == doctest::Approx(9.0f));
        CHECK(result[1] == doctest::Approx(8.0f));
        CHECK(result[2] == doctest::Approx(7.0f));
    }
    
    SUBCASE("Vector - Scalar") {
        vec3<float> v(10.0f, 20.0f, 30.0f);
        vec3<float> result;
        
        sub(v, 5.0f, result);
        
        CHECK(result[0] == doctest::Approx(5.0f));
        CHECK(result[1] == doctest::Approx(15.0f));
        CHECK(result[2] == doctest::Approx(25.0f));
    }
    
    SUBCASE("Scalar * Vector") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        mul(2.0f, v, result);
        
        CHECK(result[0] == doctest::Approx(2.0f));
        CHECK(result[1] == doctest::Approx(4.0f));
        CHECK(result[2] == doctest::Approx(6.0f));
    }
    
    SUBCASE("Scale with aliasing") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> expected(2.0f, 4.0f, 6.0f);
        
        scale(v, 2.0f, v);  // v = 2 * v
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("Scalar / Vector") {
        vec3<float> v(2.0f, 4.0f, 8.0f);
        vec3<float> result;
        
        div(16.0f, v, result);
        
        CHECK(result[0] == doctest::Approx(8.0f));  // 16/2
        CHECK(result[1] == doctest::Approx(4.0f));  // 16/4
        CHECK(result[2] == doctest::Approx(2.0f));  // 16/8
    }
    
    SUBCASE("Vector / Scalar") {
        vec3<float> v(10.0f, 20.0f, 30.0f);
        vec3<float> result;
        
        div(v, 5.0f, result);
        
        CHECK(result[0] == doctest::Approx(2.0f));
        CHECK(result[1] == doctest::Approx(4.0f));
        CHECK(result[2] == doctest::Approx(6.0f));
    }
}

// =============================================================================
// Geometric Operations Tests
// =============================================================================

TEST_CASE("Dot product") {
    SUBCASE("Basic dot product") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        
        float result = dot(a, b);
        
        CHECK(result == doctest::Approx(32.0f));  // 1*4 + 2*5 + 3*6
    }
    
    SUBCASE("Dot product with self") {
        vec3<float> a(3.0f, 4.0f, 0.0f);
        
        float result = dot(a, a);
        
        CHECK(result == doctest::Approx(25.0f));  // 3² + 4² + 0²
    }
    
    SUBCASE("Dot product - large vector") {
        vector<float, 16> a, b;
        for (size_t i = 0; i < 16; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
        }
        
        float result = dot(a, b);
        float expected = 0.0f;
        for (size_t i = 0; i < 16; ++i) {
            expected += a[i] * b[i];
        }
        
        CHECK(result == doctest::Approx(expected));
    }
}

TEST_CASE("Cross product") {
    SUBCASE("Basic cross product") {
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(0.0f, 1.0f, 0.0f);
        vec3<float> result;
        
        cross(a, b, result);
        
        CHECK(result[0] == doctest::Approx(0.0f));
        CHECK(result[1] == doctest::Approx(0.0f));
        CHECK(result[2] == doctest::Approx(1.0f));
    }
    
    SUBCASE("Cross product with aliasing - result = a × b where result is a") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> expected(-3.0f, 6.0f, -3.0f);  // (3*7-4*6, 4*5-2*7, 2*6-3*5)
        
        cross(a, b, a);  // a = a × b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Cross product with aliasing - result = a × b where result is b") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> expected(-3.0f, 6.0f, -3.0f);
        
        cross(a, b, b);  // b = a × b
        
        CHECK(approx_equal(b, expected, FLOAT_TOL));
    }
    
    SUBCASE("Cross product with self") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> expected(0.0f, 0.0f, 0.0f);
        
        cross(a, a, a);  // a = a × a (should be zero)
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

TEST_CASE("Norm operations") {
    SUBCASE("Norm squared") {
        vec3<float> v(3.0f, 4.0f, 0.0f);
        
        float result = norm_squared(v);
        
        CHECK(result == doctest::Approx(25.0f));
    }
    
    SUBCASE("Norm") {
        vec3<float> v(3.0f, 4.0f, 0.0f);
        
        float result = norm(v);
        
        CHECK(result == doctest::Approx(5.0f));
    }
    
    SUBCASE("Normalize") {
        vec3<float> v(3.0f, 4.0f, 0.0f);
        vec3<float> result;
        
        normalize(v, result);
        
        CHECK(result[0] == doctest::Approx(0.6f));
        CHECK(result[1] == doctest::Approx(0.8f));
        CHECK(result[2] == doctest::Approx(0.0f));
        
        // Check that result has unit length
        float len = norm(result);
        CHECK(len == doctest::Approx(1.0f));
    }
    
    SUBCASE("Normalize with aliasing") {
        vec3<float> v(3.0f, 4.0f, 0.0f);
        
        normalize(v, v);  // v = normalize(v)
        
        float len = norm(v);
        CHECK(len == doctest::Approx(1.0f));
    }
}

// =============================================================================
// Unary Operations Tests
// =============================================================================

TEST_CASE("Unary operations") {
    SUBCASE("Negate") {
        vec3<float> v(1.0f, -2.0f, 3.0f);
        vec3<float> result;
        
        negate(v, result);
        
        CHECK(result[0] == doctest::Approx(-1.0f));
        CHECK(result[1] == doctest::Approx(2.0f));
        CHECK(result[2] == doctest::Approx(-3.0f));
    }
    
    SUBCASE("Negate with aliasing") {
        vec3<float> v(1.0f, -2.0f, 3.0f);
        vec3<float> expected(-1.0f, 2.0f, -3.0f);
        
        negate(v, v);  // v = -v
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("Absolute value") {
        vec3<float> v(-1.0f, 2.0f, -3.0f);
        vec3<float> result;
        
        abs(v, result);
        
        CHECK(result[0] == doctest::Approx(1.0f));
        CHECK(result[1] == doctest::Approx(2.0f));
        CHECK(result[2] == doctest::Approx(3.0f));
    }
    
    SUBCASE("Absolute value with aliasing") {
        vec3<float> v(-1.0f, 2.0f, -3.0f);
        vec3<float> expected(1.0f, 2.0f, 3.0f);
        
        abs(v, v);  // v = |v|
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("Square root") {
        vec3<float> v(4.0f, 9.0f, 16.0f);
        vec3<float> result;
        
        sqrt(v, result);
        
        CHECK(result[0] == doctest::Approx(2.0f));
        CHECK(result[1] == doctest::Approx(3.0f));
        CHECK(result[2] == doctest::Approx(4.0f));
    }
    
    SUBCASE("Reciprocal square root") {
        vec3<float> v(4.0f, 9.0f, 16.0f);
        vec3<float> result;
        
        rsqrt(v, result);
        
        CHECK(result[0] == doctest::Approx(0.5f));
        CHECK(result[1] == doctest::Approx(1.0f/3.0f));
        CHECK(result[2] == doctest::Approx(0.25f));
    }
}

// =============================================================================
// Min/Max/Clamp Operations Tests
// =============================================================================

TEST_CASE("Min/Max/Clamp operations") {
    SUBCASE("Element-wise minimum") {
        vec3<float> a(1.0f, 5.0f, 3.0f);
        vec3<float> b(2.0f, 4.0f, 6.0f);
        vec3<float> result;
        
        min(a, b, result);
        
        CHECK(result[0] == doctest::Approx(1.0f));
        CHECK(result[1] == doctest::Approx(4.0f));
        CHECK(result[2] == doctest::Approx(3.0f));
    }
    
    SUBCASE("Element-wise maximum") {
        vec3<float> a(1.0f, 5.0f, 3.0f);
        vec3<float> b(2.0f, 4.0f, 6.0f);
        vec3<float> result;
        
        max(a, b, result);
        
        CHECK(result[0] == doctest::Approx(2.0f));
        CHECK(result[1] == doctest::Approx(5.0f));
        CHECK(result[2] == doctest::Approx(6.0f));
    }
    
    SUBCASE("Clamp with vector bounds") {
        vec3<float> v(0.5f, 5.0f, -2.0f);
        vec3<float> low(1.0f, 2.0f, -1.0f);
        vec3<float> high(4.0f, 4.0f, 1.0f);
        vec3<float> result;
        
        clamp(v, low, high, result);
        
        CHECK(result[0] == doctest::Approx(1.0f));   // clamped to low
        CHECK(result[1] == doctest::Approx(4.0f));   // clamped to high
        CHECK(result[2] == doctest::Approx(-1.0f));  // clamped to low
    }
    
    SUBCASE("Clamp with scalar bounds") {
        vec3<float> v(0.5f, 5.0f, 2.5f);
        vec3<float> result;
        
        clamp(v, 1.0f, 4.0f, result);
        
        CHECK(result[0] == doctest::Approx(1.0f));   // clamped to low
        CHECK(result[1] == doctest::Approx(4.0f));   // clamped to high
        CHECK(result[2] == doctest::Approx(2.5f));   // unchanged
    }
    
    SUBCASE("Min/Max/Clamp with aliasing") {
        vec3<float> a(1.0f, 5.0f, 3.0f);
        vec3<float> b(2.0f, 4.0f, 6.0f);
        
        min(a, b, a);  // a = min(a, b)
        CHECK(a[0] == doctest::Approx(1.0f));
        CHECK(a[1] == doctest::Approx(4.0f));
        CHECK(a[2] == doctest::Approx(3.0f));
        
        vec3<float> v(0.5f, 5.0f, 2.5f);
        clamp(v, 1.0f, 4.0f, v);  // v = clamp(v, 1, 4)
        CHECK(v[0] == doctest::Approx(1.0f));
        CHECK(v[1] == doctest::Approx(4.0f));
        CHECK(v[2] == doctest::Approx(2.5f));
    }
}

// =============================================================================
// Fused Multiply-Add Tests
// =============================================================================

TEST_CASE("Fused multiply-add operations") {
    SUBCASE("Basic FMA") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> c(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        fma(a, b, c, result);  // result = a * b + c
        
        CHECK(result[0] == doctest::Approx(11.0f));  // 2*5 + 1
        CHECK(result[1] == doctest::Approx(20.0f));  // 3*6 + 2
        CHECK(result[2] == doctest::Approx(31.0f));  // 4*7 + 3
    }
    
    SUBCASE("FMA with scalar a") {
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> c(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        fma(2.0f, b, c, result);  // result = 2 * b + c
        
        CHECK(result[0] == doctest::Approx(11.0f));  // 2*5 + 1
        CHECK(result[1] == doctest::Approx(14.0f));  // 2*6 + 2
        CHECK(result[2] == doctest::Approx(17.0f));  // 2*7 + 3
    }
    
    SUBCASE("FMA with scalar b") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> c(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        fma(a, 5.0f, c, result);  // result = a * 5 + c
        
        CHECK(result[0] == doctest::Approx(11.0f));  // 2*5 + 1
        CHECK(result[1] == doctest::Approx(17.0f));  // 3*5 + 2
        CHECK(result[2] == doctest::Approx(23.0f));  // 4*5 + 3
    }
    
    SUBCASE("FMA with scalar c") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> result;
        
        fma(a, b, 10.0f, result);  // result = a * b + 10
        
        CHECK(result[0] == doctest::Approx(20.0f));  // 2*5 + 10
        CHECK(result[1] == doctest::Approx(28.0f));  // 3*6 + 10
        CHECK(result[2] == doctest::Approx(38.0f));  // 4*7 + 10
    }
    
    SUBCASE("FMA with aliasing") {
        vec3<float> a(2.0f, 3.0f, 4.0f);
        vec3<float> b(5.0f, 6.0f, 7.0f);
        vec3<float> c(1.0f, 2.0f, 3.0f);
        vec3<float> expected(11.0f, 20.0f, 31.0f);
        
        fma(a, b, c, a);  // a = a * b + c
        CHECK(approx_equal(a, expected, FLOAT_TOL));
        
        vec3<float> v(2.0f, 3.0f, 4.0f);
        vec3<float> w(1.0f, 2.0f, 3.0f);
        vec3<float> expected2(11.0f, 17.0f, 23.0f);
        
        fma(v, 5.0f, w, v);  // v = v * 5 + w
        CHECK(approx_equal(v, expected2, FLOAT_TOL));
    }
}

// =============================================================================
// Edge Cases and Special Values
// =============================================================================

TEST_CASE("Edge cases and special values") {
    SUBCASE("Operations with zero vectors") {
        vec3<float> zero(0.0f, 0.0f, 0.0f);
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        add(zero, v, result);
        CHECK(approx_equal(result, v, FLOAT_TOL));
        
        mul(zero, v, result);
        CHECK(approx_equal(result, zero, FLOAT_TOL));
    }
    
    SUBCASE("Normalization of zero vector") {
        vec3<float> zero(0.0f, 0.0f, 0.0f);
        vec3<float> result;
        
        // This will produce inf or nan - just check it doesn't crash
        normalize(zero, result);
        // Result will have inf or nan components
    }
    
    SUBCASE("Very small vectors") {
        vec3<float> tiny(1e-30f, 1e-30f, 1e-30f);
        vec3<float> result;
        
        // Should handle denormals gracefully
        add(tiny, tiny, result);
        normalize(tiny, result);
    }
}

// =============================================================================
// Performance comparison test (not a unit test, but useful for verification)
// =============================================================================

TEST_CASE("Performance verification") {
    SUBCASE("Direct operations should not be slower than expression templates") {
        RandomVectorGenerator<float> rng;
        const unsigned int iterations = 1000;
        
        // Generate test data
        std::vector<vec3<float>> a_vecs, b_vecs, c_vecs;
        for (unsigned int i = 0; i < iterations; ++i) {
            a_vecs.push_back(rng.generate<3>());
            b_vecs.push_back(rng.generate<3>());
            c_vecs.push_back(vec3<float>{});
        }
        
        // Time direct operations
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < iterations; ++i) {
            add(a_vecs[i], b_vecs[i], c_vecs[i]);
        }
        auto direct_time = std::chrono::high_resolution_clock::now() - start;
        
        // Time expression template operations
        start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < iterations; ++i) {
            c_vecs[i] = a_vecs[i] + b_vecs[i];
        }
        auto expr_time = std::chrono::high_resolution_clock::now() - start;
        
        // Direct should not be significantly slower
        // (In practice, it should be faster for simple operations)
        // Use generous 10x threshold to avoid flaky failures on CI runners
        // where timing measurements are unreliable due to shared resources
        CHECK(direct_time.count() < expr_time.count() * 10);
    }
}