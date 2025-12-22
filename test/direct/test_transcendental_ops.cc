/**
 * @file test_transcendental_ops.cc
 * @brief Unit tests for direct SIMD transcendental operations
 */

#include <doctest/doctest.h>
#include <euler/direct/transcendental_ops.hh>
#include <euler/direct/vector_ops.hh>
#include <euler/vector/vector.hh>
#include <euler/core/approx_equal.hh>
#include <euler/core/types.hh>
#include <euler/core/compiler.hh>
#include <random>
#include <chrono>
#include <vector>

// Disable strict overflow warnings for xsimd's internal functions
EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW

using namespace euler;
using namespace euler::direct;

// Test configuration
constexpr float FLOAT_TOL = 1e-5f;
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
    
    template<size_t N>
    vector<T, N> generate_positive() {
        std::uniform_real_distribution<T> pos_dist(T(0.1), T(10));
        vector<T, N> v;
        for (size_t i = 0; i < N; ++i) {
            v[i] = pos_dist(gen);
        }
        return v;
    }
    
    template<size_t N>
    vector<T, N> generate_range(T min_val, T max_val) {
        std::uniform_real_distribution<T> range_dist(min_val, max_val);
        vector<T, N> v;
        for (size_t i = 0; i < N; ++i) {
            v[i] = range_dist(gen);
        }
        return v;
    }
    
private:
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
};

// =============================================================================
// Exponential and Logarithmic Functions Tests
// =============================================================================

TEST_CASE("Direct exponential operations") {
    RandomVectorGenerator<float> rng_f;
    RandomVectorGenerator<double> rng_d;
    
    SUBCASE("Basic exp - vec3<float>") {
        vec3<float> v(0.0f, 1.0f, 2.0f);
        vec3<float> result;
        
        exp(v, result);
        
        CHECK(result[0] == doctest::Approx(1.0f));
        CHECK(result[1] == doctest::Approx(std::exp(1.0f)));
        CHECK(result[2] == doctest::Approx(std::exp(2.0f)));
    }
    
    SUBCASE("exp with aliasing") {
        vec3<float> v(0.0f, 1.0f, -1.0f);
        vec3<float> expected(1.0f, std::exp(1.0f), std::exp(-1.0f));
        
        exp(v, v);  // v = exp(v)
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("log of positive values") {
        vec3<float> v(1.0f, constants<float>::e, 10.0f);
        vec3<float> result;
        
        log(v, result);
        
        CHECK(result[0] == doctest::Approx(0.0f));
        CHECK(result[1] == doctest::Approx(1.0f));
        CHECK(result[2] == doctest::Approx(std::log(10.0f)));
    }
    
    SUBCASE("log10 and log2") {
        vec3<float> v(1.0f, 10.0f, 100.0f);
        vec3<float> result10, result2;
        
        log10(v, result10);
        log2(v, result2);
        
        CHECK(result10[0] == doctest::Approx(0.0f));
        CHECK(result10[1] == doctest::Approx(1.0f));
        CHECK(result10[2] == doctest::Approx(2.0f));
        
        CHECK(result2[0] == doctest::Approx(0.0f));
        CHECK(result2[1] == doctest::Approx(std::log2(10.0f)));
        CHECK(result2[2] == doctest::Approx(std::log2(100.0f)));
    }
    
    SUBCASE("pow with scalar exponent") {
        vec3<float> v(2.0f, 3.0f, 4.0f);
        vec3<float> result;
        
        pow(v, 2.0f, result);
        
        CHECK(result[0] == doctest::Approx(4.0f));
        CHECK(result[1] == doctest::Approx(9.0f));
        CHECK(result[2] == doctest::Approx(16.0f));
    }
    
    SUBCASE("pow with vector exponent") {
        vec3<float> base(2.0f, 3.0f, 4.0f);
        vec3<float> exponent(1.0f, 2.0f, 0.5f);
        vec3<float> result;
        
        pow(base, exponent, result);
        
        CHECK(result[0] == doctest::Approx(2.0f));
        CHECK(result[1] == doctest::Approx(9.0f));
        CHECK(result[2] == doctest::Approx(2.0f));
    }
    
    SUBCASE("Random exp/log tests") {
        for (int test = 0; test < 10; ++test) {
            auto v = rng_f.generate_range<4>(-5.0f, 5.0f);
            vec4<float> exp_result, log_exp_result;
            
            exp(v, exp_result);
            log(exp_result, log_exp_result);
            
            // log(exp(v)) should equal v (within tolerance)
            CHECK(approx_equal(log_exp_result, v, FLOAT_TOL * 10));
        }
    }
    
    SUBCASE("Exponential operations - double precision") {
        vec3<double> v(0.0, 1.0, 2.0);
        vec3<double> result;
        
        exp(v, result);
        
        CHECK(result[0] == doctest::Approx(1.0));
        CHECK(result[1] == doctest::Approx(constants<double>::e));
        CHECK(result[2] == doctest::Approx(std::exp(2.0)));
    }
}

// =============================================================================
// Trigonometric Functions Tests
// =============================================================================

TEST_CASE("Direct trigonometric operations") {
    SUBCASE("Basic sin/cos/tan") {
        vec4<float> angles(0.0f, constants<float>::pi/6, constants<float>::pi/4, constants<float>::pi/2);
        vec4<float> sin_result, cos_result, tan_result;
        
        sin(angles, sin_result);
        cos(angles, cos_result);
        tan(angles, tan_result);
        
        // sin values
        CHECK(sin_result[0] == doctest::Approx(0.0f));
        CHECK(sin_result[1] == doctest::Approx(0.5f));
        CHECK(sin_result[2] == doctest::Approx(std::sqrt(2.0f)/2.0f));
        CHECK(sin_result[3] == doctest::Approx(1.0f));
        
        // cos values
        CHECK(cos_result[0] == doctest::Approx(1.0f));
        CHECK(cos_result[1] == doctest::Approx(std::sqrt(3.0f)/2.0f));
        CHECK(cos_result[2] == doctest::Approx(std::sqrt(2.0f)/2.0f));
        CHECK(cos_result[3] == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        
        // tan values
        CHECK(tan_result[0] == doctest::Approx(0.0f));
        CHECK(tan_result[1] == doctest::Approx(1.0f/std::sqrt(3.0f)));
        CHECK(tan_result[2] == doctest::Approx(1.0f));
    }
    
    SUBCASE("sincos simultaneous computation") {
        vec3<float> angles(0.0f, constants<float>::pi/2, constants<float>::pi);
        vec3<float> sin_result, cos_result;
        
        sincos(angles, sin_result, cos_result);
        
        CHECK(sin_result[0] == doctest::Approx(0.0f));
        CHECK(sin_result[1] == doctest::Approx(1.0f));
        CHECK(sin_result[2] == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        
        CHECK(cos_result[0] == doctest::Approx(1.0f));
        CHECK(cos_result[1] == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        CHECK(cos_result[2] == doctest::Approx(-1.0f));
    }
    
    SUBCASE("Trigonometric identities") {
        RandomVectorGenerator<float> rng;
        
        for (int test = 0; test < 10; ++test) {
            auto angles = rng.generate_range<4>(-constants<float>::pi, constants<float>::pi);
            vec4<float> sin_vals, cos_vals;
            vec4<float> sin_squared, cos_squared, sum;
            
            sin(angles, sin_vals);
            cos(angles, cos_vals);
            
            // sin² + cos² = 1
            mul(sin_vals, sin_vals, sin_squared);
            mul(cos_vals, cos_vals, cos_squared);
            add(sin_squared, cos_squared, sum);
            
            for (size_t i = 0; i < 4; ++i) {
                CHECK(sum[i] == doctest::Approx(1.0f).epsilon(FLOAT_TOL));
            }
        }
    }
    
    SUBCASE("Inverse trigonometric functions") {
        vec3<float> values(-1.0f, 0.0f, 1.0f);
        vec3<float> asin_result, acos_result, atan_result;
        
        asin(values, asin_result);
        acos(values, acos_result);
        atan(values, atan_result);
        
        CHECK(asin_result[0] == doctest::Approx(-constants<float>::pi/2));
        CHECK(asin_result[1] == doctest::Approx(0.0f));
        CHECK(asin_result[2] == doctest::Approx(constants<float>::pi/2));
        
        CHECK(acos_result[0] == doctest::Approx(constants<float>::pi));
        CHECK(acos_result[1] == doctest::Approx(constants<float>::pi/2));
        CHECK(acos_result[2] == doctest::Approx(0.0f));
        
        CHECK(atan_result[0] == doctest::Approx(-constants<float>::pi/4));
        CHECK(atan_result[1] == doctest::Approx(0.0f));
        CHECK(atan_result[2] == doctest::Approx(constants<float>::pi/4));
    }
    
    SUBCASE("atan2 function") {
        vec3<float> y(1.0f, 1.0f, -1.0f);
        vec3<float> x(1.0f, 0.0f, 1.0f);
        vec3<float> result;
        
        atan2(y, x, result);
        
        CHECK(result[0] == doctest::Approx(constants<float>::pi/4));
        CHECK(result[1] == doctest::Approx(constants<float>::pi/2));
        CHECK(result[2] == doctest::Approx(-constants<float>::pi/4));
    }
}

// =============================================================================
// Hyperbolic Functions Tests
// =============================================================================

TEST_CASE("Direct hyperbolic operations") {
    SUBCASE("Basic sinh/cosh/tanh") {
        vec3<float> v(0.0f, 1.0f, -1.0f);
        vec3<float> sinh_result, cosh_result, tanh_result;
        
        sinh(v, sinh_result);
        cosh(v, cosh_result);
        tanh(v, tanh_result);
        
        CHECK(sinh_result[0] == doctest::Approx(0.0f));
        CHECK(sinh_result[1] == doctest::Approx(std::sinh(1.0f)));
        CHECK(sinh_result[2] == doctest::Approx(std::sinh(-1.0f)));
        
        CHECK(cosh_result[0] == doctest::Approx(1.0f));
        CHECK(cosh_result[1] == doctest::Approx(std::cosh(1.0f)));
        CHECK(cosh_result[2] == doctest::Approx(std::cosh(-1.0f)));
        
        CHECK(tanh_result[0] == doctest::Approx(0.0f));
        CHECK(tanh_result[1] == doctest::Approx(std::tanh(1.0f)));
        CHECK(tanh_result[2] == doctest::Approx(std::tanh(-1.0f)));
    }
    
    SUBCASE("Hyperbolic identities") {
        RandomVectorGenerator<float> rng;
        
        for (int test = 0; test < 10; ++test) {
            auto v = rng.generate_range<4>(-2.0f, 2.0f);
            vec4<float> sinh_vals, cosh_vals;
            vec4<float> sinh_squared, cosh_squared, diff;
            
            sinh(v, sinh_vals);
            cosh(v, cosh_vals);
            
            // cosh² - sinh² = 1
            mul(sinh_vals, sinh_vals, sinh_squared);
            mul(cosh_vals, cosh_vals, cosh_squared);
            sub(cosh_squared, sinh_squared, diff);
            
            for (size_t i = 0; i < 4; ++i) {
                CHECK(diff[i] == doctest::Approx(1.0f).epsilon(FLOAT_TOL));
            }
        }
    }
}

// =============================================================================
// Other Mathematical Functions Tests
// =============================================================================

TEST_CASE("Direct rounding operations") {
    SUBCASE("ceil function") {
        vec4<float> v(-1.7f, -1.2f, 1.2f, 1.7f);
        vec4<float> result;
        
        ceil(v, result);
        
        CHECK(result[0] == doctest::Approx(-1.0f));
        CHECK(result[1] == doctest::Approx(-1.0f));
        CHECK(result[2] == doctest::Approx(2.0f));
        CHECK(result[3] == doctest::Approx(2.0f));
    }
    
    SUBCASE("floor function") {
        vec4<float> v(-1.7f, -1.2f, 1.2f, 1.7f);
        vec4<float> result;
        
        floor(v, result);
        
        CHECK(result[0] == doctest::Approx(-2.0f));
        CHECK(result[1] == doctest::Approx(-2.0f));
        CHECK(result[2] == doctest::Approx(1.0f));
        CHECK(result[3] == doctest::Approx(1.0f));
    }
    
    SUBCASE("round function") {
        vec4<float> v(-1.7f, -1.5f, 1.5f, 1.7f);
        vec4<float> result;
        
        round(v, result);
        
        CHECK(result[0] == doctest::Approx(-2.0f));
        CHECK(result[1] == doctest::Approx(-2.0f));  // Round half away from zero
        CHECK(result[2] == doctest::Approx(2.0f));   // Round half away from zero
        CHECK(result[3] == doctest::Approx(2.0f));
    }
    
    SUBCASE("trunc function") {
        vec4<float> v(-1.7f, -1.2f, 1.2f, 1.7f);
        vec4<float> result;
        
        trunc(v, result);
        
        CHECK(result[0] == doctest::Approx(-1.0f));
        CHECK(result[1] == doctest::Approx(-1.0f));
        CHECK(result[2] == doctest::Approx(1.0f));
        CHECK(result[3] == doctest::Approx(1.0f));
    }
}

// =============================================================================
// Edge Cases and Special Values
// =============================================================================

TEST_CASE("Edge cases and special values") {
    SUBCASE("Operations with zero") {
        vec3<float> zero(0.0f, 0.0f, 0.0f);
        vec3<float> result;
        
        exp(zero, result);
        CHECK(approx_equal(result, vec3<float>(1.0f, 1.0f, 1.0f), FLOAT_TOL));
        
        sin(zero, result);
        CHECK(approx_equal(result, zero, FLOAT_TOL));
        
        cos(zero, result);
        CHECK(approx_equal(result, vec3<float>(1.0f, 1.0f, 1.0f), FLOAT_TOL));
        
        tan(zero, result);
        CHECK(approx_equal(result, zero, FLOAT_TOL));
    }
    
    SUBCASE("Very large and small values") {
        vec3<float> large(10.0f, 20.0f, 30.0f);
        vec3<float> tiny(1e-30f, 1e-30f, 1e-30f);
        vec3<float> result;
        (void)tiny;  // unused, just testing initialization
        
        // exp of large values
        exp(large, result);
        CHECK(result[0] == doctest::Approx(std::exp(10.0f)));
        
        // log of small positive values
        vec3<float> small_pos(1e-10f, 1e-20f, 1e-30f);
        log(small_pos, result);
        // Just check it doesn't crash - results may vary
    }
}

// =============================================================================
// Performance comparison test
// =============================================================================

TEST_CASE("Performance verification") {
    SUBCASE("Direct operations should be efficient") {
        RandomVectorGenerator<float> rng;
        constexpr unsigned int iterations = 1000;
        
        // Generate test data
        std::vector<vec4<float>> input_vecs, output_vecs;
        for (unsigned int i = 0; i < iterations; ++i) {
            input_vecs.push_back(rng.generate_range<4>(-constants<float>::pi, constants<float>::pi));
            output_vecs.push_back(vec4<float>{});
        }
        
        // Time direct operations
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < iterations; ++i) {
            sin(input_vecs[i], output_vecs[i]);
        }
        auto direct_time = std::chrono::high_resolution_clock::now() - start;
        
        // Time standard operations
        start = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < iterations; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                output_vecs[i][j] = std::sin(input_vecs[i][j]);
            }
        }
        auto std_time = std::chrono::high_resolution_clock::now() - start;
        
        // Direct should not be significantly slower
        CHECK(direct_time.count() < std_time.count() * 3);
    }
}

EULER_DISABLE_WARNING_POP