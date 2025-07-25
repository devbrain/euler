#include <euler/math/basic.hh>
#include <euler/complex/complex.hh>
#include <euler/complex/complex_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>
#include <complex>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Complex math functions - Square root") {
    SUBCASE("Basic complex square root") {
        // sqrt(i) = (1+i)/sqrt(2)
        complexf z1(0.0f, 1.0f);
        complexf sqrt_z1 = sqrt(z1);
        float expected_val = 1.0f / std::sqrt(2.0f);
        CHECK(approx_equal(sqrt_z1.real(), expected_val, 1e-6f));
        CHECK(approx_equal(sqrt_z1.imag(), expected_val, 1e-6f));
        
        // sqrt(-1) = i
        complexf z2(-1.0f, 0.0f);
        complexf sqrt_z2 = sqrt(z2);
        CHECK(approx_equal(sqrt_z2.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(sqrt_z2.imag(), 1.0f, 1e-6f));
        
        // sqrt(3+4i)
        complexf z3(3.0f, 4.0f);
        complexf sqrt_z3 = sqrt(z3);
        // Verify: (sqrt_z3)^2 = z3
        complexf squared = sqrt_z3 * sqrt_z3;
        CHECK(approx_equal(squared, z3, 1e-5f));
        
        // Compare with std::complex
        std::complex<float> sz3(3.0f, 4.0f);
        std::complex<float> std_sqrt = std::sqrt(sz3);
        CHECK(approx_equal(sqrt_z3.real(), std_sqrt.real(), 1e-6f));
        CHECK(approx_equal(sqrt_z3.imag(), std_sqrt.imag(), 1e-6f));
    }
}

TEST_CASE("Complex math functions - Power") {
    SUBCASE("Complex base, complex exponent") {
        complexf base(2.0f, 1.0f);
        complexf exp(0.5f, 0.0f);
        complexf result = pow(base, exp);
        
        // Verify with std::complex
        std::complex<float> sbase(2.0f, 1.0f);
        std::complex<float> sexp(0.5f, 0.0f);
        std::complex<float> std_result = std::pow(sbase, sexp);
        CHECK(approx_equal(result.real(), std_result.real(), 1e-5f));
        CHECK(approx_equal(result.imag(), std_result.imag(), 1e-5f));
    }
    
    SUBCASE("Complex base, real exponent") {
        complexf base(1.0f, 1.0f);
        float exp = 2.0f;
        complexf result = pow(base, exp);
        
        // (1+i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2i
        CHECK(approx_equal(result.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(result.imag(), 2.0f, 1e-6f));
    }
    
    SUBCASE("Real base, complex exponent") {
        float base = 2.0f;
        complexf exp(0.0f, constants<float>::pi);
        complexf result = pow(base, exp);
        
        // 2^(i*pi) = exp(i*pi*ln(2)) = cos(pi*ln(2)) + i*sin(pi*ln(2))
        std::complex<float> std_result = std::pow(2.0f, std::complex<float>(0.0f, constants<float>::pi));
        CHECK(approx_equal(result.real(), std_result.real(), 1e-5f));
        CHECK(approx_equal(result.imag(), std_result.imag(), 1e-5f));
    }
    
    SUBCASE("Special cases") {
        // 0^0 = 1
        complexf zero(0.0f, 0.0f);
        complexf result1 = pow(zero, zero);
        CHECK(approx_equal(result1, complexf(1.0f, 0.0f)));
        
        // 0^z = 0 for z != 0
        complexf z(1.0f, 1.0f);
        complexf result2 = pow(zero, z);
        CHECK(approx_equal(result2, complexf(0.0f, 0.0f)));
    }
}

TEST_CASE("Complex math functions - Exponential") {
    SUBCASE("Complex exponential") {
        // exp(i*pi) = -1 (Euler's identity)
        complexf z1(0.0f, constants<float>::pi);
        complexf exp_z1 = exp(z1);
        CHECK(approx_equal(exp_z1.real(), -1.0f, 1e-6f));
        CHECK(approx_equal(exp_z1.imag(), 0.0f, 1e-6f));
        
        // exp(1+i)
        complexf z2(1.0f, 1.0f);
        complexf exp_z2 = exp(z2);
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_exp = std::exp(sz2);
        CHECK(approx_equal(exp_z2.real(), std_exp.real(), 1e-6f));
        CHECK(approx_equal(exp_z2.imag(), std_exp.imag(), 1e-6f));
        
        // exp(0) = 1
        complexf z3(0.0f, 0.0f);
        complexf exp_z3 = exp(z3);
        CHECK(approx_equal(exp_z3, complexf(1.0f, 0.0f)));
    }
    
    SUBCASE("Euler's formula") {
        // exp(i*theta) = cos(theta) + i*sin(theta)
        float theta = constants<float>::pi / 4; // 45 degrees
        complexf z(0.0f, theta);
        complexf exp_z = exp(z);
        
        CHECK(approx_equal(exp_z.real(), std::cos(theta), 1e-6f));
        CHECK(approx_equal(exp_z.imag(), std::sin(theta), 1e-6f));
    }
}

TEST_CASE("Complex math functions - Logarithm") {
    SUBCASE("Complex logarithm") {
        // log(1) = 0
        complexf z1(1.0f, 0.0f);
        complexf log_z1 = log(z1);
        CHECK(approx_equal(log_z1, complexf(0.0f, 0.0f)));
        
        // log(e) = 1
        complexf z2(constants<float>::e, 0.0f);
        complexf log_z2 = log(z2);
        CHECK(approx_equal(log_z2, complexf(1.0f, 0.0f), 1e-6f));
        
        // log(i) = i*pi/2
        complexf z3(0.0f, 1.0f);
        complexf log_z3 = log(z3);
        CHECK(approx_equal(log_z3.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(log_z3.imag(), constants<float>::pi / 2, 1e-6f));
        
        // log(-1) = i*pi
        complexf z4(-1.0f, 0.0f);
        complexf log_z4 = log(z4);
        CHECK(approx_equal(log_z4.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(log_z4.imag(), constants<float>::pi, 1e-6f));
        
        // General case
        complexf z5(3.0f, 4.0f);
        complexf log_z5 = log(z5);
        
        // Compare with std::complex
        std::complex<float> sz5(3.0f, 4.0f);
        std::complex<float> std_log = std::log(sz5);
        CHECK(approx_equal(log_z5.real(), std_log.real(), 1e-6f));
        CHECK(approx_equal(log_z5.imag(), std_log.imag(), 1e-6f));
    }
    
    SUBCASE("Log-exp identity") {
        complexf z(2.0f, 1.0f);
        
        // exp(log(z)) = z
        complexf result1 = exp(log(z));
        CHECK(approx_equal(result1, z, 1e-5f));
        
        // log(exp(z)) = z (modulo 2πi)
        complexf result2 = log(exp(z));
        CHECK(approx_equal(result2, z, 1e-5f));
    }
}

TEST_CASE("Complex math functions with expressions") {
    SUBCASE("Complex expressions") {
        complexf z1(1.0f, 2.0f);
        complexf z2(3.0f, -1.0f);
        
        // sqrt(z1) * exp(z2 * 0.1f) + log(z1 + z2)
        auto expr = sqrt(z1) * exp(z2 * 0.1f) + log(z1 + z2);
        complexf result = expr;
        
        // Manual calculation
        complexf sqrt_z1 = sqrt(z1);
        complexf exp_term = exp(z2 * 0.1f);
        complexf log_term = log(z1 + z2);
        complexf expected = sqrt_z1 * exp_term + log_term;
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Mixed real and complex") {
        float r = 2.0f;
        complexf z(1.0f, 1.0f);
        
        // pow(r, z) + exp(z * r) - log(r + z)
        auto expr = pow(r, z) + exp(z * r) - log(r + z);
        complexf result = expr;
        
        // Manual calculation
        complexf pow_term = pow(r, z);
        complexf exp_term = exp(z * r);
        complexf log_term = log(r + z);
        complexf expected = pow_term + exp_term - log_term;
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
}

TEST_CASE("Complex math functions - Special values") {
    SUBCASE("Operations on pure real") {
        complexf z(2.5f, 0.0f);
        
        CHECK(approx_equal(sqrt(z).real(), std::sqrt(2.5f), 1e-6f));
        CHECK(approx_equal(sqrt(z).imag(), 0.0f, 1e-6f));
        
        CHECK(approx_equal(exp(z).real(), std::exp(2.5f), 1e-6f));
        CHECK(approx_equal(exp(z).imag(), 0.0f, 1e-6f));
        
        CHECK(approx_equal(log(z).real(), std::log(2.5f), 1e-6f));
        CHECK(approx_equal(log(z).imag(), 0.0f, 1e-6f));
    }
    
    SUBCASE("Operations on pure imaginary") {
        complexf z(0.0f, 2.0f);
        
        // sqrt(2i)
        complexf sqrt_z = sqrt(z);
        // Verify: sqrt_z^2 = 2i
        complexf squared = sqrt_z * sqrt_z;
        CHECK(approx_equal(squared, z, 1e-5f));
        
        // exp(2i) = cos(2) + i*sin(2)
        complexf exp_z = exp(z);
        CHECK(approx_equal(exp_z.real(), std::cos(2.0f), 1e-6f));
        CHECK(approx_equal(exp_z.imag(), std::sin(2.0f), 1e-6f));
        
        // log(2i) = log(2) + i*pi/2
        complexf log_z = log(z);
        CHECK(approx_equal(log_z.real(), std::log(2.0f), 1e-6f));
        CHECK(approx_equal(log_z.imag(), constants<float>::pi / 2, 1e-6f));
    }
}