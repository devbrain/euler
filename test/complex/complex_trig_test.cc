#include <euler/complex/complex.hh>
#include <euler/complex/complex_ops.hh>
#include <euler/math/trigonometry.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>
#include <complex>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Complex trigonometric functions") {
    SUBCASE("Complex sine") {
        // sin(i) = i*sinh(1)
        complexf z1(0.0f, 1.0f);
        complexf sin_z1 = euler::sin(z1);
        CHECK(approx_equal(sin_z1.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(sin_z1.imag(), std::sinh(1.0f), 1e-6f));
        
        // sin(1+i)
        complexf z2(1.0f, 1.0f);
        complexf sin_z2 = euler::sin(z2);
        float expected_real = std::sin(1.0f) * std::cosh(1.0f);
        float expected_imag = std::cos(1.0f) * std::sinh(1.0f);
        CHECK(approx_equal(sin_z2.real(), expected_real, 1e-6f));
        CHECK(approx_equal(sin_z2.imag(), expected_imag, 1e-6f));
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_sin = std::sin(sz2);
        CHECK(approx_equal(sin_z2.real(), std_sin.real(), 1e-6f));
        CHECK(approx_equal(sin_z2.imag(), std_sin.imag(), 1e-6f));
    }
    
    SUBCASE("Complex cosine") {
        // cos(i) = cosh(1)
        complexf z1(0.0f, 1.0f);
        complexf cos_z1 = euler::cos(z1);
        CHECK(approx_equal(cos_z1.real(), std::cosh(1.0f), 1e-6f));
        CHECK(approx_equal(cos_z1.imag(), 0.0f, 1e-6f));
        
        // cos(1+i)
        complexf z2(1.0f, 1.0f);
        complexf cos_z2 = euler::cos(z2);
        float expected_real = std::cos(1.0f) * std::cosh(1.0f);
        float expected_imag = -std::sin(1.0f) * std::sinh(1.0f);
        CHECK(approx_equal(cos_z2.real(), expected_real, 1e-6f));
        CHECK(approx_equal(cos_z2.imag(), expected_imag, 1e-6f));
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_cos = std::cos(sz2);
        CHECK(approx_equal(cos_z2.real(), std_cos.real(), 1e-6f));
        CHECK(approx_equal(cos_z2.imag(), std_cos.imag(), 1e-6f));
    }
    
    SUBCASE("Complex tangent") {
        // tan(i) = i*tanh(1)
        complexf z1(0.0f, 1.0f);
        complexf tan_z1 = euler::tan(z1);
        CHECK(approx_equal(tan_z1.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(tan_z1.imag(), std::tanh(1.0f), 1e-6f));
        
        // tan(1+i)
        complexf z2(1.0f, 1.0f);
        complexf tan_z2 = euler::tan(z2);
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_tan = std::tan(sz2);
        CHECK(approx_equal(tan_z2.real(), std_tan.real(), 1e-5f));
        CHECK(approx_equal(tan_z2.imag(), std_tan.imag(), 1e-5f));
    }
    
    SUBCASE("Trigonometric identities") {
        complexf z(0.5f, 0.3f);
        
        // sin²(z) + cos²(z) = 1
        complexf sin_z = euler::sin(z);
        complexf cos_z = euler::cos(z);
        complexf identity1 = sin_z * sin_z + cos_z * cos_z;
        CHECK(approx_equal(identity1, complexf(1.0f, 0.0f), 1e-5f));
        
        // tan(z) = sin(z) / cos(z)
        complexf tan_z = euler::tan(z);
        complexf tan_from_sincos = sin_z / cos_z;
        CHECK(approx_equal(tan_z, tan_from_sincos, 1e-5f));
    }
}

TEST_CASE("Complex hyperbolic functions") {
    SUBCASE("Complex sinh") {
        // sinh(i) = i*sin(1)
        complexf z1(0.0f, 1.0f);
        complexf sinh_z1 = euler::sinh(z1);
        CHECK(approx_equal(sinh_z1.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(sinh_z1.imag(), std::sin(1.0f), 1e-6f));
        
        // sinh(1+i)
        complexf z2(1.0f, 1.0f);
        complexf sinh_z2 = euler::sinh(z2);
        float expected_real = std::sinh(1.0f) * std::cos(1.0f);
        float expected_imag = std::cosh(1.0f) * std::sin(1.0f);
        CHECK(approx_equal(sinh_z2.real(), expected_real, 1e-6f));
        CHECK(approx_equal(sinh_z2.imag(), expected_imag, 1e-6f));
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_sinh = std::sinh(sz2);
        CHECK(approx_equal(sinh_z2.real(), std_sinh.real(), 1e-6f));
        CHECK(approx_equal(sinh_z2.imag(), std_sinh.imag(), 1e-6f));
    }
    
    SUBCASE("Complex cosh") {
        // cosh(i) = cos(1)
        complexf z1(0.0f, 1.0f);
        complexf cosh_z1 = euler::cosh(z1);
        CHECK(approx_equal(cosh_z1.real(), std::cos(1.0f), 1e-6f));
        CHECK(approx_equal(cosh_z1.imag(), 0.0f, 1e-6f));
        
        // cosh(1+i)
        complexf z2(1.0f, 1.0f);
        complexf cosh_z2 = euler::cosh(z2);
        float expected_real = std::cosh(1.0f) * std::cos(1.0f);
        float expected_imag = std::sinh(1.0f) * std::sin(1.0f);
        CHECK(approx_equal(cosh_z2.real(), expected_real, 1e-6f));
        CHECK(approx_equal(cosh_z2.imag(), expected_imag, 1e-6f));
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_cosh = std::cosh(sz2);
        CHECK(approx_equal(cosh_z2.real(), std_cosh.real(), 1e-6f));
        CHECK(approx_equal(cosh_z2.imag(), std_cosh.imag(), 1e-6f));
    }
    
    SUBCASE("Complex tanh") {
        // tanh(i) = i*tan(1)
        complexf z1(0.0f, 1.0f);
        complexf tanh_z1 = euler::tanh(z1);
        CHECK(approx_equal(tanh_z1.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(tanh_z1.imag(), std::tan(1.0f), 1e-6f));
        
        // tanh(1+i)
        complexf z2(1.0f, 1.0f);
        complexf tanh_z2 = euler::tanh(z2);
        
        // Compare with std::complex
        std::complex<float> sz2(1.0f, 1.0f);
        std::complex<float> std_tanh = std::tanh(sz2);
        CHECK(approx_equal(tanh_z2.real(), std_tanh.real(), 1e-5f));
        CHECK(approx_equal(tanh_z2.imag(), std_tanh.imag(), 1e-5f));
    }
    
    SUBCASE("Hyperbolic identities") {
        complexf z(0.5f, 0.3f);
        
        // cosh²(z) - sinh²(z) = 1
        complexf sinh_z = euler::sinh(z);
        complexf cosh_z = euler::cosh(z);
        complexf identity1 = cosh_z * cosh_z - sinh_z * sinh_z;
        CHECK(approx_equal(identity1, complexf(1.0f, 0.0f), 1e-5f));
        
        // tanh(z) = sinh(z) / cosh(z)
        complexf tanh_z = euler::tanh(z);
        complexf tanh_from_sinhcosh = sinh_z / cosh_z;
        CHECK(approx_equal(tanh_z, tanh_from_sinhcosh, 1e-5f));
    }
}

TEST_CASE("Complex trigonometric with expression templates") {
    SUBCASE("Complex expressions") {
        complexf z1(0.5f, 0.2f);
        complexf z2(0.3f, 0.4f);
        
        // Expression with complex trigonometry
        auto expr = euler::sin(z1) * euler::cos(z2) + euler::tan(z1 + z2);
        complexf result = expr;
        
        // Manual calculation
        complexf sin_z1 = euler::sin(z1);
        complexf cos_z2 = euler::cos(z2);
        complexf tan_sum = euler::tan(z1 + z2);
        complexf expected = sin_z1 * cos_z2 + tan_sum;
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
    
    SUBCASE("Mixed real and complex") {
        float r = 0.5f;
        complexf z(0.3f, 0.4f);
        
        // Expression mixing real and complex
        auto expr = euler::sin(r) * euler::cos(z) + euler::sinh(z) * r;
        complexf result = expr;
        
        // Manual calculation
        float sin_r = std::sin(r);
        complexf cos_z = euler::cos(z);
        complexf sinh_z = euler::sinh(z);
        complexf expected = sin_r * cos_z + sinh_z * r;
        
        CHECK(approx_equal(result, expected, 1e-5f));
    }
}

TEST_CASE("Complex trigonometric special values") {
    SUBCASE("Pure real") {
        complexf z(1.5f, 0.0f);
        
        // Should match real trigonometry
        CHECK(approx_equal(euler::sin(z).real(), std::sin(1.5f), 1e-6f));
        CHECK(approx_equal(euler::sin(z).imag(), 0.0f, 1e-6f));
        
        CHECK(approx_equal(euler::cos(z).real(), std::cos(1.5f), 1e-6f));
        CHECK(approx_equal(euler::cos(z).imag(), 0.0f, 1e-6f));
    }
    
    SUBCASE("Pure imaginary") {
        complexf z(0.0f, 1.5f);
        
        // sin(iy) = i*sinh(y)
        CHECK(approx_equal(euler::sin(z).real(), 0.0f, 1e-6f));
        CHECK(approx_equal(euler::sin(z).imag(), std::sinh(1.5f), 1e-6f));
        
        // cos(iy) = cosh(y)
        CHECK(approx_equal(euler::cos(z).real(), std::cosh(1.5f), 1e-6f));
        CHECK(approx_equal(euler::cos(z).imag(), 0.0f, 1e-6f));
    }
    
    SUBCASE("Zero") {
        complexf z(0.0f, 0.0f);
        
        CHECK(approx_equal(euler::sin(z), complexf(0.0f, 0.0f)));
        CHECK(approx_equal(euler::cos(z), complexf(1.0f, 0.0f)));
        CHECK(approx_equal(euler::tan(z), complexf(0.0f, 0.0f)));
        
        CHECK(approx_equal(euler::sinh(z), complexf(0.0f, 0.0f)));
        CHECK(approx_equal(euler::cosh(z), complexf(1.0f, 0.0f)));
        CHECK(approx_equal(euler::tanh(z), complexf(0.0f, 0.0f)));
    }
}

TEST_CASE("Complex trigonometric with angles") {
    SUBCASE("Euler's formula") {
        // e^(i*theta) = cos(theta) + i*sin(theta)
        degree<float> theta = 45.0_deg;
        
        // Create complex number from angle using polar form
        complexf z = polar(1.0f, theta);
        
        // Should equal cos(45°) + i*sin(45°)
        float cos_45 = std::cos(45.0f * constants<float>::deg_to_rad);
        float sin_45 = std::sin(45.0f * constants<float>::deg_to_rad);
        
        CHECK(approx_equal(z.real(), cos_45, 1e-6f));
        CHECK(approx_equal(z.imag(), sin_45, 1e-6f));
        
        // Check magnitude and argument
        CHECK(approx_equal(z.abs(), 1.0f));
        CHECK(approx_equal(z.arg(), radian<float>(theta)));
    }
    
    SUBCASE("Complex rotation") {
        complexf z(1.0f, 1.0f);
        degree<float> rotation_angle = 90.0_deg;
        
        // Rotate by multiplying with e^(i*angle)
        complexf rotation = polar(1.0f, rotation_angle);
        complexf rotated = z * rotation;
        
        // 1+i rotated by 90° should give -1+i
        CHECK(approx_equal(rotated.real(), -1.0f, 1e-6f));
        CHECK(approx_equal(rotated.imag(), 1.0f, 1e-6f));
        
        // Check the angle increased by 90°
        degree<float> original_angle = z.arg_deg();
        degree<float> new_angle = rotated.arg_deg();
        CHECK(approx_equal((new_angle - original_angle).value(), 90.0f, 1e-5f));
    }
}