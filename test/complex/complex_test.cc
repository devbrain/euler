#include <euler/complex/complex.hh>
#include <euler/complex/complex_ops.hh>
#include <euler/core/approx_equal.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <doctest/doctest.h>
#include <cmath>
#include <complex>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Complex number construction") {
    SUBCASE("Default construction") {
        complexf z;
        CHECK(z.real() == 0.0f);
        CHECK(z.imag() == 0.0f);
    }
    
    SUBCASE("Real-only construction") {
        complexf z(3.0f);
        CHECK(z.real() == 3.0f);
        CHECK(z.imag() == 0.0f);
    }
    
    SUBCASE("Real and imaginary construction") {
        complexf z(3.0f, 4.0f);
        CHECK(z.real() == 3.0f);
        CHECK(z.imag() == 4.0f);
    }
    
    SUBCASE("Copy construction") {
        complexf z1(3.0f, 4.0f);
        complexf z2(z1);
        CHECK(z2.real() == 3.0f);
        CHECK(z2.imag() == 4.0f);
    }
    
    SUBCASE("Conversion from std::complex") {
        std::complex<float> sc(3.0f, 4.0f);
        complexf z(sc);
        CHECK(z.real() == 3.0f);
        CHECK(z.imag() == 4.0f);
    }
    
    SUBCASE("Conversion to std::complex") {
        complexf z(3.0f, 4.0f);
        std::complex<float> sc = z;
        CHECK(sc.real() == 3.0f);
        CHECK(sc.imag() == 4.0f);
    }
}

TEST_CASE("Complex number literals") {
    SUBCASE("Float imaginary literals") {
        auto z1 = 2.0_i;
        CHECK(z1.real() == 0.0f);
        CHECK(z1.imag() == 2.0f);
        
        auto z2 = 3.5_if;
        CHECK(z2.real() == 0.0f);
        CHECK(z2.imag() == 3.5f);
    }
    
    SUBCASE("Double imaginary literals") {
        auto z = 2.5_id;
        CHECK(z.real() == 0.0);
        CHECK(z.imag() == 2.5);
    }
    
    SUBCASE("Integer imaginary literals") {
        auto z = 5_i;
        CHECK(z.real() == 0.0f);
        CHECK(z.imag() == 5.0f);
    }
    
    SUBCASE("Complex expressions with literals") {
        auto z = 3.0f + 4.0_i;
        CHECK(z.real() == 3.0f);
        CHECK(z.imag() == 4.0f);
    }
}

TEST_CASE("Complex polar form") {
    SUBCASE("Polar construction with degrees") {
        auto z = complex<float>::polar(2.0f, 90.0_deg);
        CHECK(approx_equal(z.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(z.imag(), 2.0f, 1e-6f));
    }
    
    SUBCASE("Polar construction with radians") {
        auto z = complex<float>::polar(2.0f, radian<float>(constants<float>::pi / 2));
        CHECK(approx_equal(z.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(z.imag(), 2.0f, 1e-6f));
    }
    
    SUBCASE("Polar construction with raw radians") {
        auto z = complex<float>::polar(5.0f, constants<float>::pi / 4);
        float expected_real = 5.0f * std::cos(constants<float>::pi / 4);
        float expected_imag = 5.0f * std::sin(constants<float>::pi / 4);
        CHECK(approx_equal(z.real(), expected_real, 1e-6f));
        CHECK(approx_equal(z.imag(), expected_imag, 1e-6f));
    }
    
    SUBCASE("Polar helper functions") {
        auto z1 = polar(3.0f, 45.0_deg);
        auto z2 = polar(3.0f, constants<float>::pi / 4);
        CHECK(approx_equal(z1, z2));
    }
}

TEST_CASE("Complex magnitude and argument") {
    SUBCASE("Magnitude (abs)") {
        complexf z(3.0f, 4.0f);
        CHECK(approx_equal(z.abs(), 5.0f));
        CHECK(approx_equal(abs(z), 5.0f));
    }
    
    SUBCASE("Norm (squared magnitude)") {
        complexf z(3.0f, 4.0f);
        CHECK(approx_equal(z.norm(), 25.0f));
        CHECK(approx_equal(norm(z), 25.0f));
    }
    
    SUBCASE("Argument in radians") {
        complexf z(1.0f, 1.0f);
        radian<float> angle = z.arg();
        CHECK(approx_equal(angle.value(), constants<float>::pi / 4, 1e-6f));
        
        // Using free function
        radian<float> angle2 = arg(z);
        CHECK(approx_equal(angle2, angle));
    }
    
    SUBCASE("Argument in degrees") {
        complexf z(1.0f, 1.0f);
        degree<float> angle = z.arg_deg();
        CHECK(approx_equal(angle.value(), 45.0f, 1e-5f));
    }
    
    SUBCASE("Argument edge cases") {
        CHECK(approx_equal(complexf(1.0f, 0.0f).arg().value(), 0.0f));
        CHECK(approx_equal(complexf(0.0f, 1.0f).arg().value(), constants<float>::pi / 2));
        CHECK(approx_equal(complexf(-1.0f, 0.0f).arg().value(), constants<float>::pi));
        CHECK(approx_equal(complexf(0.0f, -1.0f).arg().value(), -constants<float>::pi / 2));
    }
}

TEST_CASE("Complex arithmetic operations") {
    complexf z1(3.0f, 4.0f);
    complexf z2(1.0f, 2.0f);
    
    SUBCASE("Addition") {
        auto sum = z1 + z2;
        CHECK(approx_equal(sum.real(), 4.0f));
        CHECK(approx_equal(sum.imag(), 6.0f));
        
        // In-place
        complexf z3 = z1;
        z3 += z2;
        CHECK(approx_equal(z3, sum));
    }
    
    SUBCASE("Subtraction") {
        auto diff = z1 - z2;
        CHECK(approx_equal(diff.real(), 2.0f));
        CHECK(approx_equal(diff.imag(), 2.0f));
        
        // In-place
        complexf z3 = z1;
        z3 -= z2;
        CHECK(approx_equal(z3, diff));
    }
    
    SUBCASE("Multiplication") {
        // (3+4i) * (1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        auto prod = z1 * z2;
        CHECK(approx_equal(prod.real(), -5.0f));
        CHECK(approx_equal(prod.imag(), 10.0f));
        
        // In-place
        complexf z3 = z1;
        z3 *= z2;
        CHECK(approx_equal(z3, prod));
    }
    
    SUBCASE("Division") {
        // (3+4i) / (1+2i) = (3+4i)(1-2i) / (1+4) = (3-6i+4i+8) / 5 = (11-2i) / 5
        auto quot = z1 / z2;
        CHECK(approx_equal(quot.real(), 11.0f / 5.0f));
        CHECK(approx_equal(quot.imag(), -2.0f / 5.0f));
        
        // In-place
        complexf z3 = z1;
        z3 /= z2;
        CHECK(approx_equal(z3, quot));
    }
    
    SUBCASE("Unary operators") {
        CHECK(approx_equal(+z1, z1));
        CHECK(approx_equal((-z1).real(), -3.0f));
        CHECK(approx_equal((-z1).imag(), -4.0f));
    }
}

TEST_CASE("Complex-scalar operations") {
    complexf z(3.0f, 4.0f);
    float s = 2.0f;
    
    SUBCASE("Complex + scalar") {
        auto sum = z + s;
        CHECK(approx_equal(sum.real(), 5.0f));
        CHECK(approx_equal(sum.imag(), 4.0f));
        
        // Scalar + complex
        auto sum2 = s + z;
        CHECK(approx_equal(sum2, sum));
    }
    
    SUBCASE("Complex - scalar") {
        auto diff = z - s;
        CHECK(approx_equal(diff.real(), 1.0f));
        CHECK(approx_equal(diff.imag(), 4.0f));
        
        // Scalar - complex
        auto diff2 = s - z;
        CHECK(approx_equal(diff2.real(), -1.0f));
        CHECK(approx_equal(diff2.imag(), -4.0f));
    }
    
    SUBCASE("Complex * scalar") {
        auto prod = z * s;
        CHECK(approx_equal(prod.real(), 6.0f));
        CHECK(approx_equal(prod.imag(), 8.0f));
        
        // Scalar * complex
        auto prod2 = s * z;
        CHECK(approx_equal(prod2, prod));
        
        // In-place
        complexf z2 = z;
        z2 *= s;
        CHECK(approx_equal(z2, prod));
    }
    
    SUBCASE("Complex / scalar") {
        auto quot = z / s;
        CHECK(approx_equal(quot.real(), 1.5f));
        CHECK(approx_equal(quot.imag(), 2.0f));
        
        // Scalar / complex
        // 2 / (3+4i) = 2(3-4i) / 25 = (6-8i) / 25
        auto quot2 = s / z;
        CHECK(approx_equal(quot2.real(), 6.0f / 25.0f));
        CHECK(approx_equal(quot2.imag(), -8.0f / 25.0f));
        
        // In-place
        complexf z2 = z;
        z2 /= s;
        CHECK(approx_equal(z2, quot));
    }
}

TEST_CASE("Complex conjugate") {
    complexf z(3.0f, 4.0f);
    
    auto conj_z = conj(z);
    CHECK(approx_equal(conj_z.real(), 3.0f));
    CHECK(approx_equal(conj_z.imag(), -4.0f));
    
    // Conjugate of conjugate
    CHECK(approx_equal(conj(conj_z), z));
    
    // Properties
    complexf w(1.0f, 2.0f);
    CHECK(approx_equal(conj(z + w), conj(z) + conj(w)));
    CHECK(approx_equal(conj(z * w), conj(z) * conj(w)));
}

TEST_CASE("Complex approx_equal") {
    SUBCASE("Complex-complex comparison") {
        complexf z1(3.0f, 4.0f);
        complexf z2(3.0f + 1e-7f, 4.0f - 1e-7f);
        CHECK(approx_equal(z1, z2));
        
        complexf z3(3.1f, 4.0f);
        CHECK_FALSE(approx_equal(z1, z3, 0.01f));
    }
    
    SUBCASE("Complex-scalar comparison") {
        complexf z1(3.0f, 0.0f);
        float s = 3.0f;
        CHECK(approx_equal(z1, s));
        
        complexf z2(3.0f, 0.1f);
        CHECK_FALSE(approx_equal(z2, s, 0.01f));
    }
    
    SUBCASE("Scalar-complex comparison") {
        float s = 2.0f;
        complexf z1(2.0f, 0.0f);
        CHECK(approx_equal(s, z1));
        
        complexf z2(2.0f, 0.1f);
        CHECK_FALSE(approx_equal(s, z2, 0.01f));
    }
}

TEST_CASE("Complex real and imaginary part extraction") {
    complexf z(3.0f, 4.0f);
    
    CHECK(real(z) == 3.0f);
    CHECK(imag(z) == 4.0f);
    
    // Mutable access
    z.real() = 5.0f;
    z.imag() = 6.0f;
    CHECK(z.real() == 5.0f);
    CHECK(z.imag() == 6.0f);
}

TEST_CASE("Complex comparison operators") {
    complexf z1(3.0f, 4.0f);
    complexf z2(3.0f, 4.0f);
    complexf z3(3.0f, 5.0f);
    complexf z4(4.0f, 4.0f);
    
    CHECK(z1 == z2);
    CHECK(z1 != z3);
    CHECK(z1 != z4);
    CHECK_FALSE(z1 == z3);
    CHECK_FALSE(z1 == z4);
}

TEST_CASE("Complex special values") {
    SUBCASE("Zero") {
        complexf z(0.0f, 0.0f);
        CHECK(z.abs() == 0.0f);
        CHECK(z.norm() == 0.0f);
    }
    
    SUBCASE("Pure real") {
        complexf z(3.0f, 0.0f);
        CHECK(z.abs() == 3.0f);
        CHECK(z.arg().value() == 0.0f);
    }
    
    SUBCASE("Pure imaginary") {
        complexf z(0.0f, 4.0f);
        CHECK(z.abs() == 4.0f);
        CHECK(approx_equal(z.arg().value(), constants<float>::pi / 2));
    }
}

TEST_CASE("Complex type aliases") {
    SUBCASE("Float complex") {
        complexf z(1.0f, 2.0f);
        static_assert(std::is_same_v<decltype(z)::value_type, float>);
    }
    
    SUBCASE("Double complex") {
        complexd z(1.0, 2.0);
        static_assert(std::is_same_v<decltype(z)::value_type, double>);
    }
}

TEST_CASE("Complex trigonometric integration") {
    SUBCASE("Polar form round-trip") {
        complexf z(3.0f, 4.0f);
        
        // Convert to polar and back
        float mag = z.abs();
        radian<float> phase = z.arg();
        complexf z2 = polar(mag, phase);
        
        CHECK(approx_equal(z, z2, 1e-6f));
    }
    
    SUBCASE("Using angles in complex operations") {
        // Rotation by 90 degrees
        complexf z(1.0f, 0.0f);
        complexf rotation = polar(1.0f, 90.0_deg);
        complexf rotated = z * rotation;
        
        CHECK(approx_equal(rotated.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(rotated.imag(), 1.0f, 1e-6f));
    }
    
    SUBCASE("Multiple rotations") {
        complexf z(1.0f, 0.0f);
        
        // Rotate by 30 degrees three times = 90 degrees total
        complexf rot30 = polar(1.0f, 30.0_deg);
        complexf result = z * rot30 * rot30 * rot30;
        
        CHECK(approx_equal(result.real(), 0.0f, 1e-6f));
        CHECK(approx_equal(result.imag(), 1.0f, 1e-6f));
    }
}