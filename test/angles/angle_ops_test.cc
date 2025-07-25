#include <doctest/doctest.h>
#include <euler/angles/angle_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/core/types.hh>
#include <euler/core/approx_equal.hh>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Angle operations - conversions") {
    SUBCASE("Explicit conversion functions") {
        auto d = 180.0_deg;
        auto r = to_radians(d);
        CHECK(std::abs(r.value() - constants<float>::pi) < 1e-6f);
        
        auto r2 = constants<float>::pi * 1.0_rad;
        auto d2 = to_degrees(r2);
        CHECK(std::abs(d2.value() - 180.0f) < 1e-5f);
    }
    
    SUBCASE("Identity conversions") {
        auto d = 45.0_deg;
        auto d2 = to_degrees(d);
        CHECK(d.value() == d2.value());
        
        auto r = 1.57_rad;
        auto r2 = to_radians(r);
        CHECK(r.value() == r2.value());
    }
}

TEST_CASE("Angle operations - mathematical functions") {
    SUBCASE("Absolute value") {
        CHECK(abs(45.0_deg) == 45.0_deg);
        CHECK(abs(-45.0_deg) == 45.0_deg);
        CHECK(abs(0.0_deg) == 0.0_deg);
        
        CHECK(abs(1.0_rad) == 1.0_rad);
        CHECK(abs(-1.0_rad) == 1.0_rad);
    }
    
    SUBCASE("Min/max functions") {
        auto a = 30.0_deg;
        auto b = 45.0_deg;
        auto c = 15.0_deg;
        
        CHECK(min(a, b) == a);
        CHECK(min(b, a) == a);
        CHECK(max(a, b) == b);
        CHECK(max(b, a) == b);
        
        CHECK(min(min(a, b), c) == c);
        CHECK(max(max(a, b), c) == b);
    }
    
    SUBCASE("Clamp function") {
        auto low = 0.0_deg;
        auto high = 90.0_deg;
        
        CHECK(clamp(45.0_deg, low, high) == 45.0_deg);
        CHECK(clamp(-10.0_deg, low, high) == 0.0_deg);
        CHECK(clamp(100.0_deg, low, high) == 90.0_deg);
        
        auto r_low = 0.0_rad;
        auto r_high = constants<float>::half_pi * 1.0_rad;
        
        CHECK(clamp(1.0_rad, r_low, r_high) == 1.0_rad);
        CHECK(clamp(-0.5_rad, r_low, r_high) == 0.0_rad);
        CHECK(clamp(2.0_rad, r_low, r_high) == r_high);
    }
}

TEST_CASE("Angle operations - factory functions") {
    SUBCASE("Degrees factory") {
        auto d1 = degrees(45.0f);
        CHECK(d1.value() == 45.0f);
        CHECK((std::is_same_v<decltype(d1), degree<float>>));
        
        auto d2 = degrees(90.0);
        CHECK(d2.value() == 90.0);
        CHECK((std::is_same_v<decltype(d2), degree<double>>));
        
        auto d3 = degrees<double>(180.0f);
        CHECK(d3.value() == 180.0);
        CHECK((std::is_same_v<decltype(d3), degree<double>>));
    }
    
    SUBCASE("Radians factory") {
        auto r1 = radians(1.57f);
        CHECK(r1.value() == 1.57f);
        CHECK((std::is_same_v<decltype(r1), radian<float>>));
        
        auto r2 = radians(3.14);
        CHECK(r2.value() == 3.14);
        CHECK((std::is_same_v<decltype(r2), radian<double>>));
        
        auto r3 = radians<double>(1.0f);
        CHECK(r3.value() == 1.0);
        CHECK((std::is_same_v<decltype(r3), radian<double>>));
    }
}

TEST_CASE("Angle operations - difference and comparison") {
    SUBCASE("Angle difference with wrapping") {
        auto a1 = 30.0_deg;
        auto a2 = 60.0_deg;
        CHECK(angle_difference(a1, a2).value() == 30.0f);
        CHECK(angle_difference(a2, a1).value() == -30.0f);
        
        // Test wrap-around
        auto a3 = 10.0_deg;
        auto a4 = 350.0_deg;
        CHECK(angle_difference(a4, a3).value() == 20.0f);
        CHECK(angle_difference(a3, a4).value() == -20.0f);
        
        // Test 180 degree difference
        auto a5 = 0.0_deg;
        auto a6 = 180.0_deg;
        CHECK(angle_difference(a5, a6).value() == 180.0f);
    }
    
    SUBCASE("Approximate equality") {
        auto a = 45.0_deg;
        auto b = 45.00001_deg;
        auto c = 46.0_deg;
        
        CHECK(approx_equal(a, b, 0.001f));
        CHECK(!approx_equal(a, c, 0.1f));
        
        auto r1 = 1.0_rad;
        auto r2 = 1.0000001_rad;
        CHECK(approx_equal(r1, r2));
        CHECK(approx_equal(r1, r2, 1e-6f));
        CHECK(!approx_equal(r1, r2, 1e-8f));
    }
}

TEST_CASE("Angle operations - components") {
    SUBCASE("Convert angle to sin/cos components") {
        auto components = angle_to_components(0.0_deg);
        CHECK(std::abs(components.cos - 1.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 0.0f) < 1e-6f);
        
        components = angle_to_components(90.0_deg);
        CHECK(std::abs(components.cos - 0.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 1.0f) < 1e-6f);
        
        components = angle_to_components(180.0_deg);
        CHECK(std::abs(components.cos + 1.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 0.0f) < 1e-6f);
        
        components = angle_to_components(270.0_deg);
        CHECK(std::abs(components.cos - 0.0f) < 1e-6f);
        CHECK(std::abs(components.sin + 1.0f) < 1e-6f);
        
        // Test with radians
        components = angle_to_components(constants<float>::half_pi * 1.0_rad);
        CHECK(std::abs(components.cos - 0.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 1.0f) < 1e-6f);
    }
    
    SUBCASE("Create angle from components (atan2)") {
        // Test cardinal directions
        auto angle = angle_from_components(0.0f, 1.0f);  // East
        CHECK(std::abs(angle.value() - 0.0f) < 1e-6f);
        
        angle = angle_from_components(1.0f, 0.0f);  // North
        CHECK(std::abs(angle.value() - constants<float>::half_pi) < 1e-6f);
        
        angle = angle_from_components(0.0f, -1.0f);  // West
        CHECK(std::abs(angle.value() - constants<float>::pi) < 1e-6f);
        
        angle = angle_from_components(-1.0f, 0.0f);  // South
        CHECK(std::abs(angle.value() + constants<float>::half_pi) < 1e-6f);
        
        // Test in degrees
        auto angle_deg = angle_from_components<degree<float>>(1.0f, 1.0f);  // Northeast
        CHECK(std::abs(angle_deg.value() - 45.0f) < 1e-5f);
        
        angle_deg = angle_from_components<degree<float>>(-1.0f, 1.0f);  // Southeast
        CHECK(std::abs(angle_deg.value() + 45.0f) < 1e-5f);
    }
}

TEST_CASE("Angle operations - modulation and sign") {
    SUBCASE("Angle modulation") {
        auto a = 270.0_deg;
        auto modulus = 90.0_deg;
        auto result = mod_angle(a, modulus);
        CHECK(result.value() == 0.0f);
        
        a = 100.0_deg;
        result = mod_angle(a, modulus);
        CHECK(result.value() == 10.0f);
        
        auto r = 5.0_rad;
        auto r_mod = constants<float>::pi * 1.0_rad;
        auto r_result = mod_angle(r, r_mod);
        CHECK(std::abs(r_result.value() - (5.0f - constants<float>::pi)) < 1e-6f);
    }
    
    SUBCASE("Sign function") {
        CHECK(sign(45.0_deg) == 1);
        CHECK(sign(-45.0_deg) == -1);
        CHECK(sign(0.0_deg) == 0);
        
        CHECK(sign(1.0_rad) == 1);
        CHECK(sign(-1.0_rad) == -1);
        CHECK(sign(0.0_rad) == 0);
    }
}

TEST_CASE("Angle constants") {
    using namespace angle_constants;
    
    SUBCASE("Radian constants") {
        CHECK(std::abs(pi_rad<float>.value() - constants<float>::pi) < 1e-6f);
        CHECK(std::abs(half_pi_rad<float>.value() - constants<float>::half_pi) < 1e-6f);
        CHECK(std::abs(two_pi_rad<float>.value() - 2.0f * constants<float>::pi) < 1e-6f);
        CHECK(std::abs(quarter_pi_rad<float>.value() - constants<float>::pi / 4.0f) < 1e-6f);
        
        // Test double precision
        CHECK(std::abs(pi_rad<double>.value() - constants<double>::pi) < 1e-15);
    }
    
    SUBCASE("Constants can be used in expressions") {
        auto angle = pi_rad<float> / 2.0f;
        CHECK(std::abs(angle.value() - constants<float>::half_pi) < 1e-6f);
        
        auto sum = half_pi_rad<float> + quarter_pi_rad<float>;
        CHECK(std::abs(sum.value() - 0.75f * constants<float>::pi) < 1e-6f);
    }
}