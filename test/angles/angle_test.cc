#include <doctest/doctest.h>
#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_traits.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/core/types.hh>
#include <cmath>
#include <vector>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Angle construction and basic operations") {
    SUBCASE("Default construction") {
        degree<float> d;
        CHECK(d.value() == 0.0f);
        
        radian<float> r;
        CHECK(r.value() == 0.0f);
    }
    
    SUBCASE("Value construction") {
        degree<float> d(45.0f);
        CHECK(d.value() == 45.0f);
        
        radian<float> r(1.57f);
        CHECK(r.value() == 1.57f);
    }
    
    SUBCASE("Literal operators") {
        auto d1 = 45.0_deg;
        CHECK(d1.value() == 45.0f);
        CHECK((std::is_same_v<decltype(d1), degreef>));
        
        auto d2 = 90_deg;
        CHECK(d2.value() == 90.0f);
        
        auto r1 = 1.57_rad;
        CHECK(r1.value() == 1.57f);
        CHECK((std::is_same_v<decltype(r1), radianf>));
        
        auto r2 = 3_rad;
        CHECK(r2.value() == 3.0f);
    }
    
    SUBCASE("Copy and assignment") {
        degree<float> d1(45.0f);
        degree<float> d2(d1);
        CHECK(d2.value() == 45.0f);
        
        degree<float> d3;
        d3 = d1;
        CHECK(d3.value() == 45.0f);
    }
}

TEST_CASE("Angle conversions") {
    SUBCASE("Degree to radian conversion") {
        degree<float> d(180.0f);
        radian<float> r(d);
        CHECK(std::abs(r.value() - constants<float>::pi) < 1e-6f);
        
        auto r2 = to_radians(90.0_deg);
        CHECK(std::abs(r2.value() - constants<float>::half_pi) < 1e-6f);
    }
    
    SUBCASE("Radian to degree conversion") {
        radian<float> r(constants<float>::pi);
        degree<float> d(r);
        CHECK(std::abs(d.value() - 180.0f) < 1e-5f);
        
        auto d2 = to_degrees(constants<float>::half_pi * 1.0_rad);
        CHECK(std::abs(d2.value() - 90.0f) < 1e-5f);
    }
    
    SUBCASE("Round trip conversions") {
        degree<float> d1(45.0f);
        auto r = to_radians(d1);
        auto d2 = to_degrees(r);
        CHECK(std::abs(d1.value() - d2.value()) < 1e-5f);
        
        radian<float> r1(1.234f);
        auto d = to_degrees(r1);
        auto r2 = to_radians(d);
        CHECK(std::abs(r1.value() - r2.value()) < 1e-6f);
    }
}

TEST_CASE("Angle arithmetic") {
    SUBCASE("Addition") {
        auto a = 30.0_deg;
        auto b = 45.0_deg;
        auto c = a + b;
        CHECK(c.value() == 75.0f);
        
        auto r1 = 1.0_rad;
        auto r2 = 0.5_rad;
        auto r3 = r1 + r2;
        CHECK(r3.value() == 1.5f);
    }
    
    SUBCASE("Subtraction") {
        auto a = 90.0_deg;
        auto b = 30.0_deg;
        auto c = a - b;
        CHECK(c.value() == 60.0f);
        
        auto r1 = 2.0_rad;
        auto r2 = 0.5_rad;
        auto r3 = r1 - r2;
        CHECK(r3.value() == 1.5f);
    }
    
    SUBCASE("Multiplication by scalar") {
        auto a = 45.0_deg;
        auto b = a * 2.0f;
        CHECK(b.value() == 90.0f);
        
        auto c = 3.0f * a;
        CHECK(c.value() == 135.0f);
        
        auto r = 1.0_rad;
        auto r2 = r * 3.14f;
        CHECK(r2.value() == 3.14f);
    }
    
    SUBCASE("Division by scalar") {
        auto a = 90.0_deg;
        auto b = a / 2.0f;
        CHECK(b.value() == 45.0f);
        
        auto r = 3.14_rad;
        auto r2 = r / 2.0f;
        CHECK(r2.value() == 1.57f);
    }
    
    SUBCASE("Division of angles") {
        auto a = 90.0_deg;
        auto b = 45.0_deg;
        float ratio = a / b;
        CHECK(ratio == 2.0f);
        
        auto r1 = 3.14_rad;
        auto r2 = 1.57_rad;
        float ratio2 = r1 / r2;
        CHECK(std::abs(ratio2 - 2.0f) < 1e-5f);
    }
    
    SUBCASE("Compound assignment") {
        auto a = 30.0_deg;
        a += 15.0_deg;
        CHECK(a.value() == 45.0f);
        
        a -= 20.0_deg;
        CHECK(a.value() == 25.0f);
        
        a *= 2.0f;
        CHECK(a.value() == 50.0f);
        
        a /= 5.0f;
        CHECK(a.value() == 10.0f);
    }
    
    SUBCASE("Unary operators") {
        auto a = 45.0_deg;
        auto b = -a;
        CHECK(b.value() == -45.0f);
        
        auto c = +a;
        CHECK(c.value() == 45.0f);
    }
}

TEST_CASE("Angle comparisons") {
    SUBCASE("Equality") {
        auto a = 45.0_deg;
        auto b = 45.0_deg;
        auto c = 46.0_deg;
        
        CHECK(a == b);
        CHECK(!(a == c));
        CHECK(a != c);
        CHECK(!(a != b));
    }
    
    SUBCASE("Ordering") {
        auto a = 30.0_deg;
        auto b = 45.0_deg;
        auto c = 45.0_deg;
        
        CHECK(a < b);
        CHECK(!(b < a));
        CHECK(!(b < c));
        
        CHECK(a <= b);
        CHECK(b <= c);
        CHECK(!(b <= a));
        
        CHECK(b > a);
        CHECK(!(a > b));
        CHECK(!(b > c));
        
        CHECK(b >= a);
        CHECK(b >= c);
        CHECK(!(a >= b));
    }
}

TEST_CASE("Angle increment/decrement operators") {
    SUBCASE("Pre-increment") {
        auto angle = 45.0_deg;
        auto& result = ++angle;
        CHECK(angle.value() == 46.0f);
        CHECK(result.value() == 46.0f);
        CHECK(&result == &angle);  // Should return reference to self
        
        auto rad = 1.0_rad;
        ++rad;
        CHECK(rad.value() == 2.0f);
    }
    
    SUBCASE("Post-increment") {
        auto angle = 45.0_deg;
        auto result = angle++;
        CHECK(angle.value() == 46.0f);
        CHECK(result.value() == 45.0f);  // Should return old value
        
        auto rad = 1.0_rad;
        auto old_rad = rad++;
        CHECK(rad.value() == 2.0f);
        CHECK(old_rad.value() == 1.0f);
    }
    
    SUBCASE("Pre-decrement") {
        auto angle = 45.0_deg;
        auto& result = --angle;
        CHECK(angle.value() == 44.0f);
        CHECK(result.value() == 44.0f);
        CHECK(&result == &angle);  // Should return reference to self
        
        auto rad = 2.0_rad;
        --rad;
        CHECK(rad.value() == 1.0f);
    }
    
    SUBCASE("Post-decrement") {
        auto angle = 45.0_deg;
        auto result = angle--;
        CHECK(angle.value() == 44.0f);
        CHECK(result.value() == 45.0f);  // Should return old value
        
        auto rad = 2.0_rad;
        auto old_rad = rad--;
        CHECK(rad.value() == 1.0f);
        CHECK(old_rad.value() == 2.0f);
    }
    
    SUBCASE("Usage in loops") {
        // For loop with degrees
        int count = 0;
        for (auto angle = 0.0_deg; angle < 360.0_deg; angle += 90.0_deg) {
            count++;
        }
        CHECK(count == 4);
        
        // While loop with increment
        auto angle = 0.0_deg;
        count = 0;
        while (angle < 10.0_deg) {
            count++;
            angle++;
        }
        CHECK(count == 10);
        
        // Range iteration (manual)
        std::vector<float> values;
        for (auto a = 0.0_deg; a <= 180.0_deg; a += 45.0_deg) {
            values.push_back(a.value());
        }
        CHECK(values.size() == 5);
        CHECK(values[0] == 0.0f);
        CHECK(values[1] == 45.0f);
        CHECK(values[2] == 90.0f);
        CHECK(values[3] == 135.0f);
        CHECK(values[4] == 180.0f);
    }
}

TEST_CASE("Angle wrapping") {
    SUBCASE("Wrap degrees to [-180, 180]") {
        CHECK(wrap(degree<float>(0.0f)).value() == 0.0f);
        CHECK(wrap(degree<float>(90.0f)).value() == 90.0f);
        CHECK(wrap(degree<float>(180.0f)).value() == 180.0f);
        CHECK(wrap(degree<float>(181.0f)).value() == -179.0f);
        CHECK(wrap(degree<float>(270.0f)).value() == -90.0f);
        CHECK(wrap(degree<float>(360.0f)).value() == 0.0f);
        CHECK(wrap(degree<float>(450.0f)).value() == 90.0f);
        CHECK(wrap(degree<float>(-90.0f)).value() == -90.0f);
        CHECK(wrap(degree<float>(-181.0f)).value() == 179.0f);
        CHECK(wrap(degree<float>(-270.0f)).value() == 90.0f);
        CHECK(wrap(degree<float>(-360.0f)).value() == 0.0f);
    }
    
    SUBCASE("Wrap degrees to [0, 360)") {
        CHECK(wrap_positive(degree<float>(0.0f)).value() == 0.0f);
        CHECK(wrap_positive(degree<float>(90.0f)).value() == 90.0f);
        CHECK(wrap_positive(degree<float>(180.0f)).value() == 180.0f);
        CHECK(wrap_positive(degree<float>(270.0f)).value() == 270.0f);
        CHECK(wrap_positive(degree<float>(360.0f)).value() == 0.0f);
        CHECK(wrap_positive(degree<float>(450.0f)).value() == 90.0f);
        CHECK(wrap_positive(degree<float>(-90.0f)).value() == 270.0f);
        CHECK(wrap_positive(degree<float>(-180.0f)).value() == 180.0f);
        CHECK(wrap_positive(degree<float>(-270.0f)).value() == 90.0f);
        CHECK(wrap_positive(degree<float>(-360.0f)).value() == 0.0f);
    }
    
    SUBCASE("Wrap radians to [-π, π]") {
        const float pi = constants<float>::pi;
        CHECK(std::abs(wrap(radian<float>(0.0f)).value() - 0.0f) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(pi/2)).value() - pi/2) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(pi)).value() - pi) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(1.1f * pi)).value() + 0.9f * pi) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(1.5f * pi)).value() + 0.5f * pi) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(2.0f * pi)).value() - 0.0f) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(-pi/2)).value() + pi/2) < 1e-6f);
        CHECK(std::abs(wrap(radian<float>(-1.1f * pi)).value() - 0.9f * pi) < 1e-6f);
    }
    
    SUBCASE("Wrap radians to [0, 2π)") {
        const float pi = constants<float>::pi;
        CHECK(std::abs(wrap_positive(radian<float>(0.0f)).value() - 0.0f) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(pi/2)).value() - pi/2) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(pi)).value() - pi) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(1.5f * pi)).value() - 1.5f * pi) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(2.0f * pi)).value() - 0.0f) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(-pi/2)).value() - 1.5f * pi) < 1e-6f);
        CHECK(std::abs(wrap_positive(radian<float>(-pi)).value() - pi) < 1e-6f);
    }
}

TEST_CASE("Angle interpolation") {
    SUBCASE("Degree interpolation") {
        auto a = 0.0_deg;
        auto b = 90.0_deg;
        
        CHECK(lerp(a, b, 0.0f).value() == 0.0f);
        CHECK(lerp(a, b, 0.5f).value() == 45.0f);
        CHECK(lerp(a, b, 1.0f).value() == 90.0f);
        
        // Test shortest path
        auto c = 350.0_deg;
        auto d = 10.0_deg;
        auto mid = lerp(c, d, 0.5f);
        bool is_near_zero = std::abs(mid.value() - 0.0f) < 1e-5f;
        bool is_near_360 = std::abs(mid.value() - 360.0f) < 1e-5f;
        CHECK((is_near_zero || is_near_360));
    }
    
    SUBCASE("Radian interpolation") {
        const float pi = constants<float>::pi;
        auto a = radian<float>(0.0f);
        auto b = radian<float>(pi/2);
        
        CHECK(std::abs(lerp(a, b, 0.0f).value() - 0.0f) < 1e-6f);
        CHECK(std::abs(lerp(a, b, 0.5f).value() - pi/4) < 1e-6f);
        CHECK(std::abs(lerp(a, b, 1.0f).value() - pi/2) < 1e-6f);
    }
}

TEST_CASE("Angle utilities") {
    SUBCASE("Absolute value") {
        CHECK(abs(45.0_deg).value() == 45.0f);
        CHECK(abs(-45.0_deg).value() == 45.0f);
        CHECK(abs(radian<float>(-1.57f)).value() == 1.57f);
    }
    
    SUBCASE("Min/max") {
        auto a = 30.0_deg;
        auto b = 45.0_deg;
        
        CHECK(min(a, b).value() == 30.0f);
        CHECK(max(a, b).value() == 45.0f);
    }
    
    SUBCASE("Clamp") {
        auto val = 50.0_deg;
        auto low = 30.0_deg;
        auto high = 45.0_deg;
        
        CHECK(clamp(val, low, high).value() == 45.0f);
        CHECK(clamp(20.0_deg, low, high).value() == 30.0f);
        CHECK(clamp(35.0_deg, low, high).value() == 35.0f);
    }
    
    SUBCASE("Angle difference") {
        auto a = 30.0_deg;
        auto b = 60.0_deg;
        CHECK(angle_difference(a, b).value() == 30.0f);
        
        auto c = 350.0_deg;
        auto d = 10.0_deg;
        CHECK(angle_difference(c, d).value() == 20.0f);
    }
    
    SUBCASE("Angle components") {
        auto components = angle_to_components(90.0_deg);
        CHECK(std::abs(components.cos - 0.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 1.0f) < 1e-6f);
        
        components = angle_to_components(radian<float>(constants<float>::pi));
        CHECK(std::abs(components.cos + 1.0f) < 1e-6f);
        CHECK(std::abs(components.sin - 0.0f) < 1e-6f);
    }
    
    SUBCASE("Angle from components") {
        auto angle = angle_from_components(1.0f, 0.0f);
        CHECK(std::abs(angle.value() - constants<float>::half_pi) < 1e-6f);
        
        auto angle_deg = angle_from_components<degree<float>>(0.0f, -1.0f);
        CHECK(std::abs(angle_deg.value() - 180.0f) < 1e-5f);
    }
}

TEST_CASE("Angle traits") {
    SUBCASE("Type detection") {
        CHECK(is_angle_v<degree<float>>);
        CHECK(is_angle_v<radian<double>>);
        CHECK(!is_angle_v<float>);
        CHECK(!is_angle_v<int>);
        
        CHECK(is_degree_v<degree<float>>);
        CHECK(!is_degree_v<radian<float>>);
        CHECK(!is_degree_v<float>);
        
        CHECK(is_radian_v<radian<float>>);
        CHECK(!is_radian_v<degree<float>>);
        CHECK(!is_radian_v<float>);
    }
    
    SUBCASE("Value type extraction") {
        CHECK((std::is_same_v<angle_value_type_t<degree<float>>, float>));
        CHECK((std::is_same_v<angle_value_type_t<radian<double>>, double>));
        CHECK((std::is_same_v<angle_value_type_t<int>, int>));
    }
    
    SUBCASE("Unit type extraction") {
        CHECK((std::is_same_v<angle_unit_type_t<degree<float>>, degree_tag>));
        CHECK((std::is_same_v<angle_unit_type_t<radian<float>>, radian_tag>));
        CHECK((std::is_same_v<angle_unit_type_t<float>, void>));
    }
}

TEST_CASE("Type safety") {
    SUBCASE("Cannot implicitly mix degrees and radians") {
        degree<float> d(45.0f);
        radian<float> r(1.0f);
        
        // These should not compile (uncomment to test):
        // auto bad1 = d + r;  // Error: no matching operator+
        // auto bad2 = d - r;  // Error: no matching operator-
        // bool bad3 = d < r;  // Error: no matching operator<
        
        // Must explicitly convert
        auto good1 = d + degree<float>(r);  // OK
        auto good2 = radian<float>(d) + r;  // OK
        CHECK(good1.value() > 0.0f);
        CHECK(good2.value() > 0.0f);
    }
}

TEST_CASE("Performance characteristics") {
    SUBCASE("Size is same as underlying type") {
        CHECK(sizeof(degree<float>) == sizeof(float));
        CHECK(sizeof(radian<double>) == sizeof(double));
    }
    
    SUBCASE("Alignment is same as underlying type") {
        CHECK(alignof(degree<float>) == alignof(float));
        CHECK(alignof(radian<double>) == alignof(double));
    }
}