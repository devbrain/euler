#include <euler/math/trigonometry.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Trigonometric functions") {
    SUBCASE("sin function") {
        SUBCASE("scalar values") {
            // Test common angles in radians
            CHECK(approx_equal(sin(0.0f), 0.0f));
            CHECK(approx_equal(sin(constants<float>::pi / 2), 1.0f));
            CHECK(approx_equal(sin(constants<float>::pi), 0.0f));
            CHECK(approx_equal(sin(3 * constants<float>::pi / 2), -1.0f));
            CHECK(approx_equal(sin(2 * constants<float>::pi), 0.0f, 1e-6f));
            
            // Test with angle types
            CHECK(approx_equal(sin(0.0_deg), 0.0f));
            CHECK(approx_equal(sin(90.0_deg), 1.0f));
            CHECK(approx_equal(sin(180.0_deg), 0.0f, 1e-6f));
            CHECK(approx_equal(sin(270.0_deg), -1.0f));
            CHECK(approx_equal(sin(360.0_deg), 0.0f, 1e-6f));
            
            CHECK(approx_equal(sin(0.0_rad), 0.0f));
            CHECK(approx_equal(sin(constants<float>::pi / 2 * 1.0_rad), 1.0f));
            CHECK(approx_equal(sin(constants<float>::pi * 1.0_rad), 0.0f));
        }
        
        SUBCASE("vector values") {
            vec3f v(0.0f, constants<float>::pi / 2, constants<float>::pi);
            vec3f result = sin(v);
            CHECK(approx_equal(result[0], 0.0f));
            CHECK(approx_equal(result[1], 1.0f));
            CHECK(approx_equal(result[2], 0.0f));
            
            // Vector of angles
            vector<degree<float>, 3> angles;
            angles[0] = 0.0_deg;
            angles[1] = 90.0_deg;
            angles[2] = 180.0_deg;
            vec3f angle_result = sin(angles);
            CHECK(approx_equal(angle_result[0], 0.0f));
            CHECK(approx_equal(angle_result[1], 1.0f));
            CHECK(approx_equal(angle_result[2], 0.0f));
        }
        
        SUBCASE("matrix values") {
            matrix<float, 2, 2> m;
            m(0, 0) = 0.0f;
            m(0, 1) = constants<float>::pi / 2;
            m(1, 0) = constants<float>::pi;
            m(1, 1) = 3 * constants<float>::pi / 2;
            
            matrix<float, 2, 2> result = sin(m);
            CHECK(approx_equal(result(0, 0), 0.0f));
            CHECK(approx_equal(result(0, 1), 1.0f));
            CHECK(approx_equal(result(1, 0), 0.0f));
            CHECK(approx_equal(result(1, 1), -1.0f));
        }
        
        SUBCASE("expression templates") {
            vec3f v1(0.0f, 0.5f, 1.0f);
            vec3f v2(1.0f, 0.5f, 0.0f);
            
            auto expr = sin(v1 + v2);
            vec3f result = expr;
            
            for (size_t i = 0; i < 3; ++i) {
                CHECK(approx_equal(result[i], std::sin(v1[i] + v2[i])));
            }
        }
    }
    
    SUBCASE("cos function") {
        SUBCASE("scalar values") {
            CHECK(approx_equal(cos(0.0f), 1.0f));
            CHECK(approx_equal(cos(constants<float>::pi / 2), 0.0f));
            CHECK(approx_equal(cos(constants<float>::pi), -1.0f));
            CHECK(approx_equal(cos(3 * constants<float>::pi / 2), 0.0f));
            CHECK(approx_equal(cos(2 * constants<float>::pi), 1.0f));
            
            CHECK(approx_equal(cos(0.0_deg), 1.0f));
            CHECK(approx_equal(cos(90.0_deg), 0.0f));
            CHECK(approx_equal(cos(180.0_deg), -1.0f));
            CHECK(approx_equal(cos(270.0_deg), 0.0f));
            CHECK(approx_equal(cos(360.0_deg), 1.0f));
        }
        
        SUBCASE("vector values") {
            vec3f v(0.0f, constants<float>::pi / 2, constants<float>::pi);
            vec3f result = cos(v);
            CHECK(approx_equal(result[0], 1.0f));
            CHECK(approx_equal(result[1], 0.0f));
            CHECK(approx_equal(result[2], -1.0f));
        }
    }
    
    SUBCASE("tan function") {
        SUBCASE("scalar values") {
            CHECK(approx_equal(tan(0.0f), 0.0f));
            CHECK(approx_equal(tan(constants<float>::pi / 4), 1.0f));
            CHECK(approx_equal(tan(constants<float>::pi), 0.0f));
            CHECK(approx_equal(tan(-constants<float>::pi / 4), -1.0f));
            
            CHECK(approx_equal(tan(0.0_deg), 0.0f));
            CHECK(approx_equal(tan(45.0_deg), 1.0f));
            CHECK(approx_equal(tan(180.0_deg), 0.0f));
            CHECK(approx_equal(tan(-45.0_deg), -1.0f));
        }
    }
    
    SUBCASE("sincos function") {
        SUBCASE("scalar values") {
            auto [s, c] = sincos(constants<float>::pi / 4);
            CHECK(approx_equal(s, std::sin(constants<float>::pi / 4)));
            CHECK(approx_equal(c, std::cos(constants<float>::pi / 4)));
        }
        
        SUBCASE("angle values") {
            auto [s1, c1] = sincos(45.0_deg);
            CHECK(approx_equal(s1, std::sin(45.0f * constants<float>::deg_to_rad)));
            CHECK(approx_equal(c1, std::cos(45.0f * constants<float>::deg_to_rad)));
            
            auto [s2, c2] = sincos(constants<float>::pi / 4 * 1.0_rad);
            CHECK(approx_equal(s2, std::sin(constants<float>::pi / 4)));
            CHECK(approx_equal(c2, std::cos(constants<float>::pi / 4)));
        }
    }
}

TEST_CASE("Inverse trigonometric functions") {
    SUBCASE("asin function") {
        SUBCASE("scalar values") {
            auto angle1 = euler::asin(0.0f);
            CHECK(approx_equal(angle1.value(), 0.0f));

            auto angle2 = euler::asin(1.0f);
            CHECK(approx_equal(angle2.value(), constants<float>::pi / 2));

            auto angle3 = euler::asin(-1.0f);
            CHECK(approx_equal(angle3.value(), -constants<float>::pi / 2));

            // Test degree version
            auto angle_deg = euler::asin_deg(0.5f);
            CHECK(approx_equal(angle_deg.value(), 30.0f, 0.01f));
        }

        SUBCASE("vector values") {
            vec3f v(0.0f, 0.5f, 1.0f);
            auto result = euler::asin(v);
            CHECK(approx_equal(result[0].value(), 0.0f));
            CHECK(approx_equal(result[1].value(), std::asin(0.5f)));
            CHECK(approx_equal(result[2].value(), constants<float>::pi / 2));
        }
    }

    SUBCASE("acos function") {
        SUBCASE("scalar values") {
            auto angle1 = euler::acos(1.0f);
            CHECK(approx_equal(angle1.value(), 0.0f));

            auto angle2 = euler::acos(0.0f);
            CHECK(approx_equal(angle2.value(), constants<float>::pi / 2));

            auto angle3 = euler::acos(-1.0f);
            CHECK(approx_equal(angle3.value(), constants<float>::pi));

            // Test degree version
            auto angle_deg = euler::acos_deg(0.5f);
            CHECK(approx_equal(angle_deg.value(), 60.0f, 0.01f));
        }
    }

    SUBCASE("atan function") {
        SUBCASE("scalar values") {
            auto angle1 = euler::atan(0.0f);
            CHECK(approx_equal(angle1.value(), 0.0f));

            auto angle2 = euler::atan(1.0f);
            CHECK(approx_equal(angle2.value(), constants<float>::pi / 4));

            auto angle3 = euler::atan(-1.0f);
            CHECK(approx_equal(angle3.value(), -constants<float>::pi / 4));

            // Test degree version
            auto angle_deg = euler::atan_deg(1.0f);
            CHECK(approx_equal(angle_deg.value(), 45.0f, 0.01f));
        }
    }
    
    SUBCASE("atan2 function") {
        SUBCASE("scalar values") {
            auto angle1 = euler::atan2(0.0f, 1.0f);
            CHECK(approx_equal(angle1.value(), 0.0f));

            auto angle2 = euler::atan2(1.0f, 0.0f);
            CHECK(approx_equal(angle2.value(), constants<float>::pi / 2));

            auto angle3 = euler::atan2(0.0f, -1.0f);
            CHECK(approx_equal(angle3.value(), constants<float>::pi));

            auto angle4 = euler::atan2(-1.0f, 0.0f);
            CHECK(approx_equal(angle4.value(), -constants<float>::pi / 2));

            // Test degree version
            auto angle_deg = euler::atan2_deg(1.0f, 1.0f);
            CHECK(approx_equal(angle_deg.value(), 45.0f, 0.01f));
        }

        SUBCASE("vector values") {
            vec3f y(1.0f, 0.0f, -1.0f);
            vec3f x(0.0f, 1.0f, 0.0f);
            auto result = euler::atan2(y, x);
            CHECK(approx_equal(result[0].value(), constants<float>::pi / 2));
            CHECK(approx_equal(result[1].value(), 0.0f));
            CHECK(approx_equal(result[2].value(), -constants<float>::pi / 2));
        }
    }
}

TEST_CASE("Hyperbolic functions") {
    SUBCASE("sinh function") {
        SUBCASE("scalar values") {
            CHECK(approx_equal(sinh(0.0f), 0.0f));
            CHECK(approx_equal(sinh(1.0f), std::sinh(1.0f)));
            CHECK(approx_equal(sinh(-1.0f), -std::sinh(1.0f)));
        }
        
        SUBCASE("vector values") {
            vec3f v(0.0f, 1.0f, -1.0f);
            vec3f result = sinh(v);
            CHECK(approx_equal(result[0], 0.0f));
            CHECK(approx_equal(result[1], std::sinh(1.0f)));
            CHECK(approx_equal(result[2], -std::sinh(1.0f)));
        }
        
        SUBCASE("expression templates") {
            vec3f v(0.5f, 1.0f, 1.5f);
            auto expr = sinh(2.0f * v);
            vec3f result = expr;
            
            for (size_t i = 0; i < 3; ++i) {
                CHECK(approx_equal(result[i], std::sinh(2.0f * v[i])));
            }
        }
    }
    
    SUBCASE("cosh function") {
        SUBCASE("scalar values") {
            CHECK(approx_equal(cosh(0.0f), 1.0f));
            CHECK(approx_equal(cosh(1.0f), std::cosh(1.0f)));
            CHECK(approx_equal(cosh(-1.0f), std::cosh(1.0f)));
        }
    }
    
    SUBCASE("tanh function") {
        SUBCASE("scalar values") {
            CHECK(approx_equal(tanh(0.0f), 0.0f));
            CHECK(approx_equal(tanh(1.0f), std::tanh(1.0f)));
            CHECK(approx_equal(tanh(-1.0f), -std::tanh(1.0f)));
        }
    }
    
    SUBCASE("inverse hyperbolic functions") {
        SUBCASE("asinh") {
            CHECK(approx_equal(asinh(0.0f), 0.0f));
            CHECK(approx_equal(asinh(1.0f), std::asinh(1.0f)));
            
            vec3f v(0.0f, 1.0f, 2.0f);
            vec3f result = asinh(v);
            CHECK(approx_equal(result[0], 0.0f));
            CHECK(approx_equal(result[1], std::asinh(1.0f)));
            CHECK(approx_equal(result[2], std::asinh(2.0f)));
        }
        
        SUBCASE("acosh") {
            CHECK(approx_equal(acosh(1.0f), 0.0f));
            CHECK(approx_equal(acosh(2.0f), std::acosh(2.0f)));
            
            vec3f v(1.0f, 2.0f, 3.0f);
            vec3f result = acosh(v);
            CHECK(approx_equal(result[0], 0.0f));
            CHECK(approx_equal(result[1], std::acosh(2.0f)));
            CHECK(approx_equal(result[2], std::acosh(3.0f)));
        }
        
        SUBCASE("atanh") {
            CHECK(approx_equal(atanh(0.0f), 0.0f));
            CHECK(approx_equal(atanh(0.5f), std::atanh(0.5f)));
            
            vec3f v(0.0f, 0.5f, -0.5f);
            vec3f result = atanh(v);
            CHECK(approx_equal(result[0], 0.0f));
            CHECK(approx_equal(result[1], std::atanh(0.5f)));
            CHECK(approx_equal(result[2], -std::atanh(0.5f)));
        }
    }
}

TEST_CASE("Conversion functions") {
    SUBCASE("radians function") {
        CHECK(approx_equal(to_radians_raw(180.0f), constants<float>::pi));
        CHECK(approx_equal(to_radians_raw(90.0f), constants<float>::pi / 2));
        CHECK(approx_equal(to_radians_raw(45.0f), constants<float>::pi / 4));
        
        // Test with angle types
        auto rad = to_radians_angle(180.0_deg);
        CHECK(approx_equal(rad.value(), constants<float>::pi));
    }
    
    SUBCASE("degrees function") {
        CHECK(approx_equal(to_degrees_raw(constants<float>::pi), 180.0f));
        CHECK(approx_equal(to_degrees_raw(constants<float>::pi / 2), 90.0f));
        CHECK(approx_equal(to_degrees_raw(constants<float>::pi / 4), 45.0f));
        
        // Test with angle types
        auto deg = to_degrees_angle(constants<float>::pi * 1.0_rad);
        CHECK(approx_equal(deg.value(), 180.0f));
    }
}

TEST_CASE("Trigonometric identities") {
    SUBCASE("sin^2 + cos^2 = 1") {
        for (float angle = 0.0f; angle < 2 * constants<float>::pi; angle += 0.1f) {
            float s = sin(angle);
            float c = cos(angle);
            CHECK(approx_equal(s * s + c * c, 1.0f));
        }
    }
    
    SUBCASE("tan = sin/cos") {
        for (float angle = 0.1f; angle < constants<float>::pi / 2 - 0.1f; angle += 0.1f) {
            float t = tan(angle);
            float s = sin(angle);
            float c = cos(angle);
            CHECK(approx_equal(t, s / c));
        }
    }
}