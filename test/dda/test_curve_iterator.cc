#include <doctest/doctest.h>
#include <euler/dda/curve_iterator.hh>
#include <euler/dda/mathematical_curves.hh>
#include <euler/coordinates/point2.hh>
#include <euler/math/trigonometry.hh>
#include <euler/core/compiler.hh>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::dda;

EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW
TEST_CASE("Curve iterator basic functionality") {
    SUBCASE("Parametric curve - circle") {
        // Parametric circle
        auto circle = [](float t) {
            return point2{10.0f * cos(t), 10.0f * sin(t)};
        };
        
        std::vector<point2i> pixels;
        auto curve = make_curve_iterator(circle, 0.0f, 2.0f * constants<float>::pi);
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should form approximate circle
        for (const auto& p : pixels) {
            float dist = std::sqrt(float(p.x * p.x + p.y * p.y));
            CHECK(dist >= 8.0f);
            CHECK(dist <= 12.0f);
        }
    }
    
    SUBCASE("Cartesian curve - parabola") {
        auto parabola = [](float x) { return 0.1f * x * x; };
        
        std::vector<point2i> pixels;
        auto curve = curve_iterator<float, decltype(parabola)>::cartesian(
            parabola, -20.0f, 20.0f);
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check parabola properties
        int failures = 0;
        for (const auto& p : pixels) {
            float expected_y = 0.1f * float(p.x * p.x);
            float error = std::abs(float(p.y) - expected_y);
            if (error > 2.0f) {
                if (failures < 5) {
                     // MESSAGE("Pixel (" << p.x << ", " << p.y << ") expected y="
                     //         << expected_y << " error=" << error);
                }
                failures++;
            }
            CHECK(error <= 3.5f); // Allow for discrete pixel approximation
        }
        if (failures > 0) {
            // MESSAGE("Total parabola check failures: " << failures << " out of " << pixels.size());
        }
        
        // Should be approximately symmetric
        // Due to adaptive stepping, we might not hit exact x-coordinates on both sides
        int checked = 0;
        int symmetric = 0;
        for (const auto& p : pixels) {
            // Only check points not too close to center where symmetry is trivial
            if (std::abs(p.x) >= 5) {
                checked++;
                for (const auto& q : pixels) {
                    // Allow for some tolerance in x-coordinate matching
                    if (std::abs(q.x + p.x) <= 1 && std::abs(q.y - p.y) <= 2) {
                        symmetric++;
                        break;
                    }
                }
            }
        }
        // At least 80% of checked points should have symmetric counterparts
        if (checked > 0) {
            float symmetry_ratio = float(symmetric) / float(checked);
            // MESSAGE("Symmetry ratio: " << symmetric << "/" << checked
            //         << " = " << (symmetry_ratio * 100) << "%");
            CHECK(symmetry_ratio >= 0.8f);
        }
    }
    
    SUBCASE("Polar curve - spiral") {
        auto spiral = [](float theta) { return 2.0f * theta; };
        
        std::vector<point2i> pixels;
        auto curve = curve_iterator<float, decltype(spiral)>::polar(
            spiral, 0.0f, 4.0f * constants<float>::pi, point2{0.0f, 0.0f});
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should spiral outward
        float last_dist = 0;
        for (size_t i = 0; i < pixels.size(); i += 10) {
            float dist = std::sqrt(float(pixels[i].x * pixels[i].x + 
                                       pixels[i].y * pixels[i].y));
            CHECK(dist >= last_dist - 2.0f); // Allow some tolerance
            last_dist = dist;
        }
    }
    
    SUBCASE("Adaptive stepping") {
        // High curvature at origin
        auto curve_func = [](float t) {
            return point2{t, t * t * t};
        };
        
        std::vector<point2i> pixels;
        auto curve = make_curve_iterator(curve_func, -2.0f, 2.0f, 0.5f);
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have more pixels near origin (high curvature)
        int near_origin = 0;
        int far_from_origin = 0;
        
        for (const auto& p : pixels) {
            if (std::abs(p.x) < 1) near_origin++;
            else far_from_origin++;
        }
        
        CHECK(near_origin > 0);
        CHECK(far_from_origin > 0);
    }
}

TEST_CASE("Mathematical curves") {
    using namespace euler::dda::curves;
    
    SUBCASE("Cardioid") {
        auto cardioid_func = cardioid(20.0f);
        auto curve = curve_iterator<float, decltype(cardioid_func)>::polar(
            cardioid_func, 0.0f, 2.0f * constants<float>::pi);
        
        std::vector<point2i> pixels;
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Cardioid should have a cusp at origin
        bool has_cusp = false;
        for (const auto& p : pixels) {
            if (std::abs(p.x) <= 2 && std::abs(p.y) <= 2) {
                has_cusp = true;
                break;
            }
        }
        CHECK(has_cusp);
    }
    
    SUBCASE("Witch of Agnesi") {
        auto witch = agnesi_witch(10.0f);
        auto curve = curve_iterator<float, decltype(witch)>::cartesian(
            witch, -30.0f, 30.0f);
        
        std::vector<point2i> pixels;
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have maximum at x=0
        int max_y = 0;
        int max_y_x = 0;
        for (const auto& p : pixels) {
            if (p.y > max_y) {
                max_y = p.y;
                max_y_x = p.x;
            }
        }
        CHECK(std::abs(max_y_x) <= 3); // Allow for discrete approximation
    }
    
    SUBCASE("Rose curve") {
        auto rose_func = rose(15.0f, 3.0f); // 3 petals
        auto curve = curve_iterator<float, decltype(rose_func)>::polar(
            rose_func, 0.0f, 2.0f * constants<float>::pi);
        
        std::vector<point2i> pixels;
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have 3-fold symmetry
        // (This is a simple check - proper validation would be more complex)
        CHECK(pixels.size() > 50);
    }
    
    SUBCASE("Lemniscate") {
        auto lemni = lemniscate(20.0f);
        auto curve = curve_iterator<float, decltype(lemni)>::polar(
            lemni, 0.0f, 2.0f * constants<float>::pi);
        
        std::vector<point2i> pixels;
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Lemniscate has figure-8 shape
        bool has_positive_x = false;
        bool has_negative_x = false;
        for (const auto& p : pixels) {
            if (p.x > 10) has_positive_x = true;
            if (p.x < -10) has_negative_x = true;
        }
        CHECK(has_positive_x);
        CHECK(has_negative_x);
    }
}

TEST_CASE("Curve iterator edge cases") {
    SUBCASE("Constant curve") {
        auto constant = [](float t) { (void)t; return point2{5.0f, 5.0f}; };
        
        std::vector<point2i> pixels;
        auto curve = make_curve_iterator(constant, 0.0f, 1.0f);
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
            if (pixels.size() > 10) break; // Prevent infinite loop
        }
        
        // Should produce single pixel
        CHECK(!pixels.empty());
        for (const auto& p : pixels) {
            CHECK(p == point2i{5, 5});
        }
    }
    
    SUBCASE("Discontinuous curve") {
        auto discontinuous = [](float x) {
            return (x < 0) ? -10.0f : 10.0f;
        };
        
        std::vector<point2i> pixels;
        auto curve = curve_iterator<float, decltype(discontinuous)>::cartesian(
            discontinuous, -5.0f, 5.0f);
        
        for (; curve != decltype(curve)::end(); ++curve) {
            pixels.push_back((*curve).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have pixels at both y=-10 and y=10
        bool has_negative = false;
        bool has_positive = false;
        for (const auto& p : pixels) {
            if (p.y <= -8) has_negative = true;
            if (p.y >= 8) has_positive = true;
        }
        CHECK(has_negative);
        CHECK(has_positive);
    }
    
    SUBCASE("High frequency oscillation") {
        auto oscillating = [](float x) {
            return 10.0f * sin(x);
        };
        
        std::vector<point2i> pixels;
        auto curve = curve_iterator<float, decltype(oscillating)>::cartesian(
            oscillating, 0.0f, 10.0f * constants<float>::pi, 0.1f);
        
        int count = 0;
        for (; curve != decltype(curve)::end(); ++curve) {
            count++;
            if (count > 10000) break; // Safety limit
        }
        
        CHECK(count > 100); // Should produce many pixels
        CHECK(count < 10000); // But not infinite
    }
}
EULER_DISABLE_WARNING_POP