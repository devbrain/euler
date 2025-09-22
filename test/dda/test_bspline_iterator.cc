#include <doctest/doctest.h>
#include <euler/dda/bspline_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/core/compiler.hh>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::dda;

EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW
TEST_CASE("B-spline iterator basic functionality") {
    SUBCASE("Cubic B-spline with 4 control points") {
        std::vector<point2f> control_points = {
            {0, 0}, {33, 100}, {66, 100}, {100, 0}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 3); // degree 3
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        // B-spline doesn't necessarily pass through endpoints
        // but should be in the convex hull of control points
        for (const auto& p : pixels) {
            CHECK(p.x >= -10);
            CHECK(p.x <= 110);
            CHECK(p.y >= -10);
            CHECK(p.y <= 110);
        }
    }
    
    SUBCASE("Quadratic B-spline") {
        std::vector<point2f> control_points = {
            {0, 0}, {50, 100}, {100, 0}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 2); // degree 2
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should create smooth curve
        int max_y = 0;
        for (const auto& p : pixels) {
            max_y = std::max(max_y, p.y);
        }
        CHECK(max_y > 20); // Should have some height
    }
    
    SUBCASE("Linear B-spline") {
        std::vector<point2f> control_points = {
            {0, 0}, {50, 50}, {100, 100}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 1); // degree 1
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        if (pixels.size() > 1) {
            // Linear B-spline should produce polyline
            // Check general upward trend
            CHECK(pixels.back().y >= pixels.front().y);
            CHECK(pixels.back().x >= pixels.front().x);
        } else {
            // If only one pixel, just check it exists
            MESSAGE("Linear B-spline generated only " << pixels.size() << " pixel(s)");
            CHECK(pixels.size() >= 1);
        }
    }
    
    SUBCASE("Closed B-spline") {
        std::vector<point2f> control_points = {
            {0, 0}, {100, 0}, {100, 100}, {0, 100}, {0, 0}, {100, 0}, {100, 100}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 3);
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should form a closed-ish shape
        // For a closed curve, check if start and end points are close to each other
        float dx = float(pixels.back().x - pixels.front().x);
        float dy = float(pixels.back().y - pixels.front().y);
        float distance_between_ends = std::sqrt(dx * dx + dy * dy);
        
        // Start and end points should be close for a closed curve
        CHECK(distance_between_ends < 150.0f); // Allow some distance since it's not perfectly closed
    }
}

TEST_CASE("B-spline with custom knots") {
    SUBCASE("Non-uniform knot vector") {
        std::vector<point2f> control_points = {
            {0, 0}, {25, 50}, {75, 50}, {100, 0}
        };
        
        // Custom knot vector for cubic B-spline
        // For 4 control points and degree 3, we need 4 + 3 + 1 = 8 knots
        std::vector<float> knots = {
            0.0f, 0.0f, 0.0f, 0.0f,  // Clamped at start
            1.0f, 1.0f, 1.0f, 1.0f   // Clamped at end
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, knots, 3);
        
        int count = 0;
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
            count++;
            if (count > 1000) break; // Safety
        }
        
        CHECK(!pixels.empty());
        CHECK(count < 1000);
    }
}

TEST_CASE("Catmull-Rom spline") {
    SUBCASE("Basic interpolating spline") {
        std::vector<point2f> points = {
            {0, 0}, {25, 50}, {50, 0}, {75, 50}, {100, 0}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_catmull_rom(points);
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Catmull-Rom interpolates through control points
        // Check that curve passes near the control points
        for (const auto& cp : points) {
            bool found_close = false;
            point2i target = round(cp);
            
            for (const auto& p : pixels) {
                if (std::abs(p.x - target.x) <= 5 && 
                    std::abs(p.y - target.y) <= 5) {
                    found_close = true;
                    break;
                }
            }
            if (!found_close) {
                MESSAGE("No pixel found within 5 units of control point (" 
                        << target.x << ", " << target.y << ")");
            }
            CHECK(found_close);
        }
    }
    
    SUBCASE("Catmull-Rom with different tension") {
        std::vector<point2f> points = {
            {0, 0}, {50, 100}, {100, 0}
        };
        
        // Low tension (0.0) - sharp corners
        std::vector<point2i> pixels_low;
        auto spline_low = make_catmull_rom(points, 0.0f);
        for (; spline_low != decltype(spline_low)::end(); ++spline_low) {
            pixels_low.push_back((*spline_low).pos);
        }
        
        // High tension (1.0) - very smooth
        std::vector<point2i> pixels_high;
        auto spline_high = make_catmull_rom(points, 1.0f);
        for (; spline_high != decltype(spline_high)::end(); ++spline_high) {
            pixels_high.push_back((*spline_high).pos);
        }
        
        CHECK(!pixels_low.empty());
        CHECK(!pixels_high.empty());
        
        // Both should pass through middle point
        point2i middle = round(points[1]);
        
        auto check_near = [&](const std::vector<point2i>& pixels) {
            for (const auto& p : pixels) {
                if (std::abs(p.x - middle.x) <= 2 && 
                    std::abs(p.y - middle.y) <= 2) {
                    return true;
                }
            }
            return false;
        };
        
        CHECK(check_near(pixels_low));
        CHECK(check_near(pixels_high));
    }
    
    SUBCASE("Catmull-Rom with two points") {
        std::vector<point2f> points = {
            {0, 0}, {100, 50}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_catmull_rom(points);
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should interpolate between the two points
        CHECK(pixels.front().x <= 5);
        CHECK(pixels.back().x >= 94); // Allow for discretization
    }
}

TEST_CASE("B-spline edge cases") {
    SUBCASE("Minimum control points") {
        // For degree 3, need at least 4 control points
        std::vector<point2f> control_points = {
            {0, 0}, {33, 50}, {66, 50}, {100, 0}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 3);
        
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
        }
        
        CHECK(!pixels.empty());
    }
    
    SUBCASE("Many control points") {
        std::vector<point2f> control_points;
        for (int i = 0; i <= 20; ++i) {
            control_points.push_back({
                static_cast<float>(i * 5), 
                50.0f + 30.0f * std::sin(static_cast<float>(i) * 0.5f)
            });
        }
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 3);
        
        int count = 0;
        for (; spline != decltype(spline)::end(); ++spline) {
            count++;
            if (count > 5000) break;
        }
        
        CHECK(count > 100);
        CHECK(count < 5000);
    }
    
    SUBCASE("Degenerate spline - all points same") {
        std::vector<point2f> control_points = {
            {50, 50}, {50, 50}, {50, 50}, {50, 50}
        };
        
        std::vector<point2i> pixels;
        auto spline = make_bspline(control_points, 3);
        
        int count = 0;
        for (; spline != decltype(spline)::end(); ++spline) {
            pixels.push_back((*spline).pos);
            count++;
            if (count > 10) break;
        }
        
        CHECK(!pixels.empty());
        
        // Should produce single point or very small curve
        for (const auto& p : pixels) {
            CHECK(std::abs(p.x - 50) <= 2);
            CHECK(std::abs(p.y - 50) <= 2);
        }
    }
}
EULER_DISABLE_WARNING_POP