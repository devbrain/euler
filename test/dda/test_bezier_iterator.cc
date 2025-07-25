#include <doctest/doctest.h>
#include <euler/dda/bezier_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::dda;

TEST_CASE("Quadratic Bezier iterator") {
    SUBCASE("Basic quadratic curve") {
        point2f p0{0, 0};
        point2f p1{50, 100};
        point2f p2{100, 0};
        
        std::vector<point2i> pixels;
        auto bezier = make_quadratic_bezier(p0, p1, p2);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should start at p0 and end at p2
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{100, 0});
        
        // Should go upward then downward (parabola-like)
        int max_y = 0;
        size_t max_y_index = 0;
        for (size_t i = 0; i < pixels.size(); ++i) {
            if (pixels[i].y > max_y) {
                max_y = pixels[i].y;
                max_y_index = i;
            }
        }
        CHECK(max_y > 0);
        CHECK(max_y_index > 0);
        CHECK(max_y_index < pixels.size() - 1);
    }
    
    SUBCASE("Straight line as quadratic Bezier") {
        point2f p0{0, 0};
        point2f p1{50, 50};
        point2f p2{100, 100};
        
        std::vector<point2i> pixels;
        auto bezier = make_quadratic_bezier(p0, p1, p2);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should approximate diagonal line
        for (const auto& p : pixels) {
            CHECK(std::abs(p.x - p.y) <= 2);
        }
    }
    
    SUBCASE("S-curve quadratic") {
        point2f p0{0, 0};
        point2f p1{100, 0};
        point2f p2{100, 100};
        
        std::vector<point2i> pixels;
        auto bezier = make_quadratic_bezier(p0, p1, p2);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{100, 100});
        
        // Should curve to the right first
        CHECK(pixels[pixels.size()/4].x > pixels[pixels.size()/4].y);
    }
}

TEST_CASE("Cubic Bezier iterator") {
    SUBCASE("Basic cubic curve") {
        point2f p0{0, 0};
        point2f p1{30, 100};
        point2f p2{70, 100};
        point2f p3{100, 0};
        
        std::vector<point2i> pixels;
        auto bezier = make_cubic_bezier(p0, p1, p2, p3);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{100, 0});
        
        // Should have smooth curve with peak
        int max_y = 0;
        for (const auto& p : pixels) {
            max_y = std::max(max_y, p.y);
        }
        CHECK(max_y > 50);
    }
    
    SUBCASE("S-shaped cubic Bezier") {
        point2f p0{0, 0};
        point2f p1{100, 0};
        point2f p2{0, 100};
        point2f p3{100, 100};
        
        std::vector<point2i> pixels;
        auto bezier = make_cubic_bezier(p0, p1, p2, p3);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should create S-shape
        // First half should tend right, second half should tend left
        size_t mid = pixels.size() / 2;
        float avg_x_first = 0, avg_x_second = 0;
        
        for (size_t i = 0; i < mid; ++i) {
            avg_x_first += static_cast<float>(pixels[i].x);
        }
        for (size_t i = mid; i < pixels.size(); ++i) {
            avg_x_second += static_cast<float>(pixels[i].x);
        }
        
        avg_x_first /= static_cast<float>(mid);
        avg_x_second /= static_cast<float>(pixels.size() - mid);
        
        CHECK(avg_x_first > 20.0f); // First half tends right
        CHECK(avg_x_second < 80.0f); // Second half tends left
    }
    
    SUBCASE("Loop in cubic Bezier") {
        point2f p0{0, 0};
        point2f p1{100, 0};
        point2f p2{100, 100};
        point2f p3{0, 100};
        
        std::vector<point2i> pixels;
        auto bezier = make_cubic_bezier(p0, p1, p2, p3);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{0, 100});
    }
}

TEST_CASE("General Bezier iterator") {
    SUBCASE("Linear Bezier (degree 1)") {
        std::vector<point2f> control_points = {
            {0, 0}, {100, 50}
        };
        
        std::vector<point2i> pixels;
        auto bezier = make_bezier(control_points);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should be approximately straight line
        // Check overall direction from start to end
        CHECK(pixels.front().x <= pixels.back().x); // Goes left to right
        
        // Check that it follows approximately the right slope (0.5)
        for (const auto& p : pixels) {
            float expected_y = 0.5f * float(p.x);
            float error = std::abs(float(p.y) - expected_y);
            CHECK(error <= 2.0f); // Allow some discretization error
        }
    }
    
    SUBCASE("Quartic Bezier (degree 4)") {
        std::vector<point2f> control_points = {
            {0, 0}, {25, 100}, {50, 0}, {75, 100}, {100, 0}
        };
        
        std::vector<point2i> pixels;
        auto bezier = make_bezier(control_points);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{100, 0});
        
        // Should have multiple peaks/valleys
        int direction_changes = 0;
        int last_direction = 0;
        
        for (size_t i = 1; i < pixels.size(); ++i) {
            int dy = pixels[i].y - pixels[i-1].y;
            int direction = (dy > 0) ? 1 : (dy < 0) ? -1 : 0;
            
            if (direction != 0 && last_direction != 0 && direction != last_direction) {
                direction_changes++;
            }
            if (direction != 0) {
                last_direction = direction;
            }
        }
        
        CHECK(direction_changes >= 1); // At least 1 peak for this curve
    }
}

TEST_CASE("Antialiased cubic Bezier") {
    SUBCASE("AA cubic coverage") {
        point2f p0{0, 0};
        point2f p1{30, 50};
        point2f p2{70, 50};
        point2f p3{100, 0};
        
        std::vector<aa_pixel<float>> pixels;
        auto bezier = make_aa_cubic_bezier(p0, p1, p2, p3);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back(*bezier);
        }
        
        CHECK(!pixels.empty());
        
        // Check coverage values
        for (const auto& p : pixels) {
            CHECK(p.coverage >= 0.0f);
            CHECK(p.coverage <= 1.0f);
        }
        
        // Should have partial coverage pixels
        bool has_partial = false;
        for (const auto& p : pixels) {
            if (p.coverage > 0.0f && p.coverage < 1.0f) {
                has_partial = true;
                break;
            }
        }
        CHECK(has_partial);
    }
}

TEST_CASE("Bezier edge cases") {
    SUBCASE("Single point Bezier") {
        std::vector<point2f> control_points = {{50, 50}};
        
        std::vector<point2i> pixels;
        auto bezier = make_bezier(control_points);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
            if (pixels.size() > 5) break; // Safety
        }
        
        REQUIRE(pixels.size() >= 1);
        CHECK(pixels[0] == point2i{50, 50});
    }
    
    SUBCASE("Very small Bezier") {
        point2f p0{0, 0};
        point2f p1{0.5f, 0.5f};
        point2f p2{1, 0};
        
        std::vector<point2i> pixels;
        auto bezier = make_quadratic_bezier(p0, p1, p2);
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.size() <= 15); // Small curve, but line interpolation may add pixels
    }
    
    SUBCASE("High curvature Bezier") {
        point2f p0{0, 0};
        point2f p1{100, 0};
        point2f p2{100, 100};
        point2f p3{0, 0}; // Returns to start
        
        std::vector<point2i> pixels;
        auto bezier = make_cubic_bezier(p0, p1, p2, p3, 0.1f); // Tight tolerance
        
        for (; bezier != decltype(bezier)::end(); ++bezier) {
            pixels.push_back((*bezier).pos);
            if (pixels.size() > 10000) break; // Safety limit
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == pixels.back()); // Should return to start
    }
}