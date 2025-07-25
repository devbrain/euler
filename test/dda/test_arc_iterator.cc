#include <doctest/doctest.h>
#include <euler/dda/arc_iterator.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/dda/ellipse_iterator.hh>
#include <euler/angles/angle.hh>
#include <euler/coordinates/point2.hh>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::dda;

TEST_CASE("Arc iterators") {
    SUBCASE("Basic circle arc") {
        point2f center{50, 50};
        float radius = 20;
        
        // Quarter circle arc from 0 to 90 degrees
        auto arc = make_arc_iterator(center, radius, degree<float>(0), degree<float>(90));
        
        std::vector<point2i> pixels;
        for (; arc != decltype(arc)::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check that all pixels are in the first quadrant relative to center
        for (const auto& p : pixels) {
            CHECK(p.x >= static_cast<int>(center.x));
            CHECK(p.y >= static_cast<int>(center.y));
        }
    }
    
    SUBCASE("Filled circle arc") {
        point2f center{50, 50};
        float radius = 10;
        
        // Half circle from 0 to 180 degrees
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(0), degree<float>(180));
        
        std::vector<span> spans;
        for (; filled != decltype(filled)::end(); ++filled) {
            spans.push_back(*filled);
        }
        
        CHECK(!spans.empty());
        
        // For a 0-180 degree arc (right semicircle), all spans should exist
        // Just check that we have spans
        CHECK(spans.size() > 0);
    }
    
    SUBCASE("Antialiased circle arc") {
        point2f center{50.5f, 50.5f};
        float radius = 15.0f;
        
        auto aa_arc = make_aa_arc_iterator(center, radius, degree<float>(45), degree<float>(135));
        
        std::vector<aa_pixel<float>> aa_pixels;
        for (; aa_arc != decltype(aa_arc)::end(); ++aa_arc) {
            aa_pixels.push_back(*aa_arc);
        }
        
        CHECK(!aa_pixels.empty());
        
        // Check coverage values
        for (const auto& p : aa_pixels) {
            CHECK(p.coverage >= 0.0f);
            CHECK(p.coverage <= 1.0f);
        }
    }
    
    SUBCASE("Ellipse arc") {
        point2f center{50, 50};
        float a = 30;
        float b = 20;
        
        auto arc = make_ellipse_arc_iterator(center, a, b, degree<float>(0), degree<float>(90));
        
        std::vector<point2i> pixels;
        for (; arc != decltype(arc)::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check that all pixels are in the first quadrant
        for (const auto& p : pixels) {
            CHECK(p.x >= static_cast<int>(center.x));
            CHECK(p.y >= static_cast<int>(center.y));
        }
    }
    
    SUBCASE("Arc crossing 0 degrees") {
        point2f center{50, 50};
        float radius = 20;
        
        // Arc from 270 to 90 degrees (crosses 0)
        auto arc = make_arc_iterator(center, radius, degree<float>(270), degree<float>(90));
        
        std::vector<point2i> pixels;
        for (; arc != decltype(arc)::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have pixels in quadrants 1 and 4 (right side)
        bool has_q1 = false, has_q4 = false;
        for (const auto& p : pixels) {
            if (static_cast<float>(p.x) >= center.x && static_cast<float>(p.y) >= center.y) has_q1 = true;
            if (static_cast<float>(p.x) >= center.x && static_cast<float>(p.y) < center.y) has_q4 = true;
        }
        CHECK(has_q1);
        CHECK(has_q4);
    }
    
    SUBCASE("Filled arc vertical line bug") {
        // Test case that was causing vertical line on y axis
        point2f center{50, 50};
        float radius = 20;
        
        // Test a 90-degree arc from 0 to 90 degrees
        auto filled = make_filled_arc_iterator(center, radius, degree<float>(0), degree<float>(90));
        
        std::vector<span> spans;
        for (; filled != decltype(filled)::end(); ++filled) {
            spans.push_back(*filled);
        }
        
        CHECK(!spans.empty());
        
        // Check that no span starts at x=0 (which would be the vertical line bug)
        bool has_vertical_line_bug = false;
        for (const auto& s : spans) {
            if (s.x_start == 0 || s.x_end == 0) {
                // Only flag as bug if the span is far from the actual arc
                float dy = static_cast<float>(s.y) - center.y;
                float expected_x_at_y = center.x + std::sqrt(radius * radius - dy * dy);
                if (std::abs(expected_x_at_y) > radius * 0.5f) {
                    has_vertical_line_bug = true;
                    break;
                }
            }
        }
        CHECK(!has_vertical_line_bug);
        
        // Also check that all spans are within expected bounds
        for (const auto& s : spans) {
            // For a 0-90 degree arc, all x values should be >= center.x
            CHECK(s.x_start >= static_cast<int>(center.x));
            CHECK(s.x_end >= static_cast<int>(center.x));
            // And all y values should be >= center.y
            CHECK(s.y >= static_cast<int>(center.y));
        }
    }
    
    SUBCASE("Filled ellipse arc vertical line bug") {
        // Test the same for ellipse arcs
        point2f center{50, 50};
        float a = 30;
        float b = 20;
        
        auto filled = make_filled_ellipse_arc_iterator(center, a, b, degree<float>(0), degree<float>(90));
        
        std::vector<span> spans;
        for (; filled != decltype(filled)::end(); ++filled) {
            spans.push_back(*filled);
        }
        
        CHECK(!spans.empty());
        
        // Check for vertical line bug
        bool has_vertical_line_bug = false;
        for (const auto& s : spans) {
            if (s.x_start == 0 || s.x_end == 0) {
                // Check if this is a legitimate part of the ellipse
                float dy = (static_cast<float>(s.y) - center.y) / b;
                if (std::abs(dy) <= 1.0f) {
                    float expected_x_offset = a * std::sqrt(1 - dy * dy);
                    float expected_x = center.x + expected_x_offset;
                    if (expected_x > a * 0.5f) {
                        has_vertical_line_bug = true;
                        break;
                    }
                }
            }
        }
        CHECK(!has_vertical_line_bug);
        
        // Check bounds for 0-90 degree arc
        for (const auto& s : spans) {
            CHECK(s.x_start >= static_cast<int>(center.x));
            CHECK(s.x_end >= static_cast<int>(center.x));
            CHECK(s.y >= static_cast<int>(center.y));
        }
    }
}