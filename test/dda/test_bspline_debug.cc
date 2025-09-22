#include <doctest/doctest.h>
#include <euler/dda/bspline_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/core/compiler.hh>
#include <vector>
#include <iostream>

using namespace euler;
using namespace euler::dda;

EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW
TEST_CASE("Debug B-spline end point issue") {
    SUBCASE("Cubic B-spline with 5 control points") {
        // Create control points
        std::vector<point2f> control_points = {
            {10.0f, 10.0f},   // P0
            {30.0f, 50.0f},   // P1
            {50.0f, 50.0f},   // P2
            {70.0f, 30.0f},   // P3
            {90.0f, 10.0f}    // P4
        };
        
        MESSAGE("Control points:");
        for (size_t i = 0; i < control_points.size(); ++i) {
            MESSAGE("P" << i << ": (" << control_points[i].x << ", " << control_points[i].y << ")");
        }
        
        // Create B-spline iterator
        auto bspline = make_bspline(control_points, 3); // degree 3 (cubic)
        
        std::vector<point2i> pixels;
        int count = 0;
        point2i last_pixel{0, 0};
        
        for (; bspline != decltype(bspline)::end(); ++bspline) {
            auto pixel = *bspline;
            pixels.push_back(pixel.pos);
            
            if (count < 10 || count % 10 == 0) {
                MESSAGE("Pixel " << count << ": (" << pixel.pos.x << ", " << pixel.pos.y << ")");
            }
            
            last_pixel = pixel.pos;
            count++;
            
            // Safety limit
            if (count > 1000) {
                FAIL("Too many pixels generated!");
                break;
            }
        }
        
        MESSAGE("Total pixels: " << pixels.size());
        MESSAGE("Last pixel: (" << last_pixel.x << ", " << last_pixel.y << ")");
        
        // Check that the curve ends near the last control point
        // For a cubic B-spline, the curve should end at or near P4
        auto expected_end = control_points.back();
        MESSAGE("Expected end near: (" << expected_end.x << ", " << expected_end.y << ")");
        
        // Check if last pixel is reasonably close to the last control point
        float dx = std::abs(float(last_pixel.x) - expected_end.x);
        float dy = std::abs(float(last_pixel.y) - expected_end.y);
        float distance = std::sqrt(dx * dx + dy * dy);
        
        MESSAGE("Distance from last pixel to last control point: " << distance);
        
        CHECK(distance < 5.0f); // Should be within 5 pixels
        
        // Also check for the "jump to origin" issue
        for (size_t i = 1; i < pixels.size(); ++i) {
            const auto& p = pixels[i];
            const auto& prev = pixels[i-1];
            
            // Check for huge jumps that would indicate jumping to origin
            int jump_x = std::abs(p.x - prev.x);
            int jump_y = std::abs(p.y - prev.y);
            
            if (jump_x > 30 || jump_y > 30) {
                FAIL_CHECK("Found huge jump at index " << i << ": from (" 
                        << prev.x << ", " << prev.y << ") to (" 
                        << p.x << ", " << p.y << ")");
            }
        }
    }
    
    SUBCASE("Check if B-spline properly ends at last control point") {
        // For a clamped uniform B-spline, the curve should pass through
        // or near the first and last control points
        std::vector<point2f> control_points = {
            {0.0f, 0.0f},
            {25.0f, 50.0f},
            {50.0f, 50.0f},
            {75.0f, 25.0f},
            {100.0f, 0.0f}
        };
        
        auto bspline = make_bspline(control_points, 3);
        
        point2i first_pixel{0, 0};
        point2i last_pixel{0, 0};
        bool got_first = false;
        int count = 0;
        
        for (; bspline != decltype(bspline)::end(); ++bspline) {
            auto pixel = *bspline;
            if (!got_first) {
                first_pixel = pixel.pos;
                got_first = true;
            }
            last_pixel = pixel.pos;
            count++;
            
            if (count > 1000) break;
        }
        
        // MESSAGE("First pixel: (" << first_pixel.x << ", " << first_pixel.y << ")");
        // MESSAGE("Last pixel: (" << last_pixel.x << ", " << last_pixel.y << ")");
        // MESSAGE("First control point: (" << control_points.front().x << ", " << control_points.front().y << ")");
        // MESSAGE("Last control point: (" << control_points.back().x << ", " << control_points.back().y << ")");
        
        // Check distance to first and last control points
        float dist_first = std::sqrt(
            std::pow(float(first_pixel.x) - control_points.front().x, 2.0f) +
            std::pow(float(first_pixel.y) - control_points.front().y, 2.0f)
        );
        float dist_last = std::sqrt(
            std::pow(float(last_pixel.x) - control_points.back().x, 2.0f) +
            std::pow(float(last_pixel.y) - control_points.back().y, 2.0f)
        );
        
        // MESSAGE("Distance from first pixel to first control point: " << dist_first);
        // MESSAGE("Distance from last pixel to last control point: " << dist_last);
        
        CHECK(dist_first < 2.0f);
        CHECK(dist_last < 2.0f);
    }
}
EULER_DISABLE_WARNING_POP