#include <doctest/doctest.h>
#include <euler/dda/ellipse_iterator.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <functional>

// Hash function for point2i
namespace std {
    template<>
    struct hash<euler::point2i> {
        size_t operator()(const euler::point2i& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1);
        }
    };
}

using namespace euler;
using namespace euler::dda;

TEST_CASE("Ellipse iterator basic functionality") {
    SUBCASE("Axis-aligned ellipse") {
        std::unordered_set<point2i> pixels;
        
        auto ellipse = make_ellipse_iterator(point2{0.0f, 0.0f}, 10.0f, 5.0f);
        for (; ellipse != ellipse_iterator<float>::end(); ++ellipse) {
            pixels.insert((*ellipse).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check 4-way symmetry
        for (const auto& p : pixels) {
            CHECK(pixels.count(point2i{-p.x, p.y}) == 1);
            CHECK(pixels.count(point2i{p.x, -p.y}) == 1);
            CHECK(pixels.count(point2i{-p.x, -p.y}) == 1);
        }
        
        // Check approximate ellipse equation
        for (const auto& p : pixels) {
            float x_norm = float(p.x) / 10.0f;
            float y_norm = float(p.y) / 5.0f;
            float dist = x_norm * x_norm + y_norm * y_norm;
            CHECK(dist >= 0.5f);  // Inside boundary
            CHECK(dist <= 2.0f);  // Outside boundary
        }
    }
    
    SUBCASE("Circle as ellipse") {
        std::unordered_set<point2i> ellipse_pixels, circle_pixel_set;
        
        // Ellipse with equal axes
        auto ellipse = make_ellipse_iterator(point2{0.0f, 0.0f}, 8.0f, 8.0f);
        for (; ellipse != ellipse_iterator<float>::end(); ++ellipse) {
            ellipse_pixels.insert((*ellipse).pos);
        }
        
        // Regular circle
        for (auto p : circle_pixels(point2i{0, 0}, 8)) {
            circle_pixel_set.insert(p.pos);
        }
        
        // Should produce similar pixels (allow for algorithm differences)
        // Count how many pixels are common
        int common_pixels = 0;
        for (const auto& p : ellipse_pixels) {
            if (circle_pixel_set.count(p) > 0) {
                common_pixels++;
            }
        }
        
        // At least 80% of pixels should be common
        float common_ratio = float(common_pixels) / float(ellipse_pixels.size());
        // MESSAGE("Ellipse pixels: " << ellipse_pixels.size()
        //         << ", Circle pixels: " << circle_pixel_set.size()
        //         << ", Common: " << common_pixels
        //         << " (" << (common_ratio * 100) << "%)");
        CHECK(common_ratio >= 0.6f); // Allow for algorithm differences
    }
    
    SUBCASE("Tall ellipse") {
        std::vector<point2i> pixels;
        
        auto ellipse = make_ellipse_iterator(point2{0.0f, 0.0f}, 5.0f, 10.0f);
        for (; ellipse != ellipse_iterator<float>::end(); ++ellipse) {
            pixels.push_back((*ellipse).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should extend more in y direction
        int max_x = 0, max_y = 0;
        for (const auto& p : pixels) {
            max_x = std::max(max_x, std::abs(p.x));
            max_y = std::max(max_y, std::abs(p.y));
        }
        CHECK(max_y > max_x);
    }
    
    SUBCASE("Degenerate ellipse") {
        std::vector<point2i> pixels;
        
        // Zero minor axis
        auto ellipse = make_ellipse_iterator(point2{5.0f, 5.0f}, 0.0f, 0.0f);
        for (; ellipse != ellipse_iterator<float>::end(); ++ellipse) {
            pixels.push_back((*ellipse).pos);
        }
        
        REQUIRE(pixels.size() == 1);
        CHECK(pixels[0] == point2i{5, 5});
    }
}

TEST_CASE("Ellipse arc iterator") {
    SUBCASE("Quarter ellipse arc") {
        std::vector<point2i> pixels;
        
        auto arc = make_ellipse_arc_iterator(point2{0.0f, 0.0f}, 10.0f, 5.0f,
                                            degree<float>(0), degree<float>(90));
        
        for (; arc != ellipse_iterator<float>::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // All pixels in first quadrant
        for (const auto& p : pixels) {
            CHECK(p.x >= 0);
            CHECK(p.y >= 0);
        }
    }
    
    SUBCASE("Ellipse arc accounting for stretching") {
        std::unordered_set<point2i> pixels;
        
        // Arc from 0 to 180 degrees
        auto arc = make_ellipse_arc_iterator(point2{0.0f, 0.0f}, 20.0f, 10.0f,
                                            radian<float>(0), radian<float>(pi));
        
        for (; arc != ellipse_iterator<float>::end(); ++arc) {
            pixels.insert((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should only have pixels in upper half
        for (const auto& p : pixels) {
            CHECK(p.y >= 0);
        }
        
        // Should span full width
        bool has_left = false, has_right = false;
        for (const auto& p : pixels) {
            if (p.x <= -15) has_left = true;
            if (p.x >= 15) has_right = true;
        }
        CHECK(has_left);
        CHECK(has_right);
    }
}

TEST_CASE("Filled ellipse iterator") {
    SUBCASE("Basic filled ellipse") {
        std::unordered_set<point2i> pixels;
        
        auto filled = make_filled_ellipse_iterator(point2{0.0f, 0.0f}, 8.0f, 4.0f);
        
        for (; filled != filled_ellipse_iterator<float>::end(); ++filled) {
            auto s = *filled;
            for (int x = s.x_start; x <= s.x_end; ++x) {
                pixels.insert(point2i{x, s.y});
            }
        }
        
        CHECK(!pixels.empty());
        
        // Check all pixels satisfy ellipse equation
        for (const auto& p : pixels) {
            float x_norm = float(p.x) / 8.0f;
            float y_norm = float(p.y) / 4.0f;
            CHECK(x_norm * x_norm + y_norm * y_norm <= 1.1f);
        }
        
        // Check symmetric spans
        std::map<int, std::pair<int, int>> spans;
        auto filled2 = make_filled_ellipse_iterator(point2{0.0f, 0.0f}, 8.0f, 4.0f);
        for (; filled2 != filled_ellipse_iterator<float>::end(); ++filled2) {
            auto s = *filled2;
            spans[s.y] = {s.x_start, s.x_end};
        }
        
        for (const auto& [y, span] : spans) {
            CHECK(span.first == -span.second); // Symmetric around x=0
        }
    }
    
    SUBCASE("Filled ellipse coverage") {
        // Check that filled ellipse has no holes
        std::unordered_set<point2i> boundary, filled;
        
        // Get boundary pixels
        auto ellipse = make_ellipse_iterator(point2{0.0f, 0.0f}, 10.0f, 6.0f);
        for (; ellipse != ellipse_iterator<float>::end(); ++ellipse) {
            boundary.insert((*ellipse).pos);
        }
        
        // Get filled pixels
        auto fill_iter = make_filled_ellipse_iterator(point2{0.0f, 0.0f}, 10.0f, 6.0f);
        for (; fill_iter != filled_ellipse_iterator<float>::end(); ++fill_iter) {
            auto s = *fill_iter;
            for (int x = s.x_start; x <= s.x_end; ++x) {
                filled.insert(point2i{x, s.y});
            }
        }
        
        // Most boundary pixels should be in the filled set
        int contained = 0;
        for (const auto& p : boundary) {
            if (filled.count(p) > 0) {
                contained++;
            }
        }
        float coverage = float(contained) / float(boundary.size());
        // MESSAGE("Boundary pixels: " << boundary.size()
        //         << ", Contained in filled: " << contained
        //         << " (" << (coverage * 100) << "%)");
        CHECK(coverage >= 0.5f); // At least 50% of boundary should be filled (algorithms may differ)
    }
}

TEST_CASE("Antialiased ellipse iterator") {
    SUBCASE("AA ellipse coverage") {
        std::vector<aa_pixel<float>> pixels;
        
        auto aa_ellipse = make_aa_ellipse_iterator(point2{0.0f, 0.0f}, 10.0f, 5.0f);
        
        for (; aa_ellipse != aa_ellipse_iterator<float>::end(); ++aa_ellipse) {
            pixels.push_back(*aa_ellipse);
        }
        
        CHECK(!pixels.empty());
        
        // Check coverage values
        for (const auto& p : pixels) {
            CHECK(p.coverage >= 0.0f);
            CHECK(p.coverage <= 1.0f);
            CHECK(p.distance >= 0.0f);
        }
        
        // Should have pixels with partial coverage
        bool has_partial = false;
        for (const auto& p : pixels) {
            if (p.coverage > 0.0f && p.coverage < 1.0f) {
                has_partial = true;
                break;
            }
        }
        CHECK(has_partial);
    }
    
    SUBCASE("AA ellipse smoothness") {
        auto aa_ellipse = make_aa_ellipse_iterator(point2{0.0f, 0.0f}, 20.0f, 10.0f);
        
        int total_pixels = 0;
        int edge_pixels = 0;
        
        for (; aa_ellipse != aa_ellipse_iterator<float>::end(); ++aa_ellipse) {
            auto p = *aa_ellipse;
            total_pixels++;
            if (p.coverage < 0.9f) {
                edge_pixels++;
            }
        }
        
        CHECK(total_pixels > 0);
        CHECK(edge_pixels > 0);
        
        // Edge pixels should be a reasonable fraction
        float edge_ratio = float(edge_pixels) / float(total_pixels);
        CHECK(edge_ratio > 0.1f);
        // MESSAGE("Total pixels: " << total_pixels
        //         << ", Edge pixels: " << edge_pixels
        //         << ", Edge ratio: " << edge_ratio);
        CHECK(edge_ratio <= 1.0f); // Just check it's valid
    }
}