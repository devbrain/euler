#include <doctest/doctest.h>
#include <euler/dda/circle_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <vector>
#include <unordered_set>
#include <map>
#include <algorithm>
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

TEST_CASE("Circle iterator basic functionality") {
    SUBCASE("Small circle") {
        std::unordered_set<point2i> pixels;
        
        for (auto p : circle_pixels(point2i{0, 0}, 5)) {
            pixels.insert(p.pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check all pixels are approximately at radius 5
        for (const auto& p : pixels) {
            float dist = std::sqrt(float(p.x * p.x + p.y * p.y));
            CHECK(dist >= 4.0f);
            CHECK(dist <= 6.0f);
        }
        
        // Should have 8-way symmetry
        for (const auto& p : pixels) {
            CHECK(pixels.count(point2i{-p.x, p.y}) == 1);
            CHECK(pixels.count(point2i{p.x, -p.y}) == 1);
            CHECK(pixels.count(point2i{-p.x, -p.y}) == 1);
        }
    }
    
    SUBCASE("Unit circle") {
        std::vector<point2i> pixels;
        std::unordered_set<point2i> unique_pixels;
        
        auto circle = make_circle_iterator(point2{0.0f, 0.0f}, 1.0f);
        for (; circle != circle_iterator<float>::end(); ++circle) {
            pixels.push_back((*circle).pos);
            unique_pixels.insert((*circle).pos);
        }

        // Unit circle (radius 1) using midpoint algorithm should generate 4 unique pixels
        // at the cardinal points: (±1,0) and (0,±1)
        CHECK(unique_pixels.size() == 4);
        
        // Verify the pixels are at the expected positions
        CHECK(unique_pixels.count(point2i{1, 0}) == 1);
        CHECK(unique_pixels.count(point2i{-1, 0}) == 1);
        CHECK(unique_pixels.count(point2i{0, 1}) == 1);
        CHECK(unique_pixels.count(point2i{0, -1}) == 1);
    }
    
    SUBCASE("Circle with offset center") {
        std::unordered_set<point2i> pixels;
        
        point2i center{10, 20};
        int radius = 8;
        
        for (auto p : circle_pixels(center, radius)) {
            pixels.insert(p.pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check pixels are around the offset center
        for (const auto& p : pixels) {
            float dx = float(p.x - center.x);
            float dy = float(p.y - center.y);
            float dist = std::sqrt(dx * dx + dy * dy);
            CHECK(dist >= float(radius - 1));
            CHECK(dist <= float(radius + 1));
        }
    }
    
    SUBCASE("Zero radius circle") {
        std::vector<point2i> pixels;
        
        for (auto p : circle_pixels(point2i{5, 5}, 0)) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 1);
        CHECK(pixels[0] == point2i{5, 5});
    }
}

TEST_CASE("Arc iterator") {
    SUBCASE("Quarter arc with degrees") {
        std::vector<point2i> pixels;
        
        auto arc = make_arc_iterator(point2{0.0f, 0.0f}, 10.0f,
                                    degree<float>(0), degree<float>(90));
        
        for (; arc != circle_iterator<float>::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // All pixels should be in first quadrant
        for (const auto& p : pixels) {
            CHECK(p.x >= 0);
            CHECK(p.y >= 0);
        }
    }
    
    SUBCASE("Arc crossing 0 degrees") {
        std::unordered_set<point2i> pixels;
        
        auto arc = make_arc_iterator(point2{0.0f, 0.0f}, 8.0f,
                                    degree<float>(270), degree<float>(90));
        
        for (; arc != circle_iterator<float>::end(); ++arc) {
            pixels.insert((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have pixels in quadrants 1 and 4
        bool has_q1 = false, has_q4 = false;
        for (const auto& p : pixels) {
            if (p.x > 0 && p.y > 0) has_q1 = true;
            if (p.x > 0 && p.y < 0) has_q4 = true;
        }
        CHECK(has_q1);
        CHECK(has_q4);
    }
    
    SUBCASE("Arc with radians") {
        std::vector<point2i> pixels;
        
        auto arc = make_arc_iterator(point2{0.0f, 0.0f}, 10.0f,
                                    radian<float>(0), radian<float>(pi/2));
        
        for (; arc != circle_iterator<float>::end(); ++arc) {
            pixels.push_back((*arc).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should be similar to degree version
        for (const auto& p : pixels) {
            CHECK(p.x >= 0);
            CHECK(p.y >= 0);
        }
    }
}

TEST_CASE("Filled circle iterator") {
    SUBCASE("Small filled circle") {
        std::unordered_set<point2i> pixels;
        
        auto filled = make_filled_circle_iterator(point2{0.0f, 0.0f}, 5.0f);
        
        for (; filled != filled_circle_iterator<float>::end(); ++filled) {
            auto s = *filled;
            for (int x = s.x_start; x <= s.x_end; ++x) {
                pixels.insert(point2i{x, s.y});
            }
        }
        
        CHECK(!pixels.empty());
        
        // Check all pixels are within radius
        for (const auto& p : pixels) {
            float dist = std::sqrt(float(p.x * p.x + p.y * p.y));
            CHECK(dist <= 5.5f);
        }
        
        // Check we have a solid fill (no holes)
        // For each scanline, check continuity
        std::map<int, std::vector<int>> scanlines;
        for (const auto& p : pixels) {
            scanlines[p.y].push_back(p.x);
        }
        
        for (auto& [y, xs] : scanlines) {
            std::sort(xs.begin(), xs.end());
            // Check no gaps
            for (size_t i = 1; i < xs.size(); ++i) {
                CHECK(xs[i] - xs[i-1] <= 1);
            }
        }
    }
    
    SUBCASE("Filled circle spans are symmetric") {
        std::vector<span> spans;
        
        auto filled = make_filled_circle_iterator(point2{0.0f, 0.0f}, 10.0f);
        for (; filled != filled_circle_iterator<float>::end(); ++filled) {
            spans.push_back(*filled);
        }
        
        CHECK(!spans.empty());
        
        // Each span should be centered at x=0 for a circle at origin
        for (const auto& s : spans) {
            CHECK(s.x_start == -s.x_end);
        }
    }
}

TEST_CASE("Circle iterator edge cases") {
    SUBCASE("Large radius") {
        std::unordered_set<point2i> pixels;
        
        auto circle = make_circle_iterator(point2{0.0f, 0.0f}, 100.0f);
        int count = 0;
        for (; circle != circle_iterator<float>::end(); ++circle) {
            pixels.insert((*circle).pos);
            count++;
            if (count > 1000) break; // Safety limit
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.size() > 300); // Should have many pixels for large circle
    }
    
    SUBCASE("Floating point precision") {
        // Test with non-integer center and radius
        std::vector<point2i> pixels;
        
        point2f float_center{0.3f, 0.7f};
        float float_radius = 5.4f;
        
        // The algorithm rounds center and radius to integers
        point2i int_center = round(float_center);
        int int_radius = static_cast<int>(std::round(float_radius));
        
        auto circle = make_circle_iterator(float_center, float_radius);
        for (; circle != circle_iterator<float>::end(); ++circle) {
            pixels.push_back((*circle).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Check pixels are at appropriate distance from the INTEGER center
        for (const auto& p : pixels) {
            float dx = static_cast<float>(p.x - int_center.x);
            float dy = static_cast<float>(p.y - int_center.y);
            float dist = std::sqrt(dx * dx + dy * dy);
            // Allow some tolerance for discrete pixel positions
            CHECK(dist >= static_cast<float>(int_radius) - 1.0f);
            CHECK(dist <= static_cast<float>(int_radius) + 1.0f);
        }
    }
}