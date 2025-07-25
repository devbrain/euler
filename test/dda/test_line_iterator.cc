#include <doctest/doctest.h>
#include <euler/dda/line_iterator.hh>
#include <euler/dda/aa_line_iterator.hh>
#include <euler/dda/thick_line_iterator.hh>
#include <euler/coordinates/point2.hh>
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

TEST_CASE("Line iterator basic functionality") {
    SUBCASE("Horizontal line") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{0, 0}, point2i{5, 0})) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 6);
        CHECK(pixels[0] == point2i{0, 0});
        CHECK(pixels[5] == point2i{5, 0});
        
        // Check all pixels are on the line
        for (const auto& p : pixels) {
            CHECK(p.y == 0);
        }
    }
    
    SUBCASE("Vertical line") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{0, 0}, point2i{0, 5})) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 6);
        CHECK(pixels[0] == point2i{0, 0});
        CHECK(pixels[5] == point2i{0, 5});
        
        for (const auto& p : pixels) {
            CHECK(p.x == 0);
        }
    }
    
    SUBCASE("Diagonal line") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{0, 0}, point2i{5, 5})) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 6);
        CHECK(pixels[0] == point2i{0, 0});
        CHECK(pixels[5] == point2i{5, 5});
        
        // Perfect diagonal - x should equal y
        for (const auto& p : pixels) {
            CHECK(p.x == p.y);
        }
    }
    
    SUBCASE("Line in reverse direction") {
        std::vector<point2i> forward, reverse;
        
        for (auto p : line_pixels(point2i{0, 0}, point2i{5, 3})) {
            forward.push_back(p.pos);
        }
        
        for (auto p : line_pixels(point2i{5, 3}, point2i{0, 0})) {
            reverse.push_back(p.pos);
        }
        
        // Should visit same pixels
        CHECK(forward.size() == reverse.size());
        
        // Convert to sets for comparison
        std::unordered_set<point2i> forward_set(forward.begin(), forward.end());
        std::unordered_set<point2i> reverse_set(reverse.begin(), reverse.end());
        CHECK(forward_set == reverse_set);
    }
    
    SUBCASE("Single pixel line") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{5, 5}, point2i{5, 5})) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 1);
        CHECK(pixels[0] == point2i{5, 5});
    }
    
    SUBCASE("Floating point endpoints") {
        auto line = make_line_iterator(point2{0.3f, 0.7f}, point2{5.8f, 3.2f});
        
        int count = 0;
        for (; line != line_iterator<float>::end(); ++line) {
            count++;
            auto p = (*line).pos;
            // Check pixel is within reasonable bounds
            CHECK(p.x >= 0);
            CHECK(p.x <= 6);
            CHECK(p.y >= 0);
            CHECK(p.y <= 4);
        }
        
        CHECK(count > 0);
    }
}

TEST_CASE("Antialiased line iterator") {
    SUBCASE("Wu's algorithm coverage") {
        auto line = make_aa_line_iterator(point2{0.0f, 0.0f}, point2{10.0f, 5.0f});
        
        std::vector<aa_pixel<float>> pixels;
        for (; line != aa_line_iterator<float>::end(); ++line) {
            pixels.push_back(*line);
        }
        
        CHECK(!pixels.empty());
        
        // Check coverage values are in valid range
        for (const auto& p : pixels) {
            CHECK(p.coverage >= 0.0f);
            CHECK(p.coverage <= 1.0f);
        }
        
        // For non-axis-aligned lines, should have pixels with partial coverage
        bool has_partial_coverage = false;
        for (const auto& p : pixels) {
            if (p.coverage > 0.0f && p.coverage < 1.0f) {
                has_partial_coverage = true;
                break;
            }
        }
        CHECK(has_partial_coverage);
    }
    
    SUBCASE("Gupta-Sproull algorithm") {
        auto line = make_gupta_sproull_line_iterator(point2{0.0f, 0.0f}, point2{10.0f, 5.0f});
        
        int count = 0;
        for (; line != gupta_sproull_line_iterator<float>::end(); ++line) {
            auto p = *line;
            CHECK(p.coverage >= 0.0f);
            CHECK(p.coverage <= 1.0f);
            CHECK(p.distance >= 0.0f);
            count++;
        }
        
        CHECK(count > 0);
    }
}

TEST_CASE("Thick line iterator") {
    SUBCASE("Basic thick line") {
        std::unordered_set<point2i> pixels;
        
        auto line = make_thick_line_iterator(point2{0.0f, 0.0f}, point2{10.0f, 0.0f}, 3.0f);
        for (; line != thick_line_iterator<float>::end(); ++line) {
            pixels.insert((*line).pos);
        }
        
        CHECK(!pixels.empty());
        
        // Should have multiple rows for thickness
        int min_y = 1000, max_y = -1000;
        for (const auto& p : pixels) {
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        CHECK(max_y - min_y >= 2); // At least 3 pixels thick
    }
    
    SUBCASE("Antialiased thick line") {
        auto line = make_aa_thick_line_iterator(point2{0.0f, 0.0f}, point2{10.0f, 5.0f}, 4.0f);
        
        std::vector<aa_pixel<float>> pixels;
        for (; line != aa_thick_line_iterator<float>::end(); ++line) {
            pixels.push_back(*line);
        }
        
        CHECK(!pixels.empty());
        
        // Should have pixels with varying coverage at edges
        bool has_edge_pixels = false;
        for (const auto& p : pixels) {
            if (p.coverage > 0.0f && p.coverage < 1.0f) {
                has_edge_pixels = true;
                break;
            }
        }
        CHECK(has_edge_pixels);
    }
    
    SUBCASE("Thick line spans") {
        auto spans = make_thick_line_spans(point2{0.0f, 0.0f}, point2{10.0f, 0.0f}, 5.0f);
        
        std::vector<span> span_list;
        for (; spans != thick_line_span_iterator<float>::end(); ++spans) {
            span_list.push_back(*spans);
        }
        
        CHECK(!span_list.empty());
        
        // Each span should be non-empty
        for (const auto& s : span_list) {
            CHECK(s.x_end >= s.x_start);
        }
    }
}

TEST_CASE("Line iterator edge cases") {
    SUBCASE("Very steep line") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{0, 0}, point2i{1, 100})) {
            pixels.push_back(p.pos);
        }
        
        REQUIRE(pixels.size() == 101);
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{1, 100});
    }
    
    SUBCASE("Negative coordinates") {
        std::vector<point2i> pixels;
        for (auto p : line_pixels(point2i{-5, -3}, point2i{2, 1})) {
            pixels.push_back(p.pos);
        }
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{-5, -3});
        CHECK(pixels.back() == point2i{2, 1});
    }
    
    SUBCASE("Integer specialization") {
        // Test that integer specialization produces same results
        std::vector<point2i> int_pixels, float_pixels;
        
        for (auto p : line_pixels(point2i{0, 0}, point2i{10, 7})) {
            int_pixels.push_back(p.pos);
        }
        
        auto float_line = make_line_iterator(point2{0.0f, 0.0f}, point2{10.0f, 7.0f});
        for (; float_line != line_iterator<float>::end(); ++float_line) {
            float_pixels.push_back((*float_line).pos);
        }
        
        CHECK(int_pixels == float_pixels);
    }
}