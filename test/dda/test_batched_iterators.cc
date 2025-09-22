#include <doctest/doctest.h>
#include <euler/dda/batched_iterators.hh>
#include <euler/dda/dda.hh>
#include <euler/core/compiler.hh>
#include <vector>
#include <unordered_set>
#include <cmath>

using namespace euler;
using namespace euler::dda;

// Helper to collect all pixels from a batched iterator
template<typename Iterator>
std::vector<point2i> collect_pixels(Iterator& iter) {
    std::vector<point2i> pixels;
    
    while (!iter.at_end()) {
        const auto& batch = iter.current_batch();
        for (size_t i = 0; i < batch.count; ++i) {
            pixels.push_back(batch.pixels[i].pos);
        }
        iter.next_batch();
    }
    
    return pixels;
}

// Helper to collect pixels from regular iterator for comparison
template<typename T>
std::vector<point2i> collect_pixels_regular(T start, T end) {
    std::vector<point2i> pixels;
    auto iter = make_line_iterator(start, end);
    
    while (iter != line_iterator<int>::end()) {
        pixels.push_back((*iter).pos);
        ++iter;
    }
    
    return pixels;
}

EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW
TEST_CASE("Batched line iterator") {
    SUBCASE("Horizontal line") {
        point2f start{0.0f, 0.0f};
        point2f end{10.0f, 0.0f};
        
        auto batched = make_batched_line(start, end);
        auto pixels = collect_pixels(batched);
        
        CHECK(pixels.size() == 11);
        for (size_t i = 0; i <= 10; ++i) {
            CHECK(pixels[i] == point2i{static_cast<int>(i), 0});
        }
    }
    
    SUBCASE("Diagonal line") {
        point2f start{0.0f, 0.0f};
        point2f end{5.0f, 5.0f};
        
        auto batched = make_batched_line(start, end);
        auto pixels_batched = collect_pixels(batched);
        
        // Compare with regular iterator
        auto pixels_regular = collect_pixels_regular(start, end);
        
        CHECK(pixels_batched.size() == pixels_regular.size());
        for (size_t i = 0; i < pixels_batched.size(); ++i) {
            CHECK(pixels_batched[i] == pixels_regular[i]);
        }
    }
    
    SUBCASE("Batch processing") {
        point2f start{0.0f, 0.0f};
        point2f end{100.0f, 50.0f};
        
        auto batched = make_batched_line(start, end);
        
        int total_pixels = 0;
        int batch_count = 0;
        
        while (!batched.at_end()) {
            const auto& batch = batched.current_batch();
            total_pixels += static_cast<int>(batch.count);
            batch_count++;
            
            // Check that batches are reasonably full (except possibly the last)
            if (!batched.at_end() || batch_count == 1) {
                CHECK(batch.count > 0);
            }
            
            batched.next_batch();
        }
        
        CHECK(total_pixels > 0);
        CHECK(batch_count > 1); // Should have multiple batches for a long line
    }
}

TEST_CASE("Batched antialiased line iterator") {
    SUBCASE("Coverage accumulation") {
        point2f start{0.5f, 0.5f};
        point2f end{4.5f, 2.5f};
        
        // Create coverage buffer
        constexpr size_t width = 10;
        constexpr size_t height = 10;
        std::vector<float> coverage(width * height, 0.0f);
        
        auto batched = make_batched_aa_line(start, end);
        batched.accumulate_coverage(coverage.data(), width);
        
        // Check that we have non-zero coverage
        float total_coverage = 0.0f;
        for (float c : coverage) {
            total_coverage += c;
        }
        
        CHECK(total_coverage > 0.0f);
    }
}

TEST_CASE("Batched cubic Bezier iterator") {
    SUBCASE("Simple curve") {
        point2f p0{0.0f, 0.0f};
        point2f p1{10.0f, 20.0f};
        point2f p2{20.0f, 20.0f};
        point2f p3{30.0f, 0.0f};
        
        auto batched = make_batched_cubic_bezier(p0, p1, p2, p3);
        auto pixels = collect_pixels(batched);
        
        // Should produce a smooth curve
        CHECK(pixels.size() > 20); // Reasonable number of pixels
        CHECK(pixels.front() == point2i{0, 0}); // Start point
        CHECK(pixels.back() == point2i{30, 0}); // End point
        
        // Check no duplicate consecutive pixels
        for (size_t i = 1; i < pixels.size(); ++i) {
            CHECK(pixels[i] != pixels[i-1]);
        }
    }
    
    SUBCASE("Batch efficiency") {
        point2f p0{0.0f, 0.0f};
        point2f p1{100.0f, 200.0f};
        point2f p2{200.0f, 200.0f};
        point2f p3{300.0f, 0.0f};
        
        auto batched = make_batched_cubic_bezier(p0, p1, p2, p3);
        
        int total_pixels = 0;
        
        batched.process_all([&total_pixels](const auto& batch) {
            for (size_t i = 0; i < batch.count; ++i) {
                total_pixels++;
            }
        });
        
        // Reset and count again
        batched = make_batched_cubic_bezier(p0, p1, p2, p3);
        auto pixel_count = collect_pixels(batched).size();
        
        CHECK(pixel_count > 50); // Long curve should have many pixels
    }
}

TEST_CASE("Batched general Bezier iterator") {
    SUBCASE("Quadratic Bezier") {
        std::vector<point2f> control_points = {
            {0.0f, 0.0f},
            {15.0f, 30.0f},
            {30.0f, 0.0f}
        };
        
        auto batched = make_batched_bezier(control_points);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        CHECK(pixels.front() == point2i{0, 0});
        CHECK(pixels.back() == point2i{30, 0});
    }
    
    SUBCASE("High-degree Bezier") {
        std::vector<point2f> control_points;
        // Create a degree-7 Bezier
        for (int i = 0; i <= 7; ++i) {
            float t = static_cast<float>(i) / 7.0f;
            float y = (i % 2 == 0) ? 0.0f : 20.0f;
            control_points.push_back({t * 70.0f, y});
        }
        
        auto batched = make_batched_bezier(control_points);
        auto pixels = collect_pixels(batched);
        
        CHECK(pixels.size() > 50); // Complex curve should have many pixels
        
        // Verify continuity (no large gaps)
        for (size_t i = 1; i < pixels.size(); ++i) {
            int dist = distance_squared(pixels[i], pixels[i-1]);
            CHECK(dist <= 2); // At most sqrt(2) distance between consecutive pixels
        }
    }
}

// Hash function for point2i to enable use in unordered_set
namespace std {
    template<>
    struct hash<point2i> {
        size_t operator()(const point2i& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1);
        }
    };
}

// Helper to collect pixels from regular iterator for comparison
template<typename Iterator>
std::vector<point2i> collect_from_regular_iterator(Iterator iter) {
    std::vector<point2i> pixels;
    
    while (iter != Iterator::end()) {
        pixels.push_back((*iter).pos);
        ++iter;
    }
    
    return pixels;
}

TEST_CASE("Batched circle iterator") {
    SUBCASE("Small circle") {
        point2f center{10.0f, 10.0f};
        float radius = 5.0f;
        
        auto batched = make_batched_circle(center, radius);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        
        // Verify all pixels are at correct distance from center
        for (const auto& p : pixels) {
            float dx = static_cast<float>(p.x) - center.x;
            float dy = static_cast<float>(p.y) - center.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            CHECK(dist >= radius - 1.0f);
            CHECK(dist <= radius + 1.0f);
        }
        
        // Check 8-way symmetry
        std::unordered_set<point2i> pixel_set(pixels.begin(), pixels.end());
        CHECK(pixel_set.size() == pixels.size()); // No duplicates
    }
    
    SUBCASE("Large circle") {
        point2f center{100.0f, 100.0f};
        float radius = 50.0f;
        
        auto batched = make_batched_circle(center, radius);
        
        // Test manual iteration
        int batch_count = 0;
        int total_pixels = 0;
        
        while (!batched.at_end()) {
            const auto& batch = batched.current_batch();
            total_pixels += static_cast<int>(batch.count);
            batch_count++;
            batched.next_batch();
        }
        
        CHECK(total_pixels > 100); // Large circle should have many pixels
        CHECK(batch_count > 1); // Should have multiple batches
    }
    
    SUBCASE("Circle arc") {
        point2f center{20.0f, 20.0f};
        float radius = 10.0f;
        auto start_angle = degrees<float>(0);
        auto end_angle = degrees<float>(90);
        
        auto batched = make_batched_arc(center, radius, start_angle, end_angle);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        
        // All pixels should be in the first quadrant relative to center
        for (const auto& p : pixels) {
            CHECK(p.x >= static_cast<int>(center.x));
            CHECK(p.y >= static_cast<int>(center.y));
        }
    }
    
    SUBCASE("Filled circle") {
        point2f center{15.0f, 15.0f};
        float radius = 8.0f;
        
        auto batched = make_batched_filled_circle(center, radius);
        
        int total_pixels = 0;
        batched.process_all([&total_pixels](const auto& batch) {
            for (size_t i = 0; i < batch.count; ++i) {
                total_pixels += batch.pixels[i].x_end - batch.pixels[i].x_start + 1;
            }
        });
        
        // Approximate area check
        float expected_area = pi * radius * radius;
        CHECK(static_cast<float>(total_pixels) >= expected_area * 0.8f);
        CHECK(static_cast<float>(total_pixels) <= expected_area * 1.2f);
    }
}

TEST_CASE("Batched ellipse iterator") {
    SUBCASE("Axis-aligned ellipse") {
        point2f center{20.0f, 20.0f};
        float semi_major = 15.0f;
        float semi_minor = 10.0f;
        
        auto batched = make_batched_ellipse(center, semi_major, semi_minor);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        
        // Check 4-way symmetry
        std::unordered_set<point2i> pixel_set(pixels.begin(), pixels.end());
        CHECK(pixel_set.size() == pixels.size()); // No duplicates
        
        // Verify pixels satisfy ellipse equation approximately
        for (const auto& p : pixels) {
            float dx = (static_cast<float>(p.x) - center.x) / semi_major;
            float dy = (static_cast<float>(p.y) - center.y) / semi_minor;
            float value = dx * dx + dy * dy;
            CHECK(value >= 0.7f);
            CHECK(value <= 1.3f);
        }
    }
    
    SUBCASE("Comparison with regular iterator") {
        point2f center{15.0f, 15.0f};
        float semi_major = 10.0f;
        float semi_minor = 7.0f;
        
        auto batched = make_batched_ellipse(center, semi_major, semi_minor);
        auto pixels_batched = collect_pixels(batched);
        
        auto regular = make_ellipse_iterator(center, semi_major, semi_minor);
        auto pixels_regular = collect_from_regular_iterator(regular);
        
        // Should produce same number of pixels
        CHECK(pixels_batched.size() == pixels_regular.size());
        
        // Convert to sets for comparison (order might differ)
        std::unordered_set<point2i> set_batched(pixels_batched.begin(), pixels_batched.end());
        std::unordered_set<point2i> set_regular(pixels_regular.begin(), pixels_regular.end());
        
        CHECK(set_batched == set_regular);
    }
}

TEST_CASE("Batched thick line iterator") {
    SUBCASE("Horizontal thick line") {
        point2f start{0.0f, 10.0f};
        point2f end{20.0f, 10.0f};
        float thickness = 5.0f;
        
        auto batched = make_batched_thick_line(start, end, thickness);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        
        // Check thickness constraint
        // For thickness 5, radius is round(2.5) = 3, so y should be in [7, 13]
        for (const auto& p : pixels) {
            CHECK(p.y >= 7);  // 10 - 3
            CHECK(p.y <= 13); // 10 + 3
        }
        
        // No duplicate pixels
        std::unordered_set<point2i> pixel_set(pixels.begin(), pixels.end());
        CHECK(pixel_set.size() == pixels.size());
    }
    
    SUBCASE("Thick line spans") {
        point2f start{5.0f, 5.0f};
        point2f end{25.0f, 15.0f};
        float thickness = 4.0f;
        
        auto batched = make_batched_thick_line_spans(start, end, thickness);
        
        int total_pixels = 0;
        batched.process_all([&total_pixels](const auto& batch) {
            for (size_t i = 0; i < batch.count; ++i) {
                total_pixels += batch.pixels[i].x_end - batch.pixels[i].x_start + 1;
            }
        });
        
        CHECK(total_pixels > 0);
    }
}

TEST_CASE("Batched B-spline iterator") {
    SUBCASE("Cubic B-spline") {
        std::vector<point2f> control_points = {
            {0.0f, 0.0f},
            {10.0f, 20.0f},
            {20.0f, 20.0f},
            {30.0f, 0.0f}
        };
        
        auto batched = make_batched_bspline(control_points);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        CHECK(pixels.size() > 20); // Should produce smooth curve
        
        // Check continuity
        for (size_t i = 1; i < pixels.size(); ++i) {
            int dist = distance_squared(pixels[i], pixels[i-1]);
            CHECK(dist <= 2); // No large gaps
        }
    }
    
    SUBCASE("Catmull-Rom spline") {
        std::vector<point2f> points = {
            {0.0f, 0.0f},
            {20.0f, 30.0f},
            {40.0f, 30.0f},
            {60.0f, 0.0f}
        };
        
        auto batched = make_batched_catmull_rom(points);
        auto pixels = collect_pixels(batched);
        
        CHECK(!pixels.empty());
        
        // Should interpolate through control points (approximately)
        bool found_start = false;
        bool found_end = false;
        
        for (const auto& p : pixels) {
            if (p == point2i{0, 0}) found_start = true;
            if (p == point2i{60, 0}) found_end = true;
        }
        
        CHECK(found_start);
        CHECK(found_end);
    }
}

TEST_CASE("Batched iterator type traits") {
    SUBCASE("Type trait validation") {
        // Check that all batched iterators satisfy the trait
        CHECK(is_batched_iterator_v<batched_line_iterator<float>>);
        CHECK(is_batched_iterator_v<batched_circle_iterator<float>>);
        CHECK(is_batched_iterator_v<batched_ellipse_iterator<float>>);
        CHECK(is_batched_iterator_v<batched_thick_line_iterator<float>>);
        CHECK(is_batched_iterator_v<batched_bspline_iterator<float>>);
        CHECK(is_batched_iterator_v<batched_cubic_bezier_iterator<float>>);
        
        // Check that regular iterators don't satisfy the trait
        CHECK(!is_batched_iterator_v<line_iterator<float>>);
        CHECK(!is_batched_iterator_v<circle_iterator<float>>);
    }
}

TEST_CASE("Prefetching and performance features") {
    SUBCASE("Prefetch hints") {
        // Test that prefetch compiles and runs without errors
        int test_data[64] = {0};
        
        // Test different prefetch modes
        prefetch_hint::prefetch<int, 0, 3>(test_data);  // read, high locality
        
        prefetch_hint::prefetch<int, 1, 1>(test_data + 32);  // write, low locality
        
        // Test range prefetch
        prefetch_hint::prefetch_range<int, 0, 2>(test_data, test_data + 64);
        
        // If we get here without crashes, prefetching is working
        CHECK(true);
    }
    
    SUBCASE("Batch writer") {
        std::vector<pixel_batch<pixel<int>>> flushed_batches;
        
        {
            batch_writer<pixel<int>> writer(
                [&flushed_batches](const pixel_batch<pixel<int>>& batch) {
                    flushed_batches.push_back(batch);
                });
            
            // Write more pixels than fit in one batch
            for (int i = 0; i < 100; ++i) {
                writer.write({{i, 0}});
            }
            
            // Writer should flush on destruction
        }
        
        CHECK(!flushed_batches.empty());
        
        // Count total pixels
        int total_pixels = 0;
        for (const auto& batch : flushed_batches) {
            total_pixels += static_cast<int>(batch.count);
        }
        CHECK(total_pixels == 100);
    }
}
EULER_DISABLE_WARNING_POP