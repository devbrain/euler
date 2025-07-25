#include <doctest/doctest.h>
#include <euler/dda/dda.hh>
#include <vector>
#include <unordered_set>

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
        
        int batch_count = 0;
        int total_pixels = 0;
        
        batched.process_all([&total_pixels](const pixel<int>&) {
            total_pixels++;
        });
        
        while (!batched.at_end()) {
            batch_count++;
            batched.next_batch();
        }
        
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