/**
 * @file benchmark_dda.cc
 * @brief Benchmarks for DDA operations with/without SIMD and batching
 */

#include "benchmark_utils.hh"
#include <euler/dda/dda.hh>
#include <euler/coordinates/point2.hh>
#include <random>
#include <vector>
#include <cstring>

using namespace euler;
using namespace euler::dda;
using namespace euler::benchmark;

// Test parameters
constexpr int IMAGE_SIZE = 1024;
constexpr size_t SHORT_LINES = 1000;
constexpr size_t MEDIUM_LINES = 100;
constexpr size_t LONG_LINES = 10;

// Generate random points
template<typename T>
std::vector<point2<T>> generate_random_points(size_t count, T max_coord) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0, max_coord);
    
    std::vector<point2<T>> points;
    points.reserve(count * 2); // start and end points
    
    for (size_t i = 0; i < count * 2; ++i) {
        points.emplace_back(dist(gen), dist(gen));
    }
    
    return points;
}

// Generate bezier control points
template<typename T>
std::vector<std::vector<point2<T>>> generate_bezier_curves(size_t count, T max_coord) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0, max_coord);
    
    std::vector<std::vector<point2<T>>> curves;
    curves.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        std::vector<point2<T>> control_points;
        control_points.reserve(4);
        
        // Generate 4 control points for cubic bezier
        for (int j = 0; j < 4; ++j) {
            control_points.emplace_back(dist(gen), dist(gen));
        }
        
        curves.push_back(std::move(control_points));
    }
    
    return curves;
}

// Benchmark line rasterization
template<typename T>
void benchmark_line_rasterization(size_t line_count, T line_length) {
    std::cout << "\n=== Line Rasterization (count=" << line_count 
              << ", avg_length=" << line_length << ") ===" << std::endl;
    
    // Generate test lines
    auto points = generate_random_points<T>(line_count, IMAGE_SIZE - line_length);
    std::vector<pixel<int>> pixel_buffer;
    pixel_buffer.reserve(static_cast<size_t>(line_count * line_length * 1.5));
    
    // Regular iterator version
    auto regular_bench = run_benchmark("Regular line iterator", [&]() {
        pixel_buffer.clear();
        
        for (size_t i = 0; i < line_count; ++i) {
            auto start = points[i * 2];
            auto end = points[i * 2] + euler::vector<T, 2>{line_length, line_length * 0.7f};
            
            for (auto pixel : line_pixels(start, end)) {
                pixel_buffer.push_back(pixel);
            }
        }
        
        do_not_optimize(pixel_buffer.data());
    });
    
    // Batched iterator version
    std::vector<pixel_batch<pixel<int>>> batch_buffer;
    batch_buffer.reserve(static_cast<size_t>(line_count * line_length / 8));
    
    auto batched_bench = run_benchmark("Batched line iterator", [&]() {
        pixel_buffer.clear();
        batch_buffer.clear();
        
        for (size_t i = 0; i < line_count; ++i) {
            auto start = points[i * 2];
            auto end = points[i * 2] + euler::vector<T, 2>{line_length, line_length * 0.7f};
            
            auto iter = make_batched_line(start, end);
            while (!iter.at_end()) {
                const auto& batch = iter.current_batch();
                batch_buffer.push_back(batch);
                
                // Also collect individual pixels for comparison
                for (size_t j = 0; j < batch.count; ++j) {
                    pixel_buffer.push_back(batch.pixels[j]);
                }
                
                iter.next_batch();
            }
        }
        
        do_not_optimize(pixel_buffer.data());
        do_not_optimize(batch_buffer.data());
    });
    
    compare_benchmarks("Regular", regular_bench, "Batched", batched_bench);
}

// Benchmark antialiased line rasterization
template<typename T>
void benchmark_aa_line_rasterization(size_t line_count, T line_length) {
    std::cout << "\n=== Antialiased Line Rasterization (count=" << line_count 
              << ", avg_length=" << line_length << ") ===" << std::endl;
    
    auto points = generate_random_points<T>(line_count, IMAGE_SIZE - line_length);
    std::vector<aa_pixel<T>> pixel_buffer;
    pixel_buffer.reserve(static_cast<size_t>(line_count * line_length * 2));
    
    // Regular AA iterator
    auto regular_bench = run_benchmark("Regular AA line iterator", [&]() {
        pixel_buffer.clear();
        
        for (size_t i = 0; i < line_count; ++i) {
            auto start = points[i * 2];
            auto end = points[i * 2] + euler::vector<T, 2>{line_length, line_length * 0.7f};
            
            auto iter = make_aa_line_iterator(start, end);
            for (; iter != aa_line_iterator<T>::end(); ++iter) {
                pixel_buffer.push_back(*iter);
            }
        }
        
        do_not_optimize(pixel_buffer.data());
    });
    
    // Batched AA iterator with coverage accumulation
    std::vector<T> coverage_buffer(IMAGE_SIZE * IMAGE_SIZE, 0);
    
    auto batched_bench = run_benchmark("Batched AA line iterator", [&]() {
        std::memset(coverage_buffer.data(), 0, coverage_buffer.size() * sizeof(T));
        
        for (size_t i = 0; i < line_count; ++i) {
            auto start = points[i * 2];
            auto end = points[i * 2] + euler::vector<T, 2>{line_length, line_length * 0.7f};
            
            auto iter = make_batched_aa_line(start, end);
            iter.accumulate_coverage(coverage_buffer.data(), IMAGE_SIZE);
        }
        
        do_not_optimize(coverage_buffer.data());
    });
    
    compare_benchmarks("Regular AA", regular_bench, "Batched AA", batched_bench);
}

// Benchmark cubic bezier rasterization
template<typename T>
void benchmark_bezier_rasterization(size_t curve_count) {
    std::cout << "\n=== Cubic Bezier Rasterization (count=" << curve_count << ") ===" << std::endl;
    
    auto curves = generate_bezier_curves<T>(curve_count, static_cast<T>(IMAGE_SIZE));
    std::vector<pixel<int>> pixel_buffer;
    pixel_buffer.reserve(curve_count * 200); // Estimate ~200 pixels per curve
    
    // Regular bezier iterator
    auto regular_bench = run_benchmark("Regular bezier iterator", [&]() {
        pixel_buffer.clear();
        
        for (const auto& control_points : curves) {
            T tolerance = 0.5;
            
            auto iter = make_cubic_bezier(
                control_points[0], control_points[1], 
                control_points[2], control_points[3], tolerance);
            for (; iter != cubic_bezier_iterator<T>::end(); ++iter) {
                pixel_buffer.push_back(*iter);
            }
        }
        
        do_not_optimize(pixel_buffer.data());
    });
    
    // Batched bezier iterator
    std::vector<pixel_batch<pixel<int>>> batch_buffer;
    batch_buffer.reserve(curve_count * 20);
    
    auto batched_bench = run_benchmark("Batched bezier iterator", [&]() {
        pixel_buffer.clear();
        batch_buffer.clear();
        
        for (const auto& control_points : curves) {
            T tolerance = 0.5;
            
            auto iter = make_batched_cubic_bezier(
                control_points[0], control_points[1],
                control_points[2], control_points[3], tolerance);
            
            while (!iter.at_end()) {
                const auto& batch = iter.current_batch();
                batch_buffer.push_back(batch);
                
                for (size_t j = 0; j < batch.count; ++j) {
                    pixel_buffer.push_back(batch.pixels[j]);
                }
                
                iter.next_batch();
            }
        }
        
        do_not_optimize(pixel_buffer.data());
        do_not_optimize(batch_buffer.data());
    });
    
    compare_benchmarks("Regular", regular_bench, "Batched", batched_bench);
}

// Benchmark pixel processing
void benchmark_pixel_processing() {
    std::cout << "\n=== Pixel Processing ===" << std::endl;
    
    // Generate a large set of pixels
    constexpr size_t pixel_count = 100000;
    std::vector<pixel<int>> pixels;
    pixels.reserve(pixel_count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, IMAGE_SIZE - 1);
    
    for (size_t i = 0; i < pixel_count; ++i) {
        pixels.push_back({point2i{dist(gen), dist(gen)}});
    }
    
    // Framebuffer for rendering
    std::vector<uint32_t> framebuffer(IMAGE_SIZE * IMAGE_SIZE, 0);
    
    // Individual pixel processing
    auto individual_bench = run_benchmark("Individual pixel processing", [&]() {
        std::memset(framebuffer.data(), 0, framebuffer.size() * sizeof(uint32_t));
        
        for (const auto& pixel : pixels) {
            int idx = pixel.pos.y * IMAGE_SIZE + pixel.pos.x;
            framebuffer[idx] = 0xFFFFFFFF; // White pixel
        }
        
        do_not_optimize(framebuffer.data());
    });
    
    // Batch processing
    auto batch_bench = run_benchmark("Batch pixel processing", [&]() {
        std::memset(framebuffer.data(), 0, framebuffer.size() * sizeof(uint32_t));
        
        // Process pixels in batches
        pixel_batch<pixel<int>> batch;
        
        for (const auto& pixel : pixels) {
            if (!batch.add(pixel)) {
                // Process full batch
                for (size_t i = 0; i < batch.count; ++i) {
                    int idx = batch.pixels[i].pos.y * IMAGE_SIZE + batch.pixels[i].pos.x;
                    framebuffer[idx] = 0xFFFFFFFF;
                }
                
                batch.clear();
                batch.add(pixel);
            }
        }
        
        // Process remaining pixels
        for (size_t i = 0; i < batch.count; ++i) {
            int idx = batch.pixels[i].pos.y * IMAGE_SIZE + batch.pixels[i].pos.x;
            framebuffer[idx] = 0xFFFFFFFF;
        }
        
        do_not_optimize(framebuffer.data());
    });
    
    compare_benchmarks("Individual", individual_bench, "Batch", batch_bench);
}

// Benchmark span operations
void benchmark_span_operations() {
    std::cout << "\n=== Span Operations ===" << std::endl;
    
    constexpr size_t span_count = 10000;
    std::vector<span> spans;
    spans.reserve(span_count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> y_dist(0, IMAGE_SIZE - 1);
    std::uniform_int_distribution<int> x_dist(0, IMAGE_SIZE - 100);
    std::uniform_int_distribution<int> len_dist(10, 100);
    
    for (size_t i = 0; i < span_count; ++i) {
        int y = y_dist(gen);
        int x = x_dist(gen);
        int len = len_dist(gen);
        spans.push_back({y, x, x + len});
    }
    
    std::vector<uint32_t> framebuffer(IMAGE_SIZE * IMAGE_SIZE, 0);
    
    // Regular span filling
    auto regular_bench = run_benchmark("Regular span filling", [&]() {
        std::memset(framebuffer.data(), 0, framebuffer.size() * sizeof(uint32_t));
        
        for (const auto& span : spans) {
            // Generate pixels from span manually
            for (int x = span.x_start; x <= span.x_end; ++x) {
                int idx = span.y * IMAGE_SIZE + x;
                framebuffer[idx] = 0xFFFFFFFF;
            }
        }
        
        do_not_optimize(framebuffer.data());
    });
    
    // Optimized span filling (direct memory writes)
    auto optimized_bench = run_benchmark("Optimized span filling", [&]() {
        std::memset(framebuffer.data(), 0, framebuffer.size() * sizeof(uint32_t));
        
        for (const auto& span : spans) {
            int y = span.y;
            int x_start = span.x_start;
            int x_end = span.x_end;
            
            uint32_t* row = &framebuffer[y * IMAGE_SIZE];
            
            // Fill span with memset for better performance
            std::memset(&row[x_start], 0xFF, (x_end - x_start) * sizeof(uint32_t));
        }
        
        do_not_optimize(framebuffer.data());
    });
    
    compare_benchmarks("Regular", regular_bench, "Optimized", optimized_bench);
}

int main() {
    std::cout << "Euler DDA Benchmarks" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Line rasterization benchmarks
    benchmark_line_rasterization<float>(SHORT_LINES, 50.0f);
    benchmark_line_rasterization<float>(MEDIUM_LINES, 200.0f);
    benchmark_line_rasterization<float>(LONG_LINES, 800.0f);
    
    // Antialiased line benchmarks
    benchmark_aa_line_rasterization<float>(SHORT_LINES, 50.0f);
    benchmark_aa_line_rasterization<float>(MEDIUM_LINES, 200.0f);
    
    // Bezier curve benchmarks
    benchmark_bezier_rasterization<float>(100);
    benchmark_bezier_rasterization<float>(1000);
    
    // Pixel processing benchmarks
    benchmark_pixel_processing();
    
    // Span operation benchmarks
    benchmark_span_operations();
    
    return 0;
}