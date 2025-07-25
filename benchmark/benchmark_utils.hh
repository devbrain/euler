/**
 * @file benchmark_utils.hh
 * @brief Utilities for performance benchmarking
 */
#pragma once

#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>
#include <atomic>

namespace euler::benchmark {

using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;

/**
 * @brief Timer for measuring execution time
 */
class timer {
private:
    clock_type::time_point start_time_;
    
public:
    timer() : start_time_(clock_type::now()) {}
    
    void reset() {
        start_time_ = clock_type::now();
    }
    
    double elapsed() const {
        auto end_time = clock_type::now();
        duration_type diff = end_time - start_time_;
        return diff.count();
    }
};

/**
 * @brief Statistics for a series of measurements
 */
struct benchmark_stats {
    double min;
    double max;
    double mean;
    double median;
    double stddev;
    size_t iterations;
};

/**
 * @brief Calculate statistics from a series of timings
 */
benchmark_stats calculate_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / times.size());
    
    return {
        times.front(),
        times.back(),
        mean,
        times[times.size() / 2],
        stddev,
        times.size()
    };
}

/**
 * @brief Print benchmark statistics
 */
void print_stats(const std::string& name, const benchmark_stats& stats) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Min:    " << stats.min * 1000 << " ms" << std::endl;
    std::cout << "  Max:    " << stats.max * 1000 << " ms" << std::endl;
    std::cout << "  Mean:   " << stats.mean * 1000 << " ms" << std::endl;
    std::cout << "  Median: " << stats.median * 1000 << " ms" << std::endl;
    std::cout << "  StdDev: " << stats.stddev * 1000 << " ms" << std::endl;
    std::cout << "  Iterations: " << stats.iterations << std::endl;
}

/**
 * @brief Run a benchmark with warmup and multiple iterations
 */
template<typename Func>
benchmark_stats run_benchmark(const std::string& name, 
                             Func&& func,
                             size_t warmup_iterations = 10,
                             size_t test_iterations = 100,
                             bool verbose = true) {
    if (verbose) {
        std::cout << "Running benchmark: " << name << std::endl;
    }
    
    // Warmup
    for (size_t i = 0; i < warmup_iterations; ++i) {
        func();
    }
    
    // Actual measurements
    std::vector<double> times;
    times.reserve(test_iterations);
    
    for (size_t i = 0; i < test_iterations; ++i) {
        timer t;
        func();
        times.push_back(t.elapsed());
    }
    
    auto stats = calculate_stats(times);
    
    if (verbose) {
        print_stats(name, stats);
    }
    
    return stats;
}

/**
 * @brief Compare two benchmarks and print results
 */
void compare_benchmarks(const std::string& name1, const benchmark_stats& stats1,
                       const std::string& name2, const benchmark_stats& stats2) {
    std::cout << "\nComparison: " << name1 << " vs " << name2 << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    double speedup = stats1.mean / stats2.mean;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << name1 << " mean: " << stats1.mean * 1000 << " ms" << std::endl;
    std::cout << name2 << " mean: " << stats2.mean * 1000 << " ms" << std::endl;
    
    if (speedup > 1.0) {
        std::cout << name2 << " is " << speedup << "x faster" << std::endl;
    } else {
        std::cout << name1 << " is " << (1.0 / speedup) << "x faster" << std::endl;
    }
}

/**
 * @brief Force prevent optimization of a value
 */
template<typename T>
void do_not_optimize(T&& value) {
    // Use inline assembly to prevent optimization
    #if defined(__clang__) || defined(__GNUC__)
        asm volatile("" : : "g"(value) : "memory");
    #else
        volatile T dummy = value;
        (void)dummy;
    #endif
}

/**
 * @brief Memory barrier to prevent reordering
 */
inline void compiler_barrier() {
    #if defined(__clang__) || defined(__GNUC__)
        asm volatile("" ::: "memory");
    #else
        std::atomic_thread_fence(std::memory_order_seq_cst);
    #endif
}

} // namespace euler::benchmark