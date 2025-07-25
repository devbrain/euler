/**
 * @file benchmark_all.cc
 * @brief Combined benchmark runner that executes all benchmarks in sequence
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    std::cout << "Euler Comprehensive Benchmark Suite" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << std::endl;
    
    struct BenchmarkInfo {
        std::string name;
        std::string description;
        std::string executable;
    };
    
    std::vector<BenchmarkInfo> benchmarks = {
        {"Matrix/Vector Operations", 
         "Tests basic linear algebra operations with and without SIMD",
         "./benchmark/benchmark_matrix_vector"},
        
        {"DDA Operations", 
         "Tests line and curve rasterization with regular and batched iterators",
         "./benchmark/benchmark_dda"},
        
        {"SIMD Operations", 
         "Direct comparison of scalar vs SIMD implementations",
         "./benchmark/benchmark_simd"}
    };
    
    bool all_passed = true;
    
    for (const auto& benchmark : benchmarks) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running: " << benchmark.name << std::endl;
        std::cout << "Description: " << benchmark.description << std::endl;
        std::cout << "========================================" << std::endl;
        
        int result = std::system(benchmark.executable.c_str());
        
        if (result != 0) {
            std::cerr << "\nERROR: Benchmark '" << benchmark.name 
                     << "' failed with exit code " << result << std::endl;
            all_passed = false;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Suite Complete" << std::endl;
    
    if (all_passed) {
        std::cout << "All benchmarks completed successfully!" << std::endl;
    } else {
        std::cout << "Some benchmarks failed. See errors above." << std::endl;
        return 1;
    }
    
    // Summary recommendations
    std::cout << "\n=== Performance Recommendations ===" << std::endl;
    std::cout << "1. Enable SIMD support (xsimd) for best performance" << std::endl;
    std::cout << "2. Use batched iterators for long lines and curves" << std::endl;
    std::cout << "3. Ensure -O3 optimization is enabled in release builds" << std::endl;
    std::cout << "4. Consider -march=native for CPU-specific optimizations" << std::endl;
    
    return 0;
}