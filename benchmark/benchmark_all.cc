/**
 * @file benchmark_all.cc
 * @brief Combined benchmark runner that executes all benchmarks in sequence
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

int main(int argc, char* argv[]) {
    std::cout << "Euler Comprehensive Benchmark Suite" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << std::endl;

    // Get the directory containing this executable
    std::filesystem::path exe_path;
    if (argc > 0 && argv[0] != nullptr) {
        exe_path = std::filesystem::path(argv[0]).parent_path();
        if (exe_path.empty()) {
            exe_path = ".";
        }
    } else {
        exe_path = ".";
    }

    struct BenchmarkInfo {
        std::string name;
        std::string description;
        std::string executable;
    };

#ifdef _WIN32
    const std::string exe_suffix = ".exe";
#else
    const std::string exe_suffix = "";
#endif

    std::vector<BenchmarkInfo> benchmarks = {
        {"Matrix/Vector Operations",
         "Tests basic linear algebra operations with and without SIMD",
         (exe_path / ("benchmark_matrix_vector" + exe_suffix)).string()},

        {"DDA Operations",
         "Tests line and curve rasterization with regular and batched iterators",
         (exe_path / ("benchmark_dda" + exe_suffix)).string()},

        {"SIMD Operations",
         "Direct comparison of scalar vs SIMD implementations",
         (exe_path / ("benchmark_simd" + exe_suffix)).string()}
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