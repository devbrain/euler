#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/specialized.hh>
#include <iostream>
#include <chrono>
#include <random>

using namespace euler;

int main() {
    // Test with random matrices to prevent optimization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    matrix4<float> a, b;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            a(i,j) = dist(gen);
            b(i,j) = dist(gen);
        }
    }
    
    // Use fewer iterations to avoid CI timeout (60s limit)
    // CI runners can be slow even in Release mode
#ifdef NDEBUG
    const int iterations = 50000;
#else
    const int iterations = 5000;
#endif
    
    // Test expression template multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrix4<float> result1;
    for (int i = 0; i < iterations; ++i) {
        result1 = a * b;  // Uses expression templates
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto expr_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Test direct SIMD multiplication
    start = std::chrono::high_resolution_clock::now();
    matrix4<float> result2;
    for (int i = 0; i < iterations; ++i) {
        result2 = multiply_4x4_simd(a, b);  // Direct SIMD
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Test direct multiplication
    start = std::chrono::high_resolution_clock::now();
    matrix4<float> result3;
    for (int i = 0; i < iterations; ++i) {
        result3 = multiply_direct(a, b);  // Should use SIMD internally
    }
    end = std::chrono::high_resolution_clock::now();
    auto direct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Expression template time: " << expr_time << " us" << std::endl;
    std::cout << "Direct SIMD time: " << simd_time << " us" << std::endl;
    std::cout << "Direct multiply time: " << direct_time << " us" << std::endl;
    
    // Verify results are the same
    std::cout << "\nResult verification:" << std::endl;
    std::cout << "result1(0,0) = " << result1(0,0) << std::endl;
    std::cout << "result2(0,0) = " << result2(0,0) << std::endl;
    std::cout << "result3(0,0) = " << result3(0,0) << std::endl;
    
    return 0;
}