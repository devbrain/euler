/**
 * @file benchmark_matrix_vector.cc
 * @brief Benchmarks for matrix and vector operations with/without SIMD
 */

#include "benchmark_utils.hh"
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <random>
#include <vector>
#include <cmath>

using namespace euler;
using namespace euler::benchmark;

// Test sizes
constexpr size_t SMALL_SIZE = 100;
constexpr size_t MEDIUM_SIZE = 1000;
constexpr size_t LARGE_SIZE = 10000;

// Generate random data
template<typename T>
std::vector<T> generate_random_data(size_t count, T min_val = -1.0, T max_val = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min_val, max_val);
    
    std::vector<T> data;
    data.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        data.push_back(dist(gen));
    }
    return data;
}

// Benchmark vector dot product
template<typename T>
void benchmark_dot_product(size_t size) {
    std::cout << "\n=== Vector Dot Product (size=" << size << ") ===" << std::endl;
    
    // Generate test data
    auto data1 = generate_random_data<T>(size * 3);
    auto data2 = generate_random_data<T>(size * 3);
    
    std::vector<vec3<T>> vectors1, vectors2;
    vectors1.reserve(size);
    vectors2.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        vectors1.emplace_back(data1[i*3], data1[i*3+1], data1[i*3+2]);
        vectors2.emplace_back(data2[i*3], data2[i*3+1], data2[i*3+2]);
    }
    
    T result = 0;
    
    // Scalar version
    auto scalar_bench = run_benchmark("Scalar dot product", [&]() {
        result = 0;
        for (size_t i = 0; i < size; ++i) {
            result += vectors1[i][0] * vectors2[i][0] +
                     vectors1[i][1] * vectors2[i][1] +
                     vectors1[i][2] * vectors2[i][2];
        }
        do_not_optimize(result);
    });
    
    // Using euler's dot product (potentially SIMD optimized)
    auto euler_bench = run_benchmark("Euler dot product", [&]() {
        result = 0;
        for (size_t i = 0; i < size; ++i) {
            result += dot(vectors1[i], vectors2[i]);
        }
        do_not_optimize(result);
    });
    
    compare_benchmarks("Scalar", scalar_bench, "Euler", euler_bench);
}

// Benchmark vector cross product
template<typename T>
void benchmark_cross_product(size_t size) {
    std::cout << "\n=== Vector Cross Product (size=" << size << ") ===" << std::endl;
    
    auto data1 = generate_random_data<T>(size * 3);
    auto data2 = generate_random_data<T>(size * 3);
    
    std::vector<vec3<T>> vectors1, vectors2, results;
    vectors1.reserve(size);
    vectors2.reserve(size);
    results.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        vectors1.emplace_back(data1[i*3], data1[i*3+1], data1[i*3+2]);
        vectors2.emplace_back(data2[i*3], data2[i*3+1], data2[i*3+2]);
    }
    
    // Scalar version
    auto scalar_bench = run_benchmark("Scalar cross product", [&]() {
        for (size_t i = 0; i < size; ++i) {
            results[i][0] = vectors1[i][1] * vectors2[i][2] - vectors1[i][2] * vectors2[i][1];
            results[i][1] = vectors1[i][2] * vectors2[i][0] - vectors1[i][0] * vectors2[i][2];
            results[i][2] = vectors1[i][0] * vectors2[i][1] - vectors1[i][1] * vectors2[i][0];
        }
        do_not_optimize(results.data());
    });
    
    // Using euler's cross product
    auto euler_bench = run_benchmark("Euler cross product", [&]() {
        for (size_t i = 0; i < size; ++i) {
            results[i] = cross(vectors1[i], vectors2[i]);
        }
        do_not_optimize(results.data());
    });
    
    compare_benchmarks("Scalar", scalar_bench, "Euler", euler_bench);
}

// Benchmark matrix multiplication
template<typename T>
void benchmark_matrix_multiply() {
    std::cout << "\n=== Matrix 4x4 Multiplication ===" << std::endl;
    
    constexpr size_t count = 10000;
    
    // Generate random matrices
    std::vector<matrix4<T>> matrices1, matrices2, results;
    matrices1.reserve(count);
    matrices2.reserve(count);
    results.resize(count);
    
    for (size_t i = 0; i < count; ++i) {
        auto data1 = generate_random_data<T>(16);
        auto data2 = generate_random_data<T>(16);
        
        matrices1.emplace_back(std::initializer_list<std::initializer_list<T>>{
            {data1[0], data1[1], data1[2], data1[3]},
            {data1[4], data1[5], data1[6], data1[7]},
            {data1[8], data1[9], data1[10], data1[11]},
            {data1[12], data1[13], data1[14], data1[15]}
        });
        
        matrices2.emplace_back(std::initializer_list<std::initializer_list<T>>{
            {data2[0], data2[1], data2[2], data2[3]},
            {data2[4], data2[5], data2[6], data2[7]},
            {data2[8], data2[9], data2[10], data2[11]},
            {data2[12], data2[13], data2[14], data2[15]}
        });
    }
    
    // Scalar version (column-major order)
    auto scalar_bench = run_benchmark("Scalar matrix multiply", [&]() {
        for (size_t idx = 0; idx < count; ++idx) {
            const auto& a = matrices1[idx];
            const auto& b = matrices2[idx];
            auto& c = results[idx];
            
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    T sum = 0;
                    for (int k = 0; k < 4; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    c(i, j) = sum;
                }
            }
        }
        do_not_optimize(results.data());
    });
    
    // Using euler's matrix multiplication
    auto euler_bench = run_benchmark("Euler matrix multiply", [&]() {
        for (size_t i = 0; i < count; ++i) {
            results[i] = matrices1[i] * matrices2[i];
        }
        do_not_optimize(results.data());
    });
    
    compare_benchmarks("Scalar", scalar_bench, "Euler", euler_bench);
}

// Benchmark matrix-vector multiplication
template<typename T>
void benchmark_matrix_vector_multiply() {
    std::cout << "\n=== Matrix 4x4 * Vector4 Multiplication ===" << std::endl;
    
    constexpr size_t count = 50000;
    
    // Generate test data
    std::vector<matrix4<T>> matrices;
    std::vector<vec4<T>> vectors, results;
    matrices.reserve(count);
    vectors.reserve(count);
    results.resize(count);
    
    for (size_t i = 0; i < count; ++i) {
        auto mat_data = generate_random_data<T>(16);
        auto vec_data = generate_random_data<T>(4);
        
        matrices.emplace_back(std::initializer_list<std::initializer_list<T>>{
            {mat_data[0], mat_data[1], mat_data[2], mat_data[3]},
            {mat_data[4], mat_data[5], mat_data[6], mat_data[7]},
            {mat_data[8], mat_data[9], mat_data[10], mat_data[11]},
            {mat_data[12], mat_data[13], mat_data[14], mat_data[15]}
        });
        
        vectors.emplace_back(vec_data[0], vec_data[1], vec_data[2], vec_data[3]);
    }
    
    // Scalar version
    auto scalar_bench = run_benchmark("Scalar mat4*vec4", [&]() {
        for (size_t idx = 0; idx < count; ++idx) {
            const auto& m = matrices[idx];
            const auto& v = vectors[idx];
            auto& r = results[idx];
            
            r[0] = m(0,0)*v[0] + m(0,1)*v[1] + m(0,2)*v[2] + m(0,3)*v[3];
            r[1] = m(1,0)*v[0] + m(1,1)*v[1] + m(1,2)*v[2] + m(1,3)*v[3];
            r[2] = m(2,0)*v[0] + m(2,1)*v[1] + m(2,2)*v[2] + m(2,3)*v[3];
            r[3] = m(3,0)*v[0] + m(3,1)*v[1] + m(3,2)*v[2] + m(3,3)*v[3];
        }
        do_not_optimize(results.data());
    });
    
    // Using euler's matrix-vector multiplication
    auto euler_bench = run_benchmark("Euler mat4*vec4", [&]() {
        for (size_t i = 0; i < count; ++i) {
            results[i] = matrices[i] * vectors[i];
        }
        do_not_optimize(results.data());
    });
    
    compare_benchmarks("Scalar", scalar_bench, "Euler", euler_bench);
}

// Benchmark vector normalization
template<typename T>
void benchmark_vector_normalize(size_t size) {
    std::cout << "\n=== Vector Normalization (size=" << size << ") ===" << std::endl;
    
    auto data = generate_random_data<T>(size * 3, T(0.1), T(10.0)); // Avoid near-zero vectors
    
    std::vector<vec3<T>> vectors, results;
    vectors.reserve(size);
    results.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        vectors.emplace_back(data[i*3], data[i*3+1], data[i*3+2]);
    }
    
    // Scalar version
    auto scalar_bench = run_benchmark("Scalar normalize", [&]() {
        for (size_t i = 0; i < size; ++i) {
            T len = std::sqrt(vectors[i][0] * vectors[i][0] + 
                             vectors[i][1] * vectors[i][1] + 
                             vectors[i][2] * vectors[i][2]);
            results[i][0] = vectors[i][0] / len;
            results[i][1] = vectors[i][1] / len;
            results[i][2] = vectors[i][2] / len;
        }
        do_not_optimize(results.data());
    });
    
    // Using euler's normalize
    auto euler_bench = run_benchmark("Euler normalize", [&]() {
        for (size_t i = 0; i < size; ++i) {
            results[i] = normalize(vectors[i]);
        }
        do_not_optimize(results.data());
    });
    
    compare_benchmarks("Scalar", scalar_bench, "Euler", euler_bench);
}

int main() {
    std::cout << "Euler Math Benchmarks - Matrix/Vector Operations" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Test with float
    std::cout << "\n--- Float precision ---" << std::endl;
    benchmark_dot_product<float>(SMALL_SIZE);
    benchmark_dot_product<float>(MEDIUM_SIZE);
    benchmark_dot_product<float>(LARGE_SIZE);
    
    benchmark_cross_product<float>(SMALL_SIZE);
    benchmark_cross_product<float>(MEDIUM_SIZE);
    benchmark_cross_product<float>(LARGE_SIZE);
    
    benchmark_matrix_multiply<float>();
    benchmark_matrix_vector_multiply<float>();
    
    benchmark_vector_normalize<float>(SMALL_SIZE);
    benchmark_vector_normalize<float>(MEDIUM_SIZE);
    benchmark_vector_normalize<float>(LARGE_SIZE);
    
    // Test with double
    std::cout << "\n--- Double precision ---" << std::endl;
    benchmark_dot_product<double>(SMALL_SIZE);
    benchmark_dot_product<double>(MEDIUM_SIZE);
    benchmark_dot_product<double>(LARGE_SIZE);
    
    benchmark_cross_product<double>(SMALL_SIZE);
    benchmark_cross_product<double>(MEDIUM_SIZE);
    
    benchmark_matrix_multiply<double>();
    benchmark_matrix_vector_multiply<double>();
    
    benchmark_vector_normalize<double>(MEDIUM_SIZE);
    
    return 0;
}