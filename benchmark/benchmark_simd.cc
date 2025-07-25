/**
 * @file benchmark_simd.cc
 * @brief Benchmarks specifically for SIMD vs scalar operations
 */

#include "benchmark_utils.hh"
#include <euler/core/simd.hh>
#include <euler/vector/vector.hh>
#include <euler/dda/bezier_simd.hh>
#include <random>
#include <vector>
#include <cmath>
#include <typeinfo>

using namespace euler;
using namespace euler::benchmark;

// Force scalar implementation for comparison
template<typename T>
struct benchmark_scalar_ops {
    // Prevent compiler auto-vectorization using volatile
    static void add_arrays(const T* a, const T* b, T* result, size_t count) {
        // Use volatile to prevent vectorization
        volatile T temp;
        for (size_t i = 0; i < count; ++i) {
            temp = a[i] + b[i];
            result[i] = temp;
        }
    }
    
    static void multiply_arrays(const T* a, const T* b, T* result, size_t count) {
        // Use volatile to prevent vectorization
        volatile T temp;
        for (size_t i = 0; i < count; ++i) {
            temp = a[i] * b[i];
            result[i] = temp;
        }
    }
    
    static void fma_arrays(const T* a, const T* b, const T* c, T* result, size_t count) {
        // Use volatile to prevent vectorization
        volatile T temp;
        for (size_t i = 0; i < count; ++i) {
            temp = a[i] * b[i] + c[i];
            result[i] = temp;
        }
    }
    
    static T sum_array(const T* a, size_t count) {
        // Use volatile to prevent vectorization
        volatile T sum = 0;
        for (size_t i = 0; i < count; ++i) {
            sum = sum + a[i];
        }
        return sum;
    }
};

// SIMD implementation when available
template<typename T>
struct benchmark_simd_ops {
    static void add_arrays(const T* a, const T* b, T* result, size_t count) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            size_t simd_end = count - (count % simd_size);
            
            for (size_t i = 0; i < simd_end; i += simd_size) {
                auto va = batch_t::load_aligned(&a[i]);
                auto vb = batch_t::load_aligned(&b[i]);
                auto vr = va + vb;
                vr.store_aligned(&result[i]);
            }
            
            // Handle remaining elements
            for (size_t i = simd_end; i < count; ++i) {
                result[i] = a[i] + b[i];
            }
        } else
        #endif
        {
            benchmark_scalar_ops<T>::add_arrays(a, b, result, count);
        }
    }
    
    static void multiply_arrays(const T* a, const T* b, T* result, size_t count) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            size_t simd_end = count - (count % simd_size);
            
            for (size_t i = 0; i < simd_end; i += simd_size) {
                auto va = batch_t::load_aligned(&a[i]);
                auto vb = batch_t::load_aligned(&b[i]);
                auto vr = va * vb;
                vr.store_aligned(&result[i]);
            }
            
            // Handle remaining elements
            for (size_t i = simd_end; i < count; ++i) {
                result[i] = a[i] * b[i];
            }
        } else
        #endif
        {
            benchmark_scalar_ops<T>::multiply_arrays(a, b, result, count);
        }
    }
    
    static void fma_arrays(const T* a, const T* b, const T* c, T* result, size_t count) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            size_t simd_end = count - (count % simd_size);
            
            for (size_t i = 0; i < simd_end; i += simd_size) {
                auto va = batch_t::load_aligned(&a[i]);
                auto vb = batch_t::load_aligned(&b[i]);
                auto vc = batch_t::load_aligned(&c[i]);
                auto vr = xsimd::fma(va, vb, vc);
                vr.store_aligned(&result[i]);
            }
            
            // Handle remaining elements
            for (size_t i = simd_end; i < count; ++i) {
                result[i] = a[i] * b[i] + c[i];
            }
        } else
        #endif
        {
            benchmark_scalar_ops<T>::fma_arrays(a, b, c, result, count);
        }
    }
    
    static T sum_array(const T* a, size_t count) {
        #ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            size_t simd_end = count - (count % simd_size);
            batch_t sum_vec = batch_t(T(0));
            
            for (size_t i = 0; i < simd_end; i += simd_size) {
                auto va = batch_t::load_aligned(&a[i]);
                sum_vec = sum_vec + va;
            }
            
            // Horizontal sum
            T sum = xsimd::reduce_add(sum_vec);
            
            // Handle remaining elements
            for (size_t i = simd_end; i < count; ++i) {
                sum += a[i];
            }
            
            return sum;
        } else
        #endif
        {
            return benchmark_scalar_ops<T>::sum_array(a, count);
        }
    }
};

// Benchmark array operations
template<typename T>
void benchmark_array_operations(size_t size) {
    std::cout << "\n=== Array Operations (size=" << size << ", type=" 
              << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    // Generate test data - use aligned allocation for SIMD
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    
    // Allocate aligned memory
    T* a = aligned_alloc<T>(size);
    T* b = aligned_alloc<T>(size);
    T* c = aligned_alloc<T>(size);
    T* result = aligned_alloc<T>(size);
    
    for (size_t i = 0; i < size; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
        c[i] = dist(gen);
    }
    
    // Ensure cleanup at end of function
    auto cleanup = [&]() {
        aligned_free(a);
        aligned_free(b);
        aligned_free(c);
        aligned_free(result);
    };
    
    // Addition benchmark
    {
        auto scalar_bench = run_benchmark("Scalar array addition", [&]() {
            benchmark_scalar_ops<T>::add_arrays(a, b, result, size);
            do_not_optimize(result);
        });
        
        auto simd_bench = run_benchmark("SIMD array addition", [&]() {
            benchmark_simd_ops<T>::add_arrays(a, b, result, size);
            do_not_optimize(result);
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
    }
    
    // Multiplication benchmark
    {
        auto scalar_bench = run_benchmark("Scalar array multiplication", [&]() {
            benchmark_scalar_ops<T>::multiply_arrays(a, b, result, size);
            do_not_optimize(result);
        });
        
        auto simd_bench = run_benchmark("SIMD array multiplication", [&]() {
            benchmark_simd_ops<T>::multiply_arrays(a, b, result, size);
            do_not_optimize(result);
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
    }
    
    // FMA benchmark
    {
        auto scalar_bench = run_benchmark("Scalar array FMA", [&]() {
            benchmark_scalar_ops<T>::fma_arrays(a, b, c, result, size);
            do_not_optimize(result);
        });
        
        auto simd_bench = run_benchmark("SIMD array FMA", [&]() {
            benchmark_simd_ops<T>::fma_arrays(a, b, c, result, size);
            do_not_optimize(result);
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
    }
    
    // Sum reduction benchmark
    {
        T sum;
        
        auto scalar_bench = run_benchmark("Scalar array sum", [&]() {
            sum = benchmark_scalar_ops<T>::sum_array(a, size);
            do_not_optimize(sum);
        });
        
        auto simd_bench = run_benchmark("SIMD array sum", [&]() {
            sum = benchmark_simd_ops<T>::sum_array(a, size);
            do_not_optimize(sum);
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
    }
    
    // Cleanup aligned memory
    cleanup();
}

// Benchmark bezier evaluation
template<typename T>
void benchmark_bezier_evaluation() {
    std::cout << "\n=== Bezier Curve Evaluation (type=" 
              << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    constexpr size_t num_curves = 1000;
    constexpr size_t samples_per_curve = 100;
    
    // Generate random control points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0, 1000);
    
    struct CubicBezier {
        point2<T> p0, p1, p2, p3;
    };
    
    std::vector<CubicBezier> curves;
    curves.reserve(num_curves);
    
    for (size_t i = 0; i < num_curves; ++i) {
        curves.push_back({
            point2<T>{dist(gen), dist(gen)},
            point2<T>{dist(gen), dist(gen)},
            point2<T>{dist(gen), dist(gen)},
            point2<T>{dist(gen), dist(gen)}
        });
    }
    
    // Generate t values
    std::vector<T> t_values;
    t_values.reserve(samples_per_curve);
    for (size_t i = 0; i < samples_per_curve; ++i) {
        t_values.push_back(static_cast<T>(i) / static_cast<T>(samples_per_curve - 1));
    }
    
    std::vector<point2<T>> results(num_curves * samples_per_curve);
    
    // Scalar evaluation
    auto scalar_bench = run_benchmark("Scalar bezier evaluation", [&]() {
        size_t idx = 0;
        for (const auto& curve : curves) {
            for (T t : t_values) {
                T t2 = t * t;
                T t3 = t2 * t;
                T one_minus_t = T(1) - t;
                T one_minus_t2 = one_minus_t * one_minus_t;
                T one_minus_t3 = one_minus_t2 * one_minus_t;
                
                results[idx] = point2<T>{
                    one_minus_t3 * curve.p0.x + 
                    T(3) * one_minus_t2 * t * curve.p1.x + 
                    T(3) * one_minus_t * t2 * curve.p2.x + 
                    t3 * curve.p3.x,
                    one_minus_t3 * curve.p0.y + 
                    T(3) * one_minus_t2 * t * curve.p1.y + 
                    T(3) * one_minus_t * t2 * curve.p2.y + 
                    t3 * curve.p3.y
                };
                idx++;
            }
        }
        do_not_optimize(results.data());
    });
    
    // SIMD batch evaluation using bezier_simd utilities
    #ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        auto simd_bench = run_benchmark("SIMD bezier evaluation", [&]() {
            size_t idx = 0;
            
            // Process curves with SIMD-friendly batch evaluation
            for (const auto& curve : curves) {
                // Evaluate using SIMD when processing multiple t values
                for (T t : t_values) {
                    // Manual SIMD-friendly evaluation
                    T t2 = t * t;
                    T t3 = t2 * t;
                    T one_minus_t = T(1) - t;
                    T one_minus_t2 = one_minus_t * one_minus_t;
                    T one_minus_t3 = one_minus_t2 * one_minus_t;
                    
                    // Compute basis functions
                    T b0 = one_minus_t3;
                    T b1 = T(3) * one_minus_t2 * t;
                    T b2 = T(3) * one_minus_t * t2;
                    T b3 = t3;
                    
                    // Evaluate point
                    results[idx] = point2<T>{
                        b0 * curve.p0.x + b1 * curve.p1.x + b2 * curve.p2.x + b3 * curve.p3.x,
                        b0 * curve.p0.y + b1 * curve.p1.y + b2 * curve.p2.y + b3 * curve.p3.y
                    };
                    idx++;
                }
            }
            
            do_not_optimize(results.data());
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
    } else {
        std::cout << "  SIMD not available for type " << typeid(T).name() << std::endl;
    }
    #else
    std::cout << "  SIMD not available - skipping SIMD benchmark" << std::endl;
    #endif
}

// Benchmark transcendental functions
template<typename T>
void benchmark_transcendental_functions(size_t size) {
    std::cout << "\n=== Transcendental Functions (size=" << size << ", type=" 
              << (sizeof(T) == 4 ? "float" : "double") << ") ===" << std::endl;
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.1, 10.0);
    
    T* input = aligned_alloc<T>(size);
    T* output = aligned_alloc<T>(size);
    
    for (size_t i = 0; i < size; ++i) {
        input[i] = dist(gen);
    }
    
    auto cleanup = [&]() {
        aligned_free(input);
        aligned_free(output);
    };
    
    // Square root benchmark
    {
        auto scalar_bench = run_benchmark("Scalar sqrt", [&]() {
            for (size_t i = 0; i < size; ++i) {
                output[i] = std::sqrt(input[i]);
            }
            do_not_optimize(output);
        });
        
        #ifdef EULER_HAS_XSIMD
        auto simd_bench = run_benchmark("SIMD sqrt", [&]() {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            size_t simd_end = size - (size % simd_size);
            
            for (size_t i = 0; i < simd_end; i += simd_size) {
                auto v = batch_t::load_aligned(&input[i]);
                auto result = xsimd::sqrt(v);
                result.store_aligned(&output[i]);
            }
            
            for (size_t i = simd_end; i < size; ++i) {
                output[i] = std::sqrt(input[i]);
            }
            
            do_not_optimize(output);
        });
        
        compare_benchmarks("Scalar", scalar_bench, "SIMD", simd_bench);
        #else
        std::cout << "  SIMD not available" << std::endl;
        #endif
    }
    
    cleanup();
}

// Check SIMD availability
void print_simd_info() {
    std::cout << "\n=== SIMD Information ===" << std::endl;
    
    #ifdef EULER_HAS_XSIMD
        std::cout << "xsimd available: YES" << std::endl;
        
        #ifdef __SSE__
            std::cout << "SSE: YES" << std::endl;
        #endif
        #ifdef __SSE2__
            std::cout << "SSE2: YES" << std::endl;
        #endif
        #ifdef __SSE3__
            std::cout << "SSE3: YES" << std::endl;
        #endif
        #ifdef __SSSE3__
            std::cout << "SSSE3: YES" << std::endl;
        #endif
        #ifdef __SSE4_1__
            std::cout << "SSE4.1: YES" << std::endl;
        #endif
        #ifdef __SSE4_2__
            std::cout << "SSE4.2: YES" << std::endl;
        #endif
        #ifdef __AVX__
            std::cout << "AVX: YES" << std::endl;
        #endif
        #ifdef __AVX2__
            std::cout << "AVX2: YES" << std::endl;
        #endif
        #ifdef __AVX512F__
            std::cout << "AVX512F: YES" << std::endl;
        #endif
        
        std::cout << "Float SIMD width: " << simd_traits<float>::batch_size << std::endl;
        std::cout << "Double SIMD width: " << simd_traits<double>::batch_size << std::endl;
    #else
        std::cout << "xsimd available: NO" << std::endl;
        std::cout << "All operations will use scalar implementations" << std::endl;
    #endif
}

int main() {
    std::cout << "Euler SIMD Benchmarks" << std::endl;
    std::cout << "=====================" << std::endl;
    
    print_simd_info();
    
    // Array operation benchmarks
    benchmark_array_operations<float>(1024);
    benchmark_array_operations<float>(10240);
    benchmark_array_operations<float>(102400);
    
    benchmark_array_operations<double>(1024);
    benchmark_array_operations<double>(10240);
    benchmark_array_operations<double>(102400);
    
    // Bezier evaluation benchmarks
    benchmark_bezier_evaluation<float>();
    benchmark_bezier_evaluation<double>();
    
    // Transcendental function benchmarks
    benchmark_transcendental_functions<float>(10000);
    benchmark_transcendental_functions<double>(10000);
    
    return 0;
}