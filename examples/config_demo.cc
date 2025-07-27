/**
 * @example config_demo.cc
 * @brief Demonstrates configuration macros for the Euler library
 * 
 * This example shows how to use various configuration macros
 * to customize the behavior of the Euler library.
 */

// Example configuration - uncomment to test different settings
// #define EULER_DISABLE_SIMD        // Disable SIMD optimizations
// #define EULER_DEFAULT_EPSILON 1e-8 // Use custom epsilon for comparisons
// #define EULER_DISABLE_ENFORCE      // Disable all runtime checks
// #define EULER_ENABLE_BOUNDS_CHECK  // Enable bounds checking

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace euler;

int main() {
    std::cout << "=== Euler Library Configuration Demo ===\n\n";
    
    // Show current epsilon value
    std::cout << "1. Default epsilon value:\n";
    std::cout << "   float epsilon: " << std::scientific << constants<float>::epsilon << "\n";
    std::cout << "   double epsilon: " << constants<double>::epsilon << "\n\n";
    
    // Test approximate equality with different epsilon
    std::cout << "2. Approximate equality test:\n";
    float a = 1.0f;
    float b = 1.0f + 1e-7f;
    
    std::cout << "   a = " << a << "\n";
    std::cout << "   b = " << b << "\n";
    std::cout << "   approx_equal(a, b) with default epsilon: " 
              << (approx_equal(a, b) ? "true" : "false") << "\n";
    std::cout << "   approx_equal(a, b, 1e-6f): " 
              << (approx_equal(a, b, 1e-6f) ? "true" : "false") << "\n";
    std::cout << "   approx_equal(a, b, 1e-8f): " 
              << (approx_equal(a, b, 1e-8f) ? "true" : "false") << "\n\n";
    
    // Show SIMD status
    std::cout << "3. SIMD optimization status:\n";
    #ifdef EULER_HAS_XSIMD
        std::cout << "   SIMD is ENABLED (using xsimd)\n";
        std::cout << "   Float batch size: " << simd_traits<float>::batch_size << "\n";
        std::cout << "   Double batch size: " << simd_traits<double>::batch_size << "\n";
    #else
        std::cout << "   SIMD is DISABLED\n";
        std::cout << "   All operations use scalar fallback\n";
    #endif
    std::cout << "\n";
    
    // Test runtime checks
    std::cout << "4. Runtime checks status:\n";
    #ifdef EULER_DISABLE_ENFORCE
        std::cout << "   Runtime checks are DISABLED\n";
        std::cout << "   No bounds checking or dimension validation\n";
    #else
        std::cout << "   Runtime checks are ENABLED\n";
        #ifdef EULER_DEBUG
            std::cout << "   Debug mode: verbose error messages\n";
        #elif defined(EULER_SAFE_RELEASE)
            std::cout << "   Safe release mode: basic error messages\n";
        #else
            std::cout << "   Release mode: minimal checks\n";
        #endif
    #endif
    std::cout << "\n";
    
    // Example of performance difference with/without SIMD
    std::cout << "5. Performance example:\n";
    const size_t size = 1000000;
    vector<float, 3>* vectors = new vector<float, 3>[size];
    
    // Initialize with random values
    for (size_t i = 0; i < size; ++i) {
        vectors[i] = vector<float, 3>(
            static_cast<float>(static_cast<int>(i) % 100) / 100.0f,
            static_cast<float>(static_cast<int>((i + 1)) % 100) / 100.0f,
            static_cast<float>(static_cast<int>((i + 2)) % 100) / 100.0f
        );
    }
    
    // Compute dot products
    float sum = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < size - 1; ++i) {
        sum += dot(vectors[i], vectors[i + 1]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "   Computed " << size - 1 << " dot products\n";
    std::cout << "   Time: " << duration.count() << " microseconds\n";
    std::cout << "   Sum: " << sum << "\n\n";
    
    delete[] vectors;
    
    // Show compile-time configuration summary
    std::cout << "6. Compile-time configuration summary:\n";
    std::cout << "   C++ Standard: " << __cplusplus << "\n";
    
    #ifdef EULER_DISABLE_SIMD
        std::cout << "   EULER_DISABLE_SIMD: defined\n";
    #endif
    
    #ifdef EULER_DEFAULT_EPSILON
        std::cout << "   EULER_DEFAULT_EPSILON: " << EULER_DEFAULT_EPSILON << "\n";
    #endif
    
    #ifdef EULER_DISABLE_ENFORCE
        std::cout << "   EULER_DISABLE_ENFORCE: defined\n";
    #endif
    
    #ifdef EULER_ENABLE_BOUNDS_CHECK
        std::cout << "   EULER_ENABLE_BOUNDS_CHECK: defined\n";
    #endif
    
    #ifdef EULER_DEFAULT_COLUMN_MAJOR
        std::cout << "   EULER_DEFAULT_COLUMN_MAJOR: defined\n";
    #endif
    
    std::cout << "\nTo test different configurations, uncomment the #define directives\n";
    std::cout << "at the top of this file and recompile.\n";
    
    return 0;
}