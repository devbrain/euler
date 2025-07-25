/**
 * @file bezier_simd.hh
 * @brief SIMD-optimized Bezier curve batch evaluation utilities
 * @ingroup DDAModule
 * 
 * This header provides utility functions for evaluating multiple Bezier
 * curves in batch using SIMD instructions. These are advanced utilities
 * for high-performance scenarios where many curves need to be evaluated.
 * 
 * @note The standard bezier iterators already use SIMD internally when
 *       available. These batch functions are for specialized use cases.
 */
#pragma once

#include <euler/core/simd.hh>
#include <euler/coordinates/point2.hh>
#include <euler/vector/vector.hh>
#include <euler/dda/dda_math.hh>
#include <array>
#include <vector>

namespace euler::dda {

#ifdef EULER_HAS_XSIMD

/**
 * @brief SIMD-optimized Bezier evaluation functions
 * 
 * These functions provide vectorized implementations of Bezier curve
 * evaluation for better performance on modern CPUs.
 */
namespace bezier_simd {

/**
 * @brief Evaluate multiple Bezier curves at once using SIMD
 * @tparam T Floating point type (float or double)
 * @param control_points Array of control points for multiple curves
 * @param binomial_coeffs Precomputed binomial coefficients
 * @param degree Degree of the Bezier curves
 * @param t Parameter values for evaluation
 * @param count Number of curves to evaluate
 * @return Array of evaluated points
 */
template<typename T>
inline std::vector<point2<T>> evaluate_batch(
    const point2<T>* control_points,
    const T* binomial_coeffs,
    int degree,
    const T* t_values,
    size_t count) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    constexpr size_t simd_size = simd_traits<T>::batch_size;
    
    std::vector<point2<T>> results(count);
    
    // Process in SIMD batches
    size_t i = 0;
    for (; i + simd_size <= count; i += simd_size) {
        // Load t values
        batch_t t = batch_t::load_unaligned(&t_values[i]);
        batch_t one_minus_t = batch_t(1) - t;
        
        // Initialize result accumulators
        batch_t result_x = batch_t(0);
        batch_t result_y = batch_t(0);
        
        // Evaluate basis functions and accumulate
        for (int j = 0; j <= degree; ++j) {
            // Compute basis function
            batch_t basis = batch_t(binomial_coeffs[j]);
            
            // Compute powers - this is the expensive part
            if (degree - j > 0) {
                batch_t power1 = xsimd::pow(one_minus_t, batch_t(degree - j));
                basis *= power1;
            }
            if (j > 0) {
                batch_t power2 = xsimd::pow(t, batch_t(j));
                basis *= power2;
            }
            
            // Accumulate weighted control points
            for (size_t k = 0; k < simd_size; ++k) {
                size_t idx = i + k;
                result_x[k] += basis[k] * control_points[idx * (degree + 1) + j].x;
                result_y[k] += basis[k] * control_points[idx * (degree + 1) + j].y;
            }
        }
        
        // Store results
        alignas(simd_alignment<T>()) T temp_x[simd_size];
        alignas(simd_alignment<T>()) T temp_y[simd_size];
        result_x.store_aligned(temp_x);
        result_y.store_aligned(temp_y);
        
        for (size_t k = 0; k < simd_size; ++k) {
            results[i + k] = point2<T>{temp_x[k], temp_y[k]};
        }
    }
    
    // Handle remaining elements
    for (; i < count; ++i) {
        T t_val = t_values[i];
        T one_minus_t_val = T(1) - t_val;
        point2<T> result{0, 0};
        
        for (int j = 0; j <= degree; ++j) {
            T basis = binomial_coeffs[j];
            if (degree - j > 0) {
                basis *= euler::pow(one_minus_t_val, static_cast<T>(degree - j));
            }
            if (j > 0) {
                basis *= euler::pow(t_val, static_cast<T>(j));
            }
            
            size_t idx = i * (degree + 1) + j;
            result.x += basis * control_points[idx].x;
            result.y += basis * control_points[idx].y;
        }
        
        results[i] = result;
    }
    
    return results;
}

/**
 * @brief SIMD-optimized evaluation of a single Bezier curve
 * 
 * This version processes multiple basis functions in parallel
 */
template<typename T>
inline point2<T> evaluate_single_simd(
    const std::vector<point2<T>>& control_points,
    const T* binomial_coeffs,
    int degree,
    T t) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    constexpr size_t simd_size = simd_traits<T>::batch_size;
    
    T one_minus_t = T(1) - t;
    
    // Precompute all powers
    std::vector<T> powers_t(degree + 1);
    std::vector<T> powers_one_minus_t(degree + 1);
    
    powers_t[0] = T(1);
    powers_one_minus_t[0] = T(1);
    
    for (int i = 1; i <= degree; ++i) {
        powers_t[i] = powers_t[i-1] * t;
        powers_one_minus_t[i] = powers_one_minus_t[i-1] * one_minus_t;
    }
    
    // Process control points in SIMD batches
    T result_x = 0;
    T result_y = 0;
    
    int i = 0;
    if (degree + 1 >= static_cast<int>(simd_size)) {
        for (; i + static_cast<int>(simd_size) <= degree + 1; i += simd_size) {
            batch_t basis_batch(0);
            batch_t x_batch(0);
            batch_t y_batch(0);
            
            // Load and compute basis functions for this batch
            for (size_t j = 0; j < simd_size; ++j) {
                int idx = i + j;
                T basis = binomial_coeffs[idx] * 
                         powers_one_minus_t[degree - idx] * 
                         powers_t[idx];
                basis_batch[j] = basis;
                x_batch[j] = control_points[idx].x;
                y_batch[j] = control_points[idx].y;
            }
            
            // Accumulate
            result_x += xsimd::reduce_add(basis_batch * x_batch);
            result_y += xsimd::reduce_add(basis_batch * y_batch);
        }
    }
    
    // Handle remaining control points
    for (; i <= degree; ++i) {
        T basis = binomial_coeffs[i] * 
                 powers_one_minus_t[degree - i] * 
                 powers_t[i];
        result_x += basis * control_points[i].x;
        result_y += basis * control_points[i].y;
    }
    
    return point2<T>{result_x, result_y};
}

/**
 * @brief Precompute powers for multiple t values using SIMD
 */
template<typename T>
inline void precompute_powers_simd(
    const T* t_values,
    size_t count,
    int max_degree,
    T* powers_t,
    T* powers_one_minus_t) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    constexpr size_t simd_size = simd_traits<T>::batch_size;
    
    size_t i = 0;
    for (; i + simd_size <= count; i += simd_size) {
        batch_t t = batch_t::load_unaligned(&t_values[i]);
        batch_t one_minus_t = batch_t(1) - t;
        
        // Store base powers (t^0 = 1)
        batch_t(1).store_unaligned(&powers_t[i * (max_degree + 1)]);
        batch_t(1).store_unaligned(&powers_one_minus_t[i * (max_degree + 1)]);
        
        // Compute higher powers
        batch_t t_power = t;
        batch_t one_minus_t_power = one_minus_t;
        
        for (int d = 1; d <= max_degree; ++d) {
            t_power.store_unaligned(&powers_t[i * (max_degree + 1) + d * count]);
            one_minus_t_power.store_unaligned(&powers_one_minus_t[i * (max_degree + 1) + d * count]);
            
            t_power *= t;
            one_minus_t_power *= one_minus_t;
        }
    }
    
    // Handle remaining elements
    for (; i < count; ++i) {
        T t = t_values[i];
        T one_minus_t = T(1) - t;
        
        powers_t[i * (max_degree + 1)] = T(1);
        powers_one_minus_t[i * (max_degree + 1)] = T(1);
        
        T t_power = t;
        T one_minus_t_power = one_minus_t;
        
        for (int d = 1; d <= max_degree; ++d) {
            powers_t[i * (max_degree + 1) + d * count] = t_power;
            powers_one_minus_t[i * (max_degree + 1) + d * count] = one_minus_t_power;
            
            t_power *= t;
            one_minus_t_power *= one_minus_t;
        }
    }
}

} // namespace bezier_simd

#endif // EULER_HAS_XSIMD

} // namespace euler::dda