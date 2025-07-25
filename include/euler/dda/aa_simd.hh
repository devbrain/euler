/**
 * @file aa_simd.hh
 * @brief SIMD-optimized antialiasing utilities
 * @ingroup DDAModule
 */
#pragma once

#include <euler/core/simd.hh>
#include <euler/dda/dda_traits.hh>
#include <euler/coordinates/point2.hh>
#include <euler/vector/vector.hh>
#include <euler/dda/dda_math.hh>
#include <array>
#include <vector>

namespace euler::dda {

#ifdef EULER_HAS_XSIMD

/**
 * @brief SIMD-optimized antialiasing functions
 * 
 * These functions provide vectorized implementations of common
 * antialiasing operations for better performance.
 */
namespace aa_simd {

/**
 * @brief Compute coverage for multiple pixels using SIMD
 * @tparam T Floating point type (float or double)
 * @param pixel_centers Array of pixel center coordinates
 * @param line_start Start point of the line
 * @param line_end End point of the line
 * @param count Number of pixels to process
 * @param coverage Output array for coverage values
 */
template<typename T>
inline void compute_line_coverage_batch(
    const point2<T>* pixel_centers,
    const point2<T>& line_start,
    const point2<T>& line_end,
    size_t count,
    T* coverage) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    constexpr size_t simd_size = simd_traits<T>::batch_size;
    
    // Precompute line parameters
    vec2<T> line_vec = line_end - line_start;
    T line_length_sq = dot(line_vec, line_vec);
    T inv_line_length = T(1) / sqrt(line_length_sq);
    
    // Normalized line direction
    vec2<T> line_dir = line_vec * inv_line_length;
    vec2<T> line_normal(-line_dir[1], line_dir[0]);
    
    // Process in SIMD batches
    size_t i = 0;
    for (; i + simd_size <= count; i += simd_size) {
        // Load pixel centers
        alignas(simd_alignment_v<T>::value) T px_data[simd_size];
        alignas(simd_alignment_v<T>::value) T py_data[simd_size];
        
        for (size_t j = 0; j < simd_size; ++j) {
            px_data[j] = pixel_centers[i + j].x;
            py_data[j] = pixel_centers[i + j].y;
        }
        
        batch_t px = batch_t::load_aligned(px_data);
        batch_t py = batch_t::load_aligned(py_data);
        
        // Vector from line start to pixel
        batch_t dx = px - batch_t(line_start.x);
        batch_t dy = py - batch_t(line_start.y);
        
        // Project onto line normal (perpendicular distance)
        batch_t dist = xsimd::abs(dx * batch_t(line_normal[0]) + dy * batch_t(line_normal[1]));
        
        // Compute coverage (1 - distance, clamped to [0, 1])
        batch_t cov = xsimd::max(batch_t(0), batch_t(1) - dist);
        
        // Store results
        cov.store_unaligned(&coverage[i]);
    }
    
    // Handle remaining pixels
    for (; i < count; ++i) {
        vec2<T> to_pixel = pixel_centers[i] - line_start;
        T dist = abs(dot(to_pixel, line_normal));
        coverage[i] = max(T(0), T(1) - dist);
    }
}

/**
 * @brief Process neighbor pixels for antialiasing using SIMD
 * @tparam T Floating point type
 * @param center Center pixel position
 * @param curve_point Exact point on the curve
 * @param tangent Curve tangent at this point (normalized)
 * @param neighbors Output array for neighbor pixels with coverage
 * @return Number of neighbor pixels generated
 */
template<typename T>
inline int process_aa_neighbors_simd(
    const point2i& center,
    const point2<T>& curve_point,
    const vec2<T>& tangent,
    aa_pixel<T>* neighbors) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    
    // Normal vector (perpendicular to tangent)
    vec2<T> normal(-tangent[1], tangent[0]);
    
    // 8-connected neighbor offsets
    constexpr int offsets[8][2] = {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},          {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };
    
    // Process neighbors in batches of 4 (or 8 if SIMD size allows)
    int count = 0;
    
    if constexpr (simd_traits<T>::batch_size >= 4) {
        alignas(simd_alignment_v<T>::value) T px_data[4];
        alignas(simd_alignment_v<T>::value) T py_data[4];
        alignas(simd_alignment_v<T>::value) T dist_data[4];
        
        // Process first 4 neighbors
        for (int j = 0; j < 4; ++j) {
            px_data[j] = static_cast<T>(center.x + offsets[j][0]) + T(0.5);
            py_data[j] = static_cast<T>(center.y + offsets[j][1]) + T(0.5);
        }
        
        batch_t px = batch_t::load_aligned(px_data);
        batch_t py = batch_t::load_aligned(py_data);
        
        // Distance from pixel centers to curve point
        batch_t dx = px - batch_t(curve_point.x);
        batch_t dy = py - batch_t(curve_point.y);
        
        // Project onto normal (perpendicular distance to tangent line)
        batch_t dist = xsimd::abs(dx * batch_t(normal[0]) + dy * batch_t(normal[1]));
        
        // Store distances
        dist.store_aligned(dist_data);
        
        // Generate pixels with coverage
        for (int j = 0; j < 4; ++j) {
            if (dist_data[j] < T(1.5)) {  // Within AA range
                neighbors[count].pos = point2<T>{
                    static_cast<T>(center.x + offsets[j][0]),
                    static_cast<T>(center.y + offsets[j][1])
                };
                neighbors[count].distance = static_cast<float>(dist_data[j]);
                neighbors[count].coverage = static_cast<float>(max(T(0), T(1) - dist_data[j]));
                count++;
            }
        }
        
        // Process remaining 4 neighbors
        for (int j = 0; j < 4; ++j) {
            px_data[j] = static_cast<T>(center.x + offsets[j + 4][0]) + T(0.5);
            py_data[j] = static_cast<T>(center.y + offsets[j + 4][1]) + T(0.5);
        }
        
        px = batch_t::load_aligned(px_data);
        py = batch_t::load_aligned(py_data);
        
        dx = px - batch_t(curve_point.x);
        dy = py - batch_t(curve_point.y);
        dist = xsimd::abs(dx * batch_t(normal[0]) + dy * batch_t(normal[1]));
        dist.store_aligned(dist_data);
        
        for (int j = 0; j < 4; ++j) {
            if (dist_data[j] < T(1.5)) {
                neighbors[count].pos = point2<T>{
                    static_cast<T>(center.x + offsets[j + 4][0]),
                    static_cast<T>(center.y + offsets[j + 4][1])
                };
                neighbors[count].distance = static_cast<float>(dist_data[j]);
                neighbors[count].coverage = static_cast<float>(max(T(0), T(1) - dist_data[j]));
                count++;
            }
        }
    } else {
        // Scalar fallback
        for (int i = 0; i < 8; ++i) {
            point2<T> pixel_center{
                static_cast<T>(center.x + offsets[i][0]) + T(0.5),
                static_cast<T>(center.y + offsets[i][1]) + T(0.5)
            };
            
            vec2<T> to_pixel = pixel_center - curve_point;
            T dist = abs(dot(to_pixel, normal));
            
            if (dist < T(1.5)) {
                neighbors[count].pos = point2<T>{
                    static_cast<T>(center.x + offsets[i][0]),
                    static_cast<T>(center.y + offsets[i][1])
                };
                neighbors[count].distance = static_cast<float>(dist);
                neighbors[count].coverage = static_cast<float>(max(T(0), T(1) - dist));
                count++;
            }
        }
    }
    
    return count;
}

/**
 * @brief Batch compute distances from points to a curve segment
 * @tparam T Floating point type
 * @param points Array of points to test
 * @param curve_start Start of curve segment
 * @param curve_end End of curve segment
 * @param curve_tangent Tangent at midpoint (normalized)
 * @param count Number of points
 * @param distances Output array for distances
 */
template<typename T>
inline void compute_curve_distances_batch(
    const point2<T>* points,
    const point2<T>& curve_start,
    const point2<T>& curve_end,
    const vec2<T>& curve_tangent,
    size_t count,
    T* distances) {
    
    using batch_t = typename simd_traits<T>::batch_type;
    constexpr size_t simd_size = simd_traits<T>::batch_size;
    
    // Curve midpoint and normal
    point2<T> midpoint{
        (curve_start.x + curve_end.x) * T(0.5),
        (curve_start.y + curve_end.y) * T(0.5)
    };
    vec2<T> normal(-curve_tangent[1], curve_tangent[0]);
    
    // Process in SIMD batches
    size_t i = 0;
    for (; i + simd_size <= count; i += simd_size) {
        alignas(simd_alignment_v<T>::value) T px_data[simd_size];
        alignas(simd_alignment_v<T>::value) T py_data[simd_size];
        
        for (size_t j = 0; j < simd_size; ++j) {
            px_data[j] = points[i + j].x;
            py_data[j] = points[i + j].y;
        }
        
        batch_t px = batch_t::load_aligned(px_data);
        batch_t py = batch_t::load_aligned(py_data);
        
        // Vector from midpoint to points
        batch_t dx = px - batch_t(midpoint.x);
        batch_t dy = py - batch_t(midpoint.y);
        
        // Distance along normal
        batch_t dist = xsimd::abs(dx * batch_t(normal[0]) + dy * batch_t(normal[1]));
        
        // Store results
        dist.store_unaligned(&distances[i]);
    }
    
    // Handle remaining points
    for (; i < count; ++i) {
        vec2<T> to_point = points[i] - midpoint;
        distances[i] = abs(dot(to_point, normal));
    }
}

} // namespace aa_simd

#endif // EULER_HAS_XSIMD

} // namespace euler::dda