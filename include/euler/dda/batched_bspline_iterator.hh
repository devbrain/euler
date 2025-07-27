/**
 * @file batched_bspline_iterator.hh
 * @brief Batched B-spline curve rasterization for improved performance
 * @author Euler Library Contributors
 * @date 2024
 * 
 * This file provides batched versions of B-spline rasterization iterators that
 * generate pixels in groups for improved cache utilization. The iterators support
 * uniform and non-uniform B-splines of arbitrary degree, as well as specialized
 * Catmull-Rom splines.
 * 
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/bspline_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <algorithm>
#include <vector>

namespace euler::dda {

/**
 * @brief Batched B-spline iterator for uniform and non-uniform B-splines
 * @tparam T Coordinate type (typically float or double)
 * 
 * This iterator generates pixels along B-spline curves in batches for improved
 * performance. It supports B-splines of arbitrary degree with both uniform and
 * custom knot vectors.
 * 
 * Key features:
 * - Generates pixels in batches of up to 16 for cache efficiency
 * - Uses Cox-de Boor recursion for accurate B-spline evaluation
 * - Supports adaptive stepping based on curvature
 * - Automatically fills gaps to ensure continuous pixel chains
 * - Caches basis function values for performance
 * - Handles pending pixels to maintain continuity across batches
 * 
 * B-splines are piecewise polynomial curves defined by:
 * - Control points that influence but don't necessarily lie on the curve
 * - A knot vector that controls the parameter space
 * - The degree of the polynomial pieces
 * 
 * @note For cubic B-splines (degree 3), at least 4 control points are required
 * 
 * Example usage:
 * @code
 * std::vector<point2f> control_points = {
 *     {0, 0}, {100, 200}, {200, 200}, {300, 0}
 * };
 * auto spline = make_batched_bspline(control_points);
 * while (!spline.at_end()) {
 *     const auto& batch = spline.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_pixel(batch.pixels[i].pos);
 *     }
 *     spline.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_bspline_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = pixel<int>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // B-spline parameters
    std::vector<point2<T>> control_points_;
    std::vector<T> knots_;
    int degree_;
    
    // Parameter state
    T t_;
    T t_min_, t_max_;
    T dt_;
    T tolerance_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // Pending pixel from previous batch
    bool has_pending_pixel_;
    point2i pending_pixel_;
    
    // Prefetch distance (in parameter space)
    static constexpr T PREFETCH_T_AHEAD = T(0.1);
    
    // Cache for basis function values
    mutable std::vector<T> basis_cache_;
    mutable int last_span_ = -1;
    mutable T last_t_ = T(-1);
    
    // Cox-de Boor recursion for B-spline basis functions
    T basis_function(int i, int p, T t) const {
        if (p == 0) {
            // For the rightmost interval, include t == knots_[i+1]
            if (i == static_cast<int>(knots_.size()) - degree_ - 2 && 
                t == knots_[static_cast<size_t>(i + 1)]) {
                return T(1);
            }
            return (t >= knots_[static_cast<size_t>(i)] && t < knots_[static_cast<size_t>(i + 1)]) ? T(1) : T(0);
        }
        
        T left = T(0), right = T(0);
        
        T denom1 = knots_[static_cast<size_t>(i + p)] - knots_[static_cast<size_t>(i)];
        if (denom1 != 0) {
            left = (t - knots_[static_cast<size_t>(i)]) / denom1 * basis_function(i, p - 1, t);
        }
        
        T denom2 = knots_[static_cast<size_t>(i + p + 1)] - knots_[static_cast<size_t>(i + 1)];
        if (denom2 != 0) {
            right = (knots_[static_cast<size_t>(i + p + 1)] - t) / denom2 * basis_function(i + 1, p - 1, t);
        }
        
        return left + right;
    }
    
    // Optimized evaluation with caching
    point2<T> evaluate(T t) const {
        // Find the knot span
        int span = degree_;
        while (span < static_cast<int>(knots_.size()) - degree_ - 1 && 
               t >= knots_[static_cast<size_t>(span + 1)]) {
            span++;
        }
        
        // Check if we can reuse cached basis values
        bool use_cache = (span == last_span_ && std::abs(t - last_t_) < tolerance_ * T(0.1));
        
        if (!use_cache) {
            // Compute and cache basis functions
            basis_cache_.resize(static_cast<size_t>(degree_ + 1));
            for (int i = 0; i <= degree_; ++i) {
                int cp_index = span - degree_ + i;
                if (cp_index >= 0 && cp_index < static_cast<int>(control_points_.size())) {
                    basis_cache_[static_cast<size_t>(i)] = basis_function(cp_index, degree_, t);
                } else {
                    basis_cache_[static_cast<size_t>(i)] = T(0);
                }
            }
            last_span_ = span;
            last_t_ = t;
        }
        
        // Evaluate using cached basis functions
        point2<T> result{0, 0};
        for (int i = 0; i <= degree_; ++i) {
            int cp_index = span - degree_ + i;
            if (cp_index >= 0 && cp_index < static_cast<int>(control_points_.size())) {
                T basis = basis_cache_[static_cast<size_t>(i)];
                auto vec = vec2<T>(basis * control_points_[static_cast<size_t>(cp_index)].x,
                                   basis * control_points_[static_cast<size_t>(cp_index)].y);
                result = result + vec;
            }
        }
        
        return result;
    }
    
    point2<T> derivative(T t) const {
        point2<T> result{0, 0};
        
        if (degree_ == 0) return result;
        
        int span = degree_;
        while (span < static_cast<int>(knots_.size()) - degree_ - 1 && 
               t >= knots_[static_cast<size_t>(span + 1)]) {
            span++;
        }
        
        for (int i = 0; i < degree_; ++i) {
            int cp_index = span - degree_ + i;
            if (cp_index >= 0 && cp_index + 1 < static_cast<int>(control_points_.size())) {
                T denom = knots_[static_cast<size_t>(cp_index + degree_ + 1)] - knots_[static_cast<size_t>(cp_index + 1)];
                if (denom != 0) {
                    auto diff = control_points_[static_cast<size_t>(cp_index + 1)] - control_points_[static_cast<size_t>(cp_index)];
                    T basis = basis_function(cp_index + 1, degree_ - 1, t);
                    T scale = (static_cast<T>(degree_) / denom) * basis;
                    result = result + vec2<T>(scale * diff[0], scale * diff[1]);
                }
            }
        }
        
        return result;
    }
    
    T compute_adaptive_step(T t) const {
        auto d1 = derivative(t);
        T speed = length(d1);
        
        if (speed < tolerance_ * T(0.01)) {
            return T(0.01);
        }
        
        // Estimate curvature using finite differences
        const T h = T(0.001);
        auto d2 = derivative(std::min(t + h, t_max_));
        auto curvature_vec = (d2 - d1) * (T(1) / h);
        T curvature = length(curvature_vec) / (speed * speed);
        
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.0001), T(0.01));
    }
    
    void generate_uniform_knots() {
        int n = static_cast<int>(control_points_.size());
        int m = n + degree_ + 1;
        knots_.resize(static_cast<size_t>(m));
        
        // Clamped uniform knot vector
        for (int i = 0; i <= degree_; ++i) {
            knots_[static_cast<size_t>(i)] = T(0);
        }
        
        for (int i = degree_ + 1; i < n; ++i) {
            knots_[static_cast<size_t>(i)] = T(i - degree_) / T(n - degree_);
        }
        
        for (int i = n; i < m; ++i) {
            knots_[static_cast<size_t>(i)] = T(1);
        }
    }
    
    void fill_line_gap(point2i from, point2i to) {
        line_iterator<int> line_iter(from, to);
        ++line_iter; // Skip first pixel (already in batch)
        
        while (line_iter != line_iterator<int>::end() && !current_batch_.is_full()) {
            auto pixel = *line_iter;
            if (pixel.pos != to) {
                current_batch_.add(pixel);
            }
            ++line_iter;
        }
    }
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (done_ || t_ >= t_max_) {
            done_ = true;
            return;
        }
        
        // Add pending pixel from previous batch if any
        if (has_pending_pixel_) {
            current_batch_.add({pending_pixel_});
            has_pending_pixel_ = false;
        }
        
        // Prefetch control points for future evaluations
        T prefetch_t = std::min(t_ + PREFETCH_T_AHEAD, t_max_);
        prefetch_curve_data(prefetch_t);
        
        // Fill batch with curve pixels
        while (!current_batch_.is_full() && t_ < t_max_) {
            auto pos = evaluate(t_);
            point2i pixel = round(pos);
            
            // Check if we need to fill gaps with a line
            if (!current_batch_.is_empty()) {
                auto last = current_batch_.pixels[current_batch_.count - 1].pos;
                if (distance_squared(pixel, last) > 2) {
                    fill_line_gap(last, pixel);
                    if (current_batch_.is_full()) {
                        // Save this pixel for next batch
                        has_pending_pixel_ = true;
                        pending_pixel_ = pixel;
                        break;
                    }
                }
            }
            
            // Add the pixel if there's room
            if (!current_batch_.is_full()) {
                current_batch_.add({pixel});
                
                // Advance parameter
                t_ += dt_;
                if (t_ > t_max_) t_ = t_max_;
                
                // Update step size
                dt_ = compute_adaptive_step(t_);
            }
        }
        
        // Handle final point
        if (t_ >= t_max_ && !done_) {
            auto final_pos = evaluate(t_max_);
            point2i final_pixel = round(final_pos);
            
            if (current_batch_.is_empty() ||
                current_batch_.pixels[current_batch_.count - 1].pos != final_pixel) {
                if (!current_batch_.is_full()) {
                    current_batch_.add({final_pixel});
                }
            }
            done_ = true;
        }
    }
    
    void prefetch_curve_data(T future_t) {
        // Prefetch control points that will be needed
        int span = degree_;
        while (span < static_cast<int>(knots_.size()) - degree_ - 1 && 
               future_t >= knots_[static_cast<size_t>(span + 1)]) {
            span++;
        }
        
        for (int i = 0; i <= degree_; ++i) {
            int cp_index = span - degree_ + i;
            if (cp_index >= 0 && cp_index < static_cast<int>(control_points_.size())) {
                prefetch_hint::prefetch_for_read(&control_points_[static_cast<size_t>(cp_index)]);
            }
        }
    }
    
public:
    /**
     * @brief Construct batched B-spline iterator with uniform knots
     * @param control_points Control points defining the B-spline curve
     * @param degree Degree of the B-spline (default 3 for cubic)
     * @param tolerance Tolerance for adaptive stepping (affects curve smoothness)
     * 
     * Creates a B-spline iterator with automatically generated uniform knot vector.
     * The knot vector is clamped at both ends to ensure the curve passes through
     * the first and last control points.
     * 
     * Requirements:
     * - control_points.size() >= degree + 1
     * - degree >= 1 (linear or higher)
     * 
     * Common degree values:
     * - degree = 1: Linear B-spline (polyline)
     * - degree = 2: Quadratic B-spline
     * - degree = 3: Cubic B-spline (most common, good smoothness)
     * 
     * @throws None (sets done_ = true if requirements not met)
     */
    batched_bspline_iterator(const std::vector<point2<T>>& control_points,
                            int degree = 3,
                            T tolerance = default_tolerance<T>())
        : control_points_(control_points), degree_(degree),
          tolerance_(tolerance), done_(false), has_pending_pixel_(false) {
        
        if (control_points_.size() < static_cast<size_t>(degree_ + 1)) {
            done_ = true;
            return;
        }
        
        generate_uniform_knots();
        
        // Valid parameter range
        t_min_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        t_ = t_min_;
        
        // Initial step size
        dt_ = compute_adaptive_step(t_);
        
        // Reserve space for basis cache
        basis_cache_.reserve(static_cast<size_t>(degree_ + 1));
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Construct batched B-spline iterator with custom knot vector
     * @param control_points Control points defining the B-spline curve
     * @param knots Custom knot vector
     * @param degree Degree of the B-spline (default 3 for cubic)
     * @param tolerance Tolerance for adaptive stepping
     * 
     * Creates a B-spline iterator with a user-specified knot vector. This allows
     * for non-uniform B-splines with custom parameterization.
     * 
     * Requirements:
     * - knots.size() == control_points.size() + degree + 1
     * - Knot values must be non-decreasing
     * - At least degree+1 knots at each end should be equal (for clamped curves)
     * 
     * The valid parameter range is [knots[degree], knots[n-degree-1]] where
     * n is the size of the knot vector.
     * 
     * @throws None (sets done_ = true if requirements not met)
     */
    batched_bspline_iterator(const std::vector<point2<T>>& control_points,
                            const std::vector<T>& knots,
                            int degree = 3,
                            T tolerance = default_tolerance<T>())
        : control_points_(control_points), knots_(knots), degree_(degree),
          tolerance_(tolerance), done_(false), has_pending_pixel_(false) {
        
        // Validate knot vector
        if (knots_.size() != control_points_.size() + degree_ + 1) {
            done_ = true;
            return;
        }
        
        // Valid parameter range
        t_min_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        t_ = t_min_;
        
        // Initial step size
        dt_ = compute_adaptive_step(t_);
        
        // Reserve space for basis cache
        basis_cache_.reserve(static_cast<size_t>(degree_ + 1));
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Get the current batch of pixels
     * @return Const reference to the current pixel batch
     * 
     * Returns the current batch containing up to 16 pixels. The actual
     * number of valid pixels is indicated by the batch's count member.
     */
    const batch_type& current_batch() const {
        return current_batch_;
    }
    
    /**
     * @brief Check if the iterator has finished generating all pixels
     * @return true if no more pixels are available, false otherwise
     */
    bool at_end() const {
        return done_ && current_batch_.is_empty();
    }
    
    /**
     * @brief Advance to the next batch of pixels
     * @return Reference to this iterator for chaining
     * 
     * Fills the internal batch with the next set of pixels, automatically
     * handling gap filling and continuity. If the iterator has already
     * generated all pixels, this method clears the current batch.
     */
    batched_bspline_iterator& next_batch() {
        if (!done_) {
            fill_batch();
        } else {
            current_batch_.clear();
        }
        return *this;
    }
    
    /**
     * @brief Process all pixels using a callback function
     * @tparam Callback Callable type accepting a const pixel_batch<pixel<int>>& parameter
     * @param callback Function to call for each batch of pixels
     * 
     * This convenience method iterates through all batches and calls the
     * provided callback for each non-empty batch.
     */
    template<typename Callback>
    void process_all(Callback&& callback) {
        while (!at_end()) {
            callback(current_batch_);
            next_batch();
        }
    }
};

/**
 * @brief Batched Catmull-Rom spline iterator
 * @tparam T Coordinate type (typically float or double)
 * 
 * Catmull-Rom splines are a special case of cubic B-splines that interpolate
 * through all control points (except possibly the first and last). They provide
 * C1 continuity with automatic tangent calculation based on neighboring points.
 * 
 * Key features:
 * - Interpolates through all given points
 * - Automatically calculates tangents for smooth curves
 * - No need to specify control points that are off the curve
 * - Particularly useful for animation paths and smooth interpolation
 * 
 * The conversion to B-spline form allows reuse of the efficient B-spline
 * rasterization algorithm while providing the convenience of interpolation.
 * 
 * @note Requires at least 4 points for proper interpolation
 */
template<typename T>
class batched_catmull_rom_iterator : public batched_bspline_iterator<T> {
private:
    std::vector<point2<T>> generate_bspline_controls(const std::vector<point2<T>>& points) {
        if (points.size() < 4) {
            return points;
        }
        
        std::vector<point2<T>> controls;
        controls.reserve(points.size() + 2);
        
        // Convert Catmull-Rom to B-spline control points
        controls.push_back(points[0]);
        
        for (size_t i = 0; i < points.size() - 1; ++i) {
            if (i == 0) {
                auto tangent = (points[i + 1] - points[i]) * T(0.5);
                controls.push_back(points[i] + vec2<T>(tangent[0] * T(1.0/3.0), tangent[1] * T(1.0/3.0)));
            } else if (i == points.size() - 2) {
                auto tangent = (points[i + 1] - points[i]) * T(0.5);
                controls.push_back(points[i + 1] - vec2<T>(tangent[0] * T(1.0/3.0), tangent[1] * T(1.0/3.0)));
            } else {
                auto tangent0 = (points[i + 1] - points[i - 1]) * T(0.5);
                auto tangent1 = (points[i + 2] - points[i]) * T(0.5);
                controls.push_back(points[i] + vec2<T>(tangent0[0] * T(1.0/3.0), tangent0[1] * T(1.0/3.0)));
                controls.push_back(points[i + 1] - vec2<T>(tangent1[0] * T(1.0/3.0), tangent1[1] * T(1.0/3.0)));
            }
        }
        
        controls.push_back(points.back());
        
        return controls;
    }
    
public:
    /**
     * @brief Construct Catmull-Rom spline iterator from interpolation points
     * @param points Points to interpolate through
     * @param tolerance Tolerance for adaptive stepping
     * 
     * Creates a Catmull-Rom spline that passes through the given points.
     * The curve will have continuous first derivatives (C1 continuity) at
     * each interior point.
     * 
     * For best results, provide at least 4 points. With fewer points:
     * - 2 points: Degenerates to a line
     * - 3 points: Limited tangent estimation at endpoints
     * 
     * The algorithm automatically converts the interpolation points to
     * B-spline control points for efficient evaluation.
     */
    batched_catmull_rom_iterator(const std::vector<point2<T>>& points,
                                T tolerance = default_tolerance<T>())
        : batched_bspline_iterator<T>(generate_bspline_controls(points), 3, tolerance) {
    }
};

/**
 * @brief Create a batched B-spline iterator with uniform knots
 * @tparam T Coordinate type
 * @param control_points Control points defining the B-spline
 * @param degree Degree of the B-spline (default 3 for cubic)
 * @param tolerance Tolerance for adaptive stepping
 * @return A batched_bspline_iterator instance
 * 
 * Factory function that creates a batched B-spline iterator with automatic
 * uniform knot generation. This is the preferred way to create B-spline
 * iterators as it provides type deduction.
 * 
 * @see batched_bspline_iterator
 */
template<typename T>
auto make_batched_bspline(const std::vector<point2<T>>& control_points,
                         int degree = 3,
                         T tolerance = default_tolerance<T>()) {
    return batched_bspline_iterator<T>(control_points, degree, tolerance);
}

/**
 * @brief Create a batched Catmull-Rom spline iterator
 * @tparam T Coordinate type
 * @param points Points to interpolate through
 * @param tolerance Tolerance for adaptive stepping
 * @return A batched_catmull_rom_iterator instance
 * 
 * Factory function that creates a batched Catmull-Rom spline iterator.
 * The resulting curve will pass through all given points with smooth
 * tangent continuity.
 * 
 * @see batched_catmull_rom_iterator
 */
template<typename T>
auto make_batched_catmull_rom(const std::vector<point2<T>>& points,
                             T tolerance = default_tolerance<T>()) {
    return batched_catmull_rom_iterator<T>(points, tolerance);
}

} // namespace euler::dda