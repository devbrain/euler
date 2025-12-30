/**
 * @file aa_curve_iterator.hh
 * @brief Antialiased curve rasterization
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/curve_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/dda/aa_simd.hh>
#include <euler/core/error.hh>
#include <algorithm>
#include <array>

namespace euler::dda {

/**
 * @brief Antialiased curve iterator with adaptive stepping
 * @tparam T Coordinate type (must be floating point)
 * @tparam CurveFunc Function type for curve evaluation
 * 
 * Generates pixels with coverage values for smooth curves.
 * Uses the curve tangent to compute perpendicular distance for coverage.
 */
template<typename T, typename CurveFunc>
class aa_curve_iterator : public dda_iterator_base<aa_curve_iterator<T, CurveFunc>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>, 
                  "Antialiased curves require floating point coordinates");
    
    using base = dda_iterator_base<aa_curve_iterator<T, CurveFunc>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    // Curve function and parameters
    CurveFunc curve_func_;
    T t_current_;
    T t_end_;
    T tolerance_;
    
    // Pixel buffer for current position
    // Need space for up to 9 pixels: 1 center + 8 neighbors
    std::array<value_type, 9> pixels_{};  // Value-initialize to zeros
    int pixel_count_;
    int pixel_index_;
    
    // Evaluate curve at parameter t
    point2<T> evaluate_curve(T t) const {
        return curve_func_(t);
    }
    
    // Compute derivative at parameter t
    point2<T> compute_derivative(T t) const {
        const T h = T(0.0001);
        auto p1 = evaluate_curve(t - h);
        auto p2 = evaluate_curve(t + h);
        T scale = T(1) / (T(2) * h);
        return point2<T>{(p2.x - p1.x) * scale, (p2.y - p1.y) * scale};
    }
    
    // Generate antialiased pixels for current curve position
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (t_current_ >= t_end_) {
            this->done_ = true;
            return;
        }
        
        // Get current position on curve
        auto exact_pos = evaluate_curve(t_current_);
        
        // Compute tangent and normal vectors
        auto tangent = compute_derivative(t_current_);
        T tangent_len = length(tangent);
        
        if (tangent_len < T(0.001)) {
            // Nearly stationary point, just output center pixel
            auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
            p.pos = exact_pos;
            p.coverage = T(1);
            p.distance = T(0);
            
            // Advance parameter
            t_current_ = min(t_current_ + tolerance_, t_end_);
            return;
        }
        
        // Normalize tangent and compute normal
        tangent = tangent / tangent_len;
        vec2<T> normal(-tangent.y, tangent.x);
        
        // Get integer pixel center
        T px = floor(exact_pos.x);
        T py = floor(exact_pos.y);
        point2i center{static_cast<int>(px), static_cast<int>(py)};
        
#ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            // Use SIMD to process neighbor pixels
            pixel_count_ = aa_simd::process_aa_neighbors_simd(
                center, exact_pos, vec2<T>(tangent.x, tangent.y), 
                pixels_.data());
            
            // Always include the center pixel with full coverage
            bool center_included = false;
            for (int i = 0; i < pixel_count_; ++i) {
                if (static_cast<int>(pixels_[static_cast<size_t>(i)].pos.x) == center.x && 
                    static_cast<int>(pixels_[static_cast<size_t>(i)].pos.y) == center.y) {
                    center_included = true;
                    break;
                }
            }
            
            if (!center_included && pixel_count_ < 9) {
                auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
                p.pos = exact_pos;
                p.coverage = T(1);
                p.distance = T(0);
            }
        } else
#endif
        {
            // Original scalar implementation
            // Add center pixel with the exact floating point position
            auto& p0 = pixels_[static_cast<size_t>(pixel_count_++)];
            p0.pos = exact_pos;  // Use the exact position for now
            p0.coverage = T(1);
            p0.distance = T(0);
            
            // Compute distance from pixel centers to curve
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dx = 0; dx <= 1; ++dx) {
                    T cx = px + static_cast<T>(dx);
                    T cy = py + static_cast<T>(dy);
                
                // Vector from curve point to pixel center
                vec2<T> to_pixel(cx + T(0.5) - exact_pos.x, 
                                cy + T(0.5) - exact_pos.y);
                
                    // Distance from pixel center to curve (approximated)
                    T dist = abs(dot(to_pixel, normal));
                    
                    // Coverage based on distance (using error function approximation)
                    T coverage = T(0);
                    if (dist < T(1.5)) {
                        coverage = T(0.5) * (T(1) - dist / T(1.5));
                        if (dist < T(0.5)) {
                            coverage = T(1) - T(0.5) * dist;
                        }
                    }
                    
                    if (coverage > T(0.01) && pixel_count_ < 9) {
                        auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
                        p.pos = point2<T>{cx, cy};
                        p.coverage = coverage;
                        p.distance = dist;
                    }
                }
            }
        }
        
        // Advance parameter with adaptive stepping
        T step = tolerance_ / (T(1) + tangent_len * T(0.1));
        step = max(step, tolerance_ * T(0.01));
        step = min(step, tolerance_);
        t_current_ = min(t_current_ + step, t_end_);
        
        // Don't mark as done yet - we still have pixels to output
        // done_ will be set when generate_pixels is called with no more curve to process
    }
    
public:
    /**
     * @brief Construct antialiased parametric curve iterator
     */
    template<typename F = CurveFunc, 
             std::enable_if_t<is_curve_function_v<F, T>, int> = 0>
    aa_curve_iterator(F f, T t_start, T t_end, T tolerance = default_tolerance<T>())
        : curve_func_(f), t_current_(t_start), t_end_(t_end), tolerance_(tolerance),
          pixel_count_(0), pixel_index_(0) {
        if (t_start > t_end) {
            std::swap(t_current_, t_end_);
        }
        generate_pixels();
    }
    
    /**
     * @brief Create antialiased Cartesian curve iterator
     */
    template<typename F, 
             std::enable_if_t<is_cartesian_curve_function_v<F, T>, int> = 0>
    static auto cartesian(F f, T x_start, T x_end,
                         T tolerance = default_tolerance<T>()) {
        auto wrapper = [f](T x) -> point2<T> { return {x, f(x)}; };
        return aa_curve_iterator<T, decltype(wrapper)>(wrapper, x_start, x_end, tolerance);
    }
    
    /**
     * @brief Create antialiased polar curve iterator
     */
    template<typename F, 
             std::enable_if_t<is_polar_curve_function_v<F, T>, int> = 0>
    static auto polar(F f, T theta_start, T theta_end,
                     point2<T> origin = {0, 0},
                     T tolerance = default_tolerance<T>()) {
        auto wrapper = [f, origin](T theta) -> point2<T> {
            T r = f(theta);
            return point2<T>{origin.x + r * cos(theta), origin.y + r * sin(theta)};
        };
        return aa_curve_iterator<T, decltype(wrapper)>(wrapper, theta_start, theta_end, tolerance);
    }
    
    value_type operator*() const {
        EULER_CHECK_INDEX(pixel_index_, pixel_count_);
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    aa_curve_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            generate_pixels();
        }
        return *this;
    }
    
    // Override is_done to check pixel consumption
    constexpr bool is_done() const {
        return this->done_ && pixel_index_ >= pixel_count_;
    }};

/**
 * @brief Helper to create antialiased parametric curve iterator
 */
template<typename T, typename F>
auto make_aa_curve_iterator(F f, T t_start, T t_end, T tolerance = default_tolerance<T>()) {
    return aa_curve_iterator<T, F>(f, t_start, t_end, tolerance);
}

/**
 * @brief Helper to create antialiased Cartesian curve iterator
 */
template<typename T, typename F>
auto make_aa_cartesian_curve(F f, T x_start, T x_end, T tolerance = default_tolerance<T>()) {
    auto wrapper = [f](T x) -> point2<T> { return {x, f(x)}; };
    return aa_curve_iterator<T, decltype(wrapper)>(wrapper, x_start, x_end, tolerance);
}

/**
 * @brief Helper to create antialiased polar curve iterator
 */
template<typename T, typename F>
auto make_aa_polar_curve(F f, T theta_start, T theta_end, 
                        point2<T> origin = {0, 0}, 
                        T tolerance = default_tolerance<T>()) {
    auto wrapper = [f, origin](T theta) -> point2<T> {
        T r = f(theta);
        return point2<T>{origin.x + r * cos(theta), origin.y + r * sin(theta)};
    };
    return aa_curve_iterator<T, decltype(wrapper)>(wrapper, theta_start, theta_end, tolerance);
}

} // namespace euler::dda