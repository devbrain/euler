/**
 * @file curve_iterator.hh
 * @brief Generic curve rasterization with adaptive stepping
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>
#include <euler/dda/dda_math.hh>
#include <algorithm>
#include <functional>

namespace euler::dda {

/**
 * @brief Generic curve iterator with adaptive stepping
 * @tparam T Coordinate type
 * @tparam CurveFunc Function type for curve evaluation
 * 
 * Supports parametric, Cartesian, and polar curves.
 * Automatically adjusts step size based on local curvature.
 */
template<typename T, typename CurveFunc>
class curve_iterator : public dda_iterator_base<curve_iterator<T, CurveFunc>, pixel<int>, T> {
    using base = dda_iterator_base<curve_iterator<T, CurveFunc>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    CurveFunc curve_func_;
    curve_type type_;
    T t_current_;
    T t_end_;
    T tolerance_;
    T min_step_;
    T max_step_;
    point2<T> origin_;  // For polar curves
    
    point2i last_pixel_;
    value_type current_pixel_;
    bool first_pixel_;
    
    // Compute next parameter value using adaptive stepping
    T compute_next_t() const {
        const T h = tolerance_ * T(0.1);  // Small value for derivative estimation
        
        // Estimate first and second derivatives
        auto p0 = evaluate_curve(t_current_ - h);
        auto p1 = evaluate_curve(t_current_);
        auto p2 = evaluate_curve(t_current_ + h);
        
        vec2<T> d1 = (p2 - p0) / (T(2) * h);
        auto v1 = vec2<T>(p2.x - T(2) * p1.x + p0.x, p2.y - T(2) * p1.y + p0.y);
        vec2<T> d2 = v1 / (h * h);
        
        // Estimate curvature
        T speed = length(d1);
        T curvature = T(0);
        
        if (speed > tolerance_ * T(0.01)) {
            // κ = |v × a| / |v|³
            T cross_z = cross_2d(d1, d2);
            curvature = abs(cross_z) / (speed * speed * speed);
        }
        
        // Adaptive step size based on curvature
        T step = max_step_;
        if (curvature > T(0)) {
            // Step size inversely proportional to curvature
            step = min(max_step_, tolerance_ / (T(1) + curvature));
            step = max(min_step_, step);
        }
        
        return min(t_current_ + step, t_end_);
    }
    
    point2<T> evaluate_curve(T t) const {
        return curve_func_(t);
    }
    
public:
    /**
     * @brief Construct parametric curve iterator
     * @param f Curve function (t) -> point2<T>
     * @param t_start Starting parameter value
     * @param t_end Ending parameter value
     * @param tolerance Maximum deviation from true curve
     */
    template<typename F = CurveFunc, 
             std::enable_if_t<is_curve_function_v<F, T>, int> = 0>
    curve_iterator(F f, T t_start, T t_end, 
                   T tolerance = default_tolerance<T>())
        : curve_func_(f), type_(curve_type::parametric),
          t_current_(t_start), t_end_(t_end), 
          tolerance_(tolerance), first_pixel_(true) {
        
        min_step_ = tolerance_ * T(0.1);
        max_step_ = abs(t_end - t_start) * T(0.1);
        
        if (t_start > t_end) {
            std::swap(t_current_, t_end_);
        }
        
        auto p = evaluate_curve(t_current_);
        current_pixel_.pos = round(p);
        last_pixel_ = current_pixel_.pos;
    }
    
    /**
     * @brief Create Cartesian curve iterator (y = f(x))
     */
    template<typename F, 
             std::enable_if_t<is_cartesian_curve_function_v<F, T>, int> = 0>
    static auto cartesian(F f, T x_start, T x_end,
                         T tolerance = default_tolerance<T>()) {
        auto wrapper = [f](T x) -> point2<T> { return {x, f(x)}; };
        return curve_iterator<T, decltype(wrapper)>(wrapper, x_start, x_end, tolerance);
    }
    
    /**
     * @brief Create polar curve iterator (r = f(θ))
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
        return curve_iterator<T, decltype(wrapper)>(wrapper, theta_start, theta_end, tolerance);
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        return current_pixel_;
    }
    
    /**
     * @brief Advance to next pixel
     */
    curve_iterator& operator++() {
        if (t_current_ >= t_end_) {
            this->done_ = true;
            return *this;
        }
        
        // Use line iterator to fill gaps between curve points
        T t_next = compute_next_t();
        auto p_next = evaluate_curve(t_next);
        point2i pixel_next = round(p_next);
        
        // If we've moved more than one pixel, use line iterator to fill gap
        if (distance_squared(pixel_next, last_pixel_) > 1) {
            // Move one pixel along the line towards next position
            line_iterator<int> line(last_pixel_, pixel_next);
            if (!first_pixel_) {
                ++line;  // Skip the first pixel (already emitted)
            }
            if (line != line_iterator<int>::end()) {
                current_pixel_ = *line;
                last_pixel_ = current_pixel_.pos;
                first_pixel_ = false;
                return *this;
            }
        }
        
        // Normal case: move to next curve point
        t_current_ = t_next;
        current_pixel_.pos = pixel_next;
        last_pixel_ = pixel_next;
        first_pixel_ = false;
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    curve_iterator operator++(int) {
        curve_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper to create parametric curve iterator
 */
template<typename T, typename F>
auto make_curve_iterator(F f, T t_start, T t_end, T tolerance = default_tolerance<T>()) {
    return curve_iterator<T, F>(f, t_start, t_end, tolerance);
}

/**
 * @brief Helper to create Cartesian curve iterator
 */
template<typename T, typename F>
auto make_cartesian_curve(F f, T x_start, T x_end, T tolerance = default_tolerance<T>()) {
    return curve_iterator<T, F>::cartesian(f, x_start, x_end, tolerance);
}

/**
 * @brief Helper to create polar curve iterator
 */
template<typename T, typename F>
auto make_polar_curve(F f, T theta_start, T theta_end, 
                     point2<T> origin = {0, 0}, 
                     T tolerance = default_tolerance<T>()) {
    return curve_iterator<T, F>::polar(f, theta_start, theta_end, origin, tolerance);
}

} // namespace euler::dda