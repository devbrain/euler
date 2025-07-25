/**
 * @file bspline_iterator.hh
 * @brief B-spline curve rasterization
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <algorithm>
#include <vector>

namespace euler::dda {

/**
 * @brief B-spline iterator for uniform cubic B-splines
 * @tparam T Coordinate type
 * 
 * Rasterizes B-spline curves using Cox-de Boor recursion.
 */
template<typename T>
class bspline_iterator : public dda_iterator_base<bspline_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<bspline_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    std::vector<point2<T>> control_points_;
    std::vector<T> knots_;
    int degree_;
    int num_segments_;
    
    // Current state
    T t_;
    T t_min_, t_max_;
    T dt_;
    T tolerance_;
    
    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;
    
    // Cox-de Boor recursion for B-spline basis functions
    T basis_function(int i, int p, T t) const {
        if (p == 0) {
            // For the rightmost interval, we need to include t == knots_[i+1]
            // when it's the last valid interval
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
    
    point2<T> evaluate(T t) const {
        point2<T> result{0, 0};
        
        // Find the knot span
        int span = degree_;
        while (span < static_cast<int>(knots_.size()) - degree_ - 1 && 
               t >= knots_[static_cast<size_t>(span + 1)]) {
            span++;
        }
        
        // Evaluate using basis functions
        for (int i = 0; i <= degree_; ++i) {
            int cp_index = span - degree_ + i;
            if (cp_index >= 0 && cp_index < static_cast<int>(control_points_.size())) {
                T basis = basis_function(cp_index, degree_, t);
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
        
        // Derivative of B-spline is a B-spline of degree p-1
        // with control points Q_i = p * (P_{i+1} - P_i) / (t_{i+p+1} - t_{i+1})
        
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
    
    T compute_adaptive_step(T t) const {
        const T h = T(0.001);
        auto d1 = derivative(t);
        vec2<T> d2 = (derivative(t + h) - d1) / h;
        
        T speed = length(d1);
        if (speed < tolerance_ * T(0.01)) {
            return T(0.1);
        }
        
        T cross_z = cross_2d(vec2<T>(d1.x, d1.y), d2);
        T curvature = abs(cross_z) / (speed * speed * speed);
        
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.001), T(0.1));
    }
    
public:
    /**
     * @brief Construct uniform B-spline iterator
     * @param control_points Control polygon vertices
     * @param degree B-spline degree (default 3 for cubic)
     * @param tolerance Maximum pixel deviation
     */
    bspline_iterator(const std::vector<point2<T>>& control_points,
                    int degree = 3,
                    T tolerance = default_tolerance<T>())
        : control_points_(control_points), degree_(degree),
          tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        if (control_points_.size() < static_cast<size_t>(degree_ + 1)) {
            this->done_ = true;
            return;
        }
        
        generate_uniform_knots();
        
        // Valid parameter range
        t_min_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        t_ = t_min_;
        
        auto start = evaluate(t_);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step(t_);
    }
    
    /**
     * @brief Construct B-spline with custom knot vector
     */
    bspline_iterator(const std::vector<point2<T>>& control_points,
                    const std::vector<T>& knots,
                    int degree = 3,
                    T tolerance = default_tolerance<T>())
        : control_points_(control_points), knots_(knots), degree_(degree),
          tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        // Validate knot vector
        if (knots_.size() != control_points_.size() + static_cast<size_t>(degree_) + 1) {
            this->done_ = true;
            return;
        }
        
        t_min_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        t_ = t_min_;
        
        auto start = evaluate(t_);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step(t_);
    }
    
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return {last_pixel_};
    }
    
    bspline_iterator& operator++() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                return *this;
            }
            using_line_ = false;
            // Don't continue to next spline point - we already emitted it as the end of the line
            return *this;
        }
        
        if (t_ >= t_max_) {
            this->done_ = true;
            return *this;
        }
        
        t_ += dt_;
        if (t_ > t_max_) t_ = t_max_;
        
        auto next_point = evaluate(t_);
        point2i next_pixel = round(next_point);
        
        if (distance_squared(next_pixel, last_pixel_) > 1) {
            line_iter_ = line_iterator<int>(last_pixel_, next_pixel);
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                using_line_ = true;
                // Update last_pixel_ to the target of the line so we continue from there
                last_pixel_ = next_pixel;
                return *this;
            }
        }
        
        last_pixel_ = next_pixel;
        dt_ = compute_adaptive_step(t_);
        
        return *this;
    }
    
    bspline_iterator operator++(int) {
        bspline_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Catmull-Rom spline iterator (special case of B-spline)
 * @tparam T Coordinate type
 * 
 * Interpolating spline that passes through control points.
 */
template<typename T>
class catmull_rom_iterator : public dda_iterator_base<catmull_rom_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<catmull_rom_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    std::vector<point2<T>> points_;
    T tension_;  // 0.5 for standard Catmull-Rom
    
    int current_segment_;
    T t_;
    T dt_;
    T tolerance_;
    
    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;
    
    point2<T> interpolate(int i, T t) const {
        // Get four control points for segment i
        auto p0 = (i > 0) ? points_[static_cast<size_t>(i - 1)] : points_[0];
        auto p1 = points_[static_cast<size_t>(i)];
        auto p2 = points_[static_cast<size_t>(i + 1)];
        auto p3 = (i + 2 < static_cast<int>(points_.size())) ? 
                  points_[static_cast<size_t>(i + 2)] : points_.back();
        
        // Catmull-Rom matrix multiplication
        T t2 = t * t;
        T t3 = t2 * t;
        
        auto d0 = p2 - p0;
        auto d1 = p3 - p1;
        auto d2 = p2 - p1;
        auto d3 = p1 - p2;
        
        return point2<T>{
            p1.x + tension_ * d0[0] * t + 
            (T(3) * d2[0] - T(2) * tension_ * d0[0] - tension_ * d1[0]) * t2 +
            (T(2) * d3[0] + tension_ * d0[0] + tension_ * d1[0]) * t3,
            
            p1.y + tension_ * d0[1] * t + 
            (T(3) * d2[1] - T(2) * tension_ * d0[1] - tension_ * d1[1]) * t2 +
            (T(2) * d3[1] + tension_ * d0[1] + tension_ * d1[1]) * t3
        };
    }
    
    point2<T> derivative(int i, T t) const {
        auto p0 = (i > 0) ? points_[static_cast<size_t>(i - 1)] : points_[0];
        auto p1 = points_[static_cast<size_t>(i)];
        auto p2 = points_[static_cast<size_t>(i + 1)];
        auto p3 = (i + 2 < static_cast<int>(points_.size())) ? 
                  points_[static_cast<size_t>(i + 2)] : points_.back();
        
        T t2 = t * t;
        
        auto d0 = p2 - p0;
        auto d1 = p3 - p1;
        auto d2 = p2 - p1;
        auto d3 = p1 - p2;
        
        return point2<T>{
            tension_ * d0[0] + 
            (T(6) * d2[0] - T(4) * tension_ * d0[0] - T(2) * tension_ * d1[0]) * t +
            (T(6) * d3[0] + T(3) * tension_ * d0[0] + T(3) * tension_ * d1[0]) * t2,
            
            tension_ * d0[1] + 
            (T(6) * d2[1] - T(4) * tension_ * d0[1] - T(2) * tension_ * d1[1]) * t +
            (T(6) * d3[1] + T(3) * tension_ * d0[1] + T(3) * tension_ * d1[1]) * t2
        };
    }
    
    T compute_adaptive_step() const {
        const T h = T(0.001);
        auto d1 = derivative(current_segment_, t_);
        vec2<T> d2 = (derivative(current_segment_, t_ + h) - d1) / h;
        
        T speed = length(d1);
        if (speed < tolerance_ * T(0.01)) {
            return T(0.1);
        }
        
        T cross_z = cross_2d(vec2<T>(d1.x, d1.y), d2);
        T curvature = abs(cross_z) / (speed * speed * speed);
        
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.001), T(0.1));
    }
    
public:
    /**
     * @brief Construct Catmull-Rom spline iterator
     * @param points Points to interpolate through
     * @param tension Tension parameter (0.5 = standard)
     * @param tolerance Maximum pixel deviation
     */
    catmull_rom_iterator(const std::vector<point2<T>>& points,
                        T tension = T(0.5),
                        T tolerance = default_tolerance<T>())
        : points_(points), tension_(tension), 
          current_segment_(0), t_(0), tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        if (points_.size() < 2) {
            this->done_ = true;
            return;
        }
        
        auto start = interpolate(0, 0);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step();
    }
    
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return {last_pixel_};
    }
    
    catmull_rom_iterator& operator++() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                return *this;
            }
            using_line_ = false;
            // Don't continue to next spline point - we already emitted it as the end of the line
            return *this;
        }
        
        t_ += dt_;
        
        if (t_ >= T(1)) {
            current_segment_++;
            if (current_segment_ >= static_cast<int>(points_.size()) - 1) {
                this->done_ = true;
                return *this;
            }
            t_ -= T(1);
        }
        
        auto next_point = interpolate(current_segment_, t_);
        point2i next_pixel = round(next_point);
        
        if (distance_squared(next_pixel, last_pixel_) > 1) {
            line_iter_ = line_iterator<int>(last_pixel_, next_pixel);
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                using_line_ = true;
                // Update last_pixel_ to the target of the line so we continue from there
                last_pixel_ = next_pixel;
                return *this;
            }
        }
        
        last_pixel_ = next_pixel;
        dt_ = compute_adaptive_step();
        
        return *this;
    }
    
    catmull_rom_iterator operator++(int) {
        catmull_rom_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper functions
 */
template<typename T>
auto make_bspline(const std::vector<point2<T>>& control_points,
                 int degree = 3,
                 T tolerance = default_tolerance<T>()) {
    return bspline_iterator<T>(control_points, degree, tolerance);
}

template<typename T>
auto make_bspline(const std::vector<point2<T>>& control_points,
                 const std::vector<T>& knots,
                 int degree = 3,
                 T tolerance = default_tolerance<T>()) {
    return bspline_iterator<T>(control_points, knots, degree, tolerance);
}

template<typename T>
auto make_catmull_rom(const std::vector<point2<T>>& points,
                     T tension = T(0.5),
                     T tolerance = default_tolerance<T>()) {
    return catmull_rom_iterator<T>(points, tension, tolerance);
}

} // namespace euler::dda