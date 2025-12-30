/**
 * @file aa_bspline_iterator.hh
 * @brief Antialiased B-spline rasterization
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/bspline_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/dda/dda_math.hh>
#include <algorithm>
#include <array>
#include <vector>

namespace euler::dda {

/**
 * @brief Antialiased B-spline iterator
 * @tparam T Coordinate type (must be floating point)
 * 
 * Generates pixels with coverage values for smooth B-spline curves.
 */
template<typename T>
class aa_bspline_iterator : public dda_iterator_base<aa_bspline_iterator<T>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>, 
                  "Antialiased B-splines require floating point coordinates");
    
    using base = dda_iterator_base<aa_bspline_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    // B-spline parameters
    std::vector<point2<T>> control_points_;
    int degree_;
    std::vector<T> knots_;
    T t_;
    T t_max_;
    T tolerance_;
    
    // Pixel buffer
    std::array<value_type, 4> pixels_;
    int pixel_count_;
    int pixel_index_;
    
    // Cox-de Boor recursion for basis functions
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
        
        T denom_left = knots_[static_cast<size_t>(i + p)] - knots_[static_cast<size_t>(i)];
        if (denom_left != T(0)) {
            left = (t - knots_[static_cast<size_t>(i)]) / denom_left * 
                   basis_function(i, p - 1, t);
        }
        
        T denom_right = knots_[static_cast<size_t>(i + p + 1)] - knots_[static_cast<size_t>(i + 1)];
        if (denom_right != T(0)) {
            right = (knots_[static_cast<size_t>(i + p + 1)] - t) / denom_right * 
                    basis_function(i + 1, p - 1, t);
        }
        
        return left + right;
    }
    
    // Evaluate B-spline at parameter t
    point2<T> evaluate(T t) const {
        point2<T> result{T(0), T(0)};
        
        // Find the knot span containing t
        int k = degree_;
        while (k < static_cast<int>(knots_.size()) - degree_ - 1 && 
               t >= knots_[static_cast<size_t>(k + 1)]) {
            k++;
        }
        
        // Compute the point using the basis functions
        for (int i = k - degree_; i <= k; ++i) {
            if (i >= 0 && i < static_cast<int>(control_points_.size())) {
                T basis = basis_function(i, degree_, t);
                auto vec = vec2<T>(basis * control_points_[static_cast<size_t>(i)].x,
                                   basis * control_points_[static_cast<size_t>(i)].y);
                result = result + vec;
            }
        }
        
        return result;
    }
    
    // Compute derivative at parameter t
    point2<T> compute_derivative(T t) const {
        const T h = T(0.0001);
        
        // Handle boundary cases
        if (t - h < knots_[static_cast<size_t>(degree_)]) {
            auto p1 = evaluate(t);
            auto p2 = evaluate(t + h);
            T scale = T(1) / h;
            return point2<T>{(p2.x - p1.x) * scale, (p2.y - p1.y) * scale};
        } else if (t + h > t_max_) {
            auto p1 = evaluate(t - h);
            auto p2 = evaluate(t);
            T scale = T(1) / h;
            return point2<T>{(p2.x - p1.x) * scale, (p2.y - p1.y) * scale};
        }
        
        // Central difference
        auto p1 = evaluate(t - h);
        auto p2 = evaluate(t + h);
        T scale = T(1) / (T(2) * h);
        return point2<T>{(p2.x - p1.x) * scale, (p2.y - p1.y) * scale};
    }
    
    // Generate antialiased pixels
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (t_ > t_max_) {
            this->done_ = true;
            return;
        }
        
        // Get current position on B-spline
        auto exact_pos = evaluate(t_);
        
        // Compute tangent and normal vectors
        auto tangent = compute_derivative(t_);
        T tangent_len = length(tangent);
        
        if (tangent_len < T(0.001)) {
            // Nearly stationary point
            auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
            p.pos = exact_pos;
            p.coverage = T(1);
            p.distance = T(0);
            
            // Advance parameter
            t_ = min(t_ + tolerance_ * T(0.1), t_max_);
            return;
        }
        
        // Normalize tangent and compute normal
        tangent = tangent / tangent_len;
        vec2<T> normal(-tangent.y, tangent.x);
        
        // Get integer pixel bounds
        T px = floor(exact_pos.x);
        T py = floor(exact_pos.y);
        
        // Check 2x2 pixel neighborhood
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                T cx = px + static_cast<T>(dx);
                T cy = py + static_cast<T>(dy);
                
                // Vector from curve point to pixel center
                vec2<T> to_pixel(cx + T(0.5) - exact_pos.x, 
                                cy + T(0.5) - exact_pos.y);
                
                // Perpendicular distance from pixel center to tangent line
                T dist = abs(dot(to_pixel, normal));
                
                // Coverage computation
                T coverage = T(0);
                if (dist < T(1.0)) {
                    // Linear falloff for simplicity
                    coverage = T(1) - dist;
                } else if (dist < T(1.5)) {
                    // Extended falloff for smoother edges
                    coverage = T(2) * (T(1.5) - dist);
                }
                
                if (coverage > T(0.01) && pixel_count_ < 4) {
                    auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
                    p.pos = point2<T>{cx, cy};
                    p.coverage = min(coverage, T(1));
                    p.distance = dist;
                }
            }
        }
        
        // Adaptive step size based on curvature
        T step = tolerance_ / (T(1) + T(10) * tangent_len);
        step = max(step, tolerance_ * T(0.01));
        step = min(step, tolerance_ * T(0.5));
        
        t_ = min(t_ + step, t_max_);
        
        // If we're at the end, mark as done
        if (t_ >= t_max_) {
            this->done_ = true;
        }
    }
    
    // Generate uniform knot vector if not provided
    void generate_uniform_knots() {
        int n = static_cast<int>(control_points_.size()) - 1;
        int m = n + degree_ + 1;
        knots_.resize(static_cast<size_t>(m + 1));
        
        // Clamped uniform B-spline
        for (int i = 0; i <= degree_; ++i) {
            knots_[static_cast<size_t>(i)] = T(0);
        }
        
        for (int i = degree_ + 1; i < m - degree_; ++i) {
            knots_[static_cast<size_t>(i)] = T(i - degree_) / T(n - degree_ + 1);
        }
        
        for (int i = m - degree_; i <= m; ++i) {
            knots_[static_cast<size_t>(i)] = T(1);
        }
    }
    
public:
    /**
     * @brief Construct antialiased B-spline iterator with uniform knots
     */
    aa_bspline_iterator(const std::vector<point2<T>>& control_points, 
                       int degree = 3,
                       T tolerance = default_tolerance<T>())
        : control_points_(control_points), degree_(degree), 
          tolerance_(tolerance) {
        
        if (control_points_.size() < static_cast<size_t>(degree_ + 1)) {
            this->done_ = true;
            return;
        }
        
        generate_uniform_knots();
        t_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        
        generate_pixels();
    }
    
    /**
     * @brief Construct antialiased B-spline iterator with custom knots
     */
    aa_bspline_iterator(const std::vector<point2<T>>& control_points,
                       const std::vector<T>& knots,
                       int degree = 3,
                       T tolerance = default_tolerance<T>())
        : control_points_(control_points), degree_(degree), 
          knots_(knots), tolerance_(tolerance) {
        
        if (control_points_.size() < static_cast<size_t>(degree_ + 1) ||
            knots_.size() != control_points_.size() + static_cast<size_t>(degree_ + 1)) {
            this->done_ = true;
            return;
        }
        
        t_ = knots_[static_cast<size_t>(degree_)];
        t_max_ = knots_[knots_.size() - static_cast<size_t>(degree_) - 1];
        
        generate_pixels();
    }
    
    value_type operator*() const {
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    aa_bspline_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            generate_pixels();
        }
        return *this;
    }};

/**
 * @brief Helper to create antialiased B-spline iterator with uniform knots
 */
template<typename T>
auto make_aa_bspline(const std::vector<point2<T>>& control_points, 
                    int degree = 3,
                    T tolerance = default_tolerance<T>()) {
    return aa_bspline_iterator<T>(control_points, degree, tolerance);
}

/**
 * @brief Helper to create antialiased B-spline iterator with custom knots
 */
template<typename T>
auto make_aa_bspline(const std::vector<point2<T>>& control_points,
                    const std::vector<T>& knots,
                    int degree = 3,
                    T tolerance = default_tolerance<T>()) {
    return aa_bspline_iterator<T>(control_points, knots, degree, tolerance);
}

/**
 * @brief Helper to create antialiased Catmull-Rom spline (interpolating)
 */
template<typename T>
auto make_aa_catmull_rom(const std::vector<point2<T>>& points,
                        T tolerance = default_tolerance<T>()) {
    // Catmull-Rom is a special case of B-spline with specific knots
    std::vector<point2<T>> control_points = points;
    
    // Add phantom points for C2 continuity at endpoints
    if (points.size() >= 2) {
        auto p0 = point2<T>{points[0].x * T(2) - points[1].x,
                           points[0].y * T(2) - points[1].y};
        auto pn = point2<T>{points.back().x * T(2) - points[points.size() - 2].x,
                           points.back().y * T(2) - points[points.size() - 2].y};
        control_points.insert(control_points.begin(), p0);
        control_points.push_back(pn);
    }
    
    // Catmull-Rom uses degree 3 with centripetal parameterization
    return make_aa_bspline(control_points, 3, tolerance);
}

} // namespace euler::dda