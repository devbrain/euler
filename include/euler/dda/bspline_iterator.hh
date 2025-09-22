/**
 * @file bspline_iterator.hh
 * @brief B-spline curve rasterization with C++20 concepts
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/direct/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/concepts.hh>
#include <algorithm>
#include <vector>
#include <span>
#include <ranges>

namespace euler::dda {

/**
 * @brief B-spline iterator with C++20 concepts for flexible point containers
 * @tparam T Coordinate type
 *
 * Rasterizes B-spline curves using Cox-de Boor recursion.
 * Now accepts any container or range of points.
 */
template<typename T>
class bspline_iterator : public dda_iterator_base<bspline_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<bspline_iterator<T>, pixel<int>, T>;

public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;

    // Default constructor for end sentinel
    bspline_iterator() : line_iter_{point2i{0,0}, point2i{0,0}} {
        this->done_ = true;
    }

    // Static end sentinel
    static bspline_iterator end() {
        bspline_iterator it;
        it.done_ = true;
        return it;
    }

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

        // Compute only the non-zero basis functions
        int first_basis = std::max(0, span - degree_);
        int last_basis = std::min(static_cast<int>(control_points_.size()) - 1, span);

        for (int i = first_basis; i <= last_basis; ++i) {
            T basis = basis_function(i, degree_, t);
            result.x += basis * control_points_[static_cast<size_t>(i)].x;
            result.y += basis * control_points_[static_cast<size_t>(i)].y;
        }

        return result;
    }

    // Estimate step size based on curvature
    T estimate_step_size(T t) const {
        const T base_step = T(0.001);
        const T max_step = T(0.1);

        // Sample points for curvature estimation
        point2<T> p0 = evaluate(std::max(t_min_, t - base_step));
        point2<T> p1 = evaluate(t);
        point2<T> p2 = evaluate(std::min(t_max_, t + base_step));

        // Estimate curvature using second derivative approximation
        vec2<T> v1 = vec2<T>(p1.x - p0.x, p1.y - p0.y);
        vec2<T> v2 = vec2<T>(p2.x - p1.x, p2.y - p1.y);

        T speed = euler::direct::norm(v1);
        if (speed < T(0.001)) {
            return max_step;
        }

        vec2<T> accel = v2 - v1;
        T curvature = euler::direct::norm(accel) / (speed * speed);

        // Adaptive step size based on curvature
        if (curvature < T(0.001)) {
            return max_step;
        }

        T step = std::min(max_step, tolerance_ / (curvature * speed));
        return std::max(base_step, step);
    }

    void advance_to_next_pixel() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ == decltype(line_iter_)::end()) {
                using_line_ = false;
            } else {
                return;
            }
        }

        // Safety counter to prevent infinite loops
        int iterations = 0;
        const int max_iterations = 10000;

        while (t_ <= t_max_ && !this->done_ && iterations < max_iterations) {
            point2<T> current = evaluate(t_);
            point2i current_pixel{static_cast<int>(std::round(current.x)),
                                 static_cast<int>(std::round(current.y))};

            if (current_pixel != last_pixel_) {
                // Check if we need to fill gaps
                int dx = std::abs(current_pixel.x - last_pixel_.x);
                int dy = std::abs(current_pixel.y - last_pixel_.y);

                if (dx > 1 || dy > 1) {
                    // Use line iterator to fill gap
                    line_iter_ = make_line_iterator(last_pixel_, current_pixel);
                    ++line_iter_; // Skip the first pixel (already drawn)
                    if (line_iter_ != decltype(line_iter_)::end()) {
                        using_line_ = true;
                        last_pixel_ = current_pixel;
                        return;
                    }
                }

                last_pixel_ = current_pixel;
                return;
            }

            // Adaptive step size
            dt_ = estimate_step_size(t_);

            // Make sure we don't skip past t_max_
            if (t_ + dt_ > t_max_) {
                dt_ = t_max_ - t_;
            }

            t_ += dt_;
            iterations++;
        }

        this->done_ = true;
    }

    void init_uniform_knots(size_t n) {
        size_t m = n + degree_ + 1;
        knots_.resize(m);

        // Clamped knot vector
        for (int i = 0; i <= degree_; ++i) {
            knots_[static_cast<size_t>(i)] = T(0);
        }

        for (size_t i = static_cast<size_t>(degree_ + 1); i < m - static_cast<size_t>(degree_ + 1); ++i) {
            knots_[i] = static_cast<T>(i - static_cast<size_t>(degree_)) /
                       static_cast<T>(n - static_cast<size_t>(degree_));
        }

        for (size_t i = m - static_cast<size_t>(degree_ + 1); i < m; ++i) {
            knots_[i] = T(1);
        }
    }

public:
    /**
     * @brief Construct B-spline from any point container (C++20)
     * @tparam PointContainer Any container satisfying point2_container concept
     * @param control_points Container of control points
     * @param degree B-spline degree (default 3 for cubic)
     * @param tolerance Maximum pixel deviation
     */
    template<point2_container PointContainer>
    bspline_iterator(const PointContainer& control_points,
                    int degree = 3,
                    T tolerance = default_tolerance<T>())
        : control_points_(std::begin(control_points), std::end(control_points)),
          degree_(degree),
          tolerance_(tolerance),
          using_line_(false),
          line_iter_{point2i{0,0}, point2i{0,0}} {

        if (control_points_.size() < static_cast<size_t>(degree_ + 1)) {
            this->done_ = true;
            return;
        }

        init_uniform_knots(control_points_.size());

        // Valid parameter range - for clamped B-splines, we want to go from 0 to 1
        t_min_ = T(0);
        t_max_ = T(1);
        t_ = t_min_;
        dt_ = T(0.001);

        num_segments_ = static_cast<int>(control_points_.size()) - degree_;

        point2<T> start = evaluate(t_min_);
        last_pixel_ = point2i{static_cast<int>(std::round(start.x)),
                             static_cast<int>(std::round(start.y))};
    }


    /**
     * @brief Construct B-spline with custom knot vector (accepts any container)
     */
    template<point2_container PointContainer, typename KnotContainer>
        requires std::ranges::sized_range<KnotContainer>
    bspline_iterator(const PointContainer& control_points,
                    const KnotContainer& knots,
                    int degree = 3,
                    T tolerance = default_tolerance<T>())
        : control_points_(std::begin(control_points), std::end(control_points)),
          knots_(std::begin(knots), std::end(knots)),
          degree_(degree),
          tolerance_(tolerance),
          using_line_(false),
          line_iter_{point2i{0,0}, point2i{0,0}} {

        if (control_points_.size() < static_cast<size_t>(degree_ + 1) ||
            knots_.size() != control_points_.size() + static_cast<size_t>(degree_ + 1)) {
            this->done_ = true;
            return;
        }

        // Valid parameter range - for clamped B-splines, we want to go from 0 to 1
        t_min_ = T(0);
        t_max_ = T(1);
        t_ = t_min_;
        dt_ = T(0.001);

        num_segments_ = static_cast<int>(control_points_.size()) - degree_;

        point2<T> start = evaluate(t_min_);
        last_pixel_ = point2i{static_cast<int>(std::round(start.x)),
                             static_cast<int>(std::round(start.y))};
    }


    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return pixel<int>{last_pixel_};
    }

    /**
     * @brief Advance to next pixel
     */
    bspline_iterator& operator++() {
        advance_to_next_pixel();
        return *this;
    }

    /**
     * @brief Post-increment
     */
    bspline_iterator operator++(int) {
        bspline_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
};

/**
 * @brief Catmull-Rom spline iterator (special case of B-spline)
 * @tparam T Coordinate type
 */
template<typename T>
class catmull_rom_iterator : public dda_iterator_base<catmull_rom_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<catmull_rom_iterator<T>, pixel<int>, T>;

public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;

    // Default constructor for end sentinel
    catmull_rom_iterator() : line_iter_{point2i{0,0}, point2i{0,0}} {
        this->done_ = true;
    }

    // Static end sentinel
    static catmull_rom_iterator end() {
        catmull_rom_iterator it;
        it.done_ = true;
        return it;
    }

private:
    std::vector<point2<T>> points_;
    T tension_;  // 0.5 for standard Catmull-Rom

    // Current state
    int current_segment_;
    T t_;
    T dt_;
    T tolerance_;

    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;

    point2<T> evaluate_segment(int segment, T t) const {
        if (segment < 0 || segment >= static_cast<int>(points_.size()) - 1) {
            return point2<T>{0, 0};
        }

        // Get control points for this segment
        point2<T> p0 = (segment > 0) ? points_[static_cast<size_t>(segment - 1)] : points_[0];
        point2<T> p1 = points_[static_cast<size_t>(segment)];
        point2<T> p2 = points_[static_cast<size_t>(segment + 1)];
        point2<T> p3 = (segment < static_cast<int>(points_.size()) - 2) ?
                      points_[static_cast<size_t>(segment + 2)] : points_.back();

        // Catmull-Rom basis matrix with tension
        T t2 = t * t;
        T t3 = t2 * t;

        T b0 = -tension_ * t3 + 2 * tension_ * t2 - tension_ * t;
        T b1 = (2 - tension_) * t3 + (tension_ - 3) * t2 + 1;
        T b2 = (tension_ - 2) * t3 + (3 - 2 * tension_) * t2 + tension_ * t;
        T b3 = tension_ * t3 - tension_ * t2;

        return point2<T>{
            b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x,
            b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y
        };
    }

    T estimate_step_size() const {
        const T base_step = T(0.01);
        const T max_step = T(0.1);

        // Sample points for curvature estimation
        point2<T> p0 = evaluate_segment(current_segment_, std::max(T(0), t_ - base_step));
        point2<T> p1 = evaluate_segment(current_segment_, t_);
        point2<T> p2 = evaluate_segment(current_segment_, std::min(T(1), t_ + base_step));

        // Estimate curvature
        vec2<T> v1 = vec2<T>(p1.x - p0.x, p1.y - p0.y);
        vec2<T> v2 = vec2<T>(p2.x - p1.x, p2.y - p1.y);

        T speed = euler::direct::norm(v1);
        if (speed < T(0.001)) {
            return max_step;
        }

        vec2<T> accel = v2 - v1;
        T curvature = euler::direct::norm(accel) / (speed * speed);

        if (curvature < T(0.001)) {
            return max_step;
        }

        T step = std::min(max_step, tolerance_ / (curvature * speed));
        return std::max(base_step, step);
    }

    void advance_to_next_pixel() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ == decltype(line_iter_)::end()) {
                using_line_ = false;
                t_ += dt_;
            } else {
                return;
            }
        }

        while (current_segment_ < static_cast<int>(points_.size()) - 1 && !this->done_) {
            while (t_ <= T(1)) {
                point2<T> current = evaluate_segment(current_segment_, t_);
                point2i current_pixel{static_cast<int>(std::round(current.x)),
                                     static_cast<int>(std::round(current.y))};

                if (current_pixel != last_pixel_) {
                    // Check if we need to fill gaps
                    int dx = std::abs(current_pixel.x - last_pixel_.x);
                    int dy = std::abs(current_pixel.y - last_pixel_.y);

                    if (dx > 1 || dy > 1) {
                        // Use line iterator to fill gap
                        line_iter_ = make_line_iterator(last_pixel_, current_pixel);
                        ++line_iter_; // Skip the first pixel
                        if (line_iter_ != decltype(line_iter_)::end()) {
                            using_line_ = true;
                            last_pixel_ = current_pixel;
                            return;
                        }
                    }

                    last_pixel_ = current_pixel;
                    return;
                }

                dt_ = estimate_step_size();
                t_ += dt_;
            }

            // Move to next segment
            current_segment_++;
            t_ = T(0);
            dt_ = T(0.01);
        }

        this->done_ = true;
    }

public:
    /**
     * @brief Construct Catmull-Rom spline from any point container
     * @tparam PointContainer Any container satisfying point2_container concept
     * @param points Container of points to interpolate
     * @param tension Tension parameter (0.5 for standard Catmull-Rom)
     * @param tolerance Maximum pixel deviation
     */
    template<point2_container PointContainer>
    catmull_rom_iterator(const PointContainer& points,
                        T tension = T(0.5),
                        T tolerance = default_tolerance<T>())
        : points_(std::begin(points), std::end(points)),
          tension_(tension),
          current_segment_(0),
          t_(0),
          dt_(T(0.01)),
          tolerance_(tolerance),
          using_line_(false),
          line_iter_{point2i{0,0}, point2i{0,0}} {

        if (points_.size() < 2) {
            this->done_ = true;
            return;
        }

        point2<T> start = points_[0];
        last_pixel_ = point2i{static_cast<int>(std::round(start.x)),
                             static_cast<int>(std::round(start.y))};
    }


    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return pixel<int>{last_pixel_};
    }

    catmull_rom_iterator& operator++() {
        advance_to_next_pixel();
        return *this;
    }

    catmull_rom_iterator operator++(int) {
        catmull_rom_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
};

/**
 * @brief Factory functions with C++20 concepts
 */
template<point2_container PointContainer>
auto make_bspline(const PointContainer& control_points,
                 int degree = 3,
                 typename PointContainer::value_type::value_type tolerance =
                     default_tolerance<typename PointContainer::value_type::value_type>()) {
    using T = typename PointContainer::value_type::value_type;
    return bspline_iterator<T>(control_points, degree, tolerance);
}

template<point2_container PointContainer, typename KnotContainer>
    requires std::ranges::sized_range<KnotContainer>
auto make_bspline(const PointContainer& control_points,
                 const KnotContainer& knots,
                 int degree = 3,
                 typename PointContainer::value_type::value_type tolerance =
                     default_tolerance<typename PointContainer::value_type::value_type>()) {
    using T = typename PointContainer::value_type::value_type;
    return bspline_iterator<T>(control_points, knots, degree, tolerance);
}

template<point2_container PointContainer>
auto make_catmull_rom(const PointContainer& points,
                     typename PointContainer::value_type::value_type tension =
                         typename PointContainer::value_type::value_type(0.5),
                     typename PointContainer::value_type::value_type tolerance =
                         default_tolerance<typename PointContainer::value_type::value_type>()) {
    using T = typename PointContainer::value_type::value_type;
    return catmull_rom_iterator<T>(points, tension, tolerance);
}

} // namespace euler::dda