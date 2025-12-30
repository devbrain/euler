/**
 * @file thick_line_iterator.hh
 * @brief Thick line rasterization with variable width
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/error.hh>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <functional>
#include <limits>
#include <utility>
#include <euler/math/basic.hh>

namespace euler::dda {

// Import necessary types and functions
using std::numeric_limits;
using std::vector;
using std::unordered_set;
using std::pair;
using std::hash;
using std::size_t;
using std::is_floating_point_v;
using euler::clamp;
using euler::vec2;

// Hash function for point2i
template<typename T>
struct point2i_hash {
    size_t operator()(const point2<T>& p) const {
        return hash<T>{}(p.x) ^ (hash<T>{}(p.y) << 1);
    }
};

// Forward declaration
template<typename T> class filled_circle_iterator;

/**
 * @brief Range wrapper for filled circle used in thick lines
 */
template<typename T>
class filled_circle_range {
    point2<T> center_;
    T radius_;
    
public:
    filled_circle_range(point2<T> center, T radius)
        : center_(center), radius_(radius) {}
    
    auto begin() const { return filled_circle_iterator<T>(center_, radius_); }
    auto end() const { return dda_sentinel{}; }
};

/**
 * @brief Thick line iterator using brush-based approach
 * @tparam T Coordinate type
 * 
 * Generates pixels for lines with specified thickness.
 * Uses circular brush swept along the line path.
 */
template<typename T>
class thick_line_iterator : public dda_iterator_base<thick_line_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<thick_line_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    line_iterator<T> line_iter_;
    T thickness_;
    int radius_;
    
    // Current state
    vector<value_type> current_pixels_;
    size_t pixel_index_;
    unordered_set<point2i, point2i_hash<int>> emitted_pixels_;  // To avoid duplicates
    
    void generate_brush_pixels() {
        current_pixels_.clear();
        pixel_index_ = 0;
        
        if (line_iter_ == line_iterator<T>::end()) {
            this->done_ = true;
            return;
        }
        
        auto center = (*line_iter_).pos;
        
        // Use filled circle iterator for the brush
        for (auto span : filled_circle_range<int>(center, radius_)) {
            for (int x = span.x_start; x <= span.x_end; ++x) {
                point2i p{x, span.y};
                if (emitted_pixels_.find(p) == emitted_pixels_.end()) {
                    current_pixels_.push_back({p});
                    emitted_pixels_.insert(p);
                }
            }
        }
        
        ++line_iter_;
    }
    
public:
    /**
     * @brief Construct thick line iterator
     * @param start Starting point
     * @param end Ending point
     * @param thickness Line thickness in pixels
     */
    thick_line_iterator(point2<T> start, point2<T> end, T thickness)
        : line_iter_(start, end), thickness_(thickness) {
        
        radius_ = static_cast<int>(round(thickness / T(2)));
        if (radius_ < 0) radius_ = 0;
        
        generate_brush_pixels();
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        EULER_CHECK_INDEX(pixel_index_, current_pixels_.size());
        return current_pixels_[pixel_index_];
    }
    
    /**
     * @brief Advance to next pixel
     */
    thick_line_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= current_pixels_.size()) {
            generate_brush_pixels();
        }
        return *this;
    }
};

/**
 * @brief Antialiased thick line iterator
 * @tparam T Coordinate type (float or double)
 * 
 * Generates antialiased pixels for thick lines.
 * Uses distance field approach for smooth edges.
 */
template<typename T>
class aa_thick_line_iterator : public dda_iterator_base<aa_thick_line_iterator<T>, aa_pixel<T>, T> {
    static_assert(is_floating_point_v<T>, 
                  "Antialiased thick lines require floating point coordinates");
    
    using base = dda_iterator_base<aa_thick_line_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2<T> start_, end_;
    T half_thickness_;
    vec2<T> line_vec_;
    vec2<T> normal_;
    T line_length_;
    
    // Bounding box for the thick line
    int x_min_, x_max_, y_min_, y_max_;
    int current_x_, current_y_;
    
    T distance_to_segment(point2<T> p) const {
        // Handle zero-length line (degenerate case)
        if (line_length_ <= T(0)) {
            return distance(p, start_);
        }
        // Project point onto line segment
        auto v = p - start_;
        T t = clamp(dot(v, line_vec_) / (line_length_ * line_length_), T(0), T(1));
        auto projection = point2<T>{start_.x + t * line_vec_.x(), start_.y + t * line_vec_.y()};
        return distance(p, projection);
    }
    
public:
    /**
     * @brief Construct antialiased thick line iterator
     */
    aa_thick_line_iterator(point2<T> start, point2<T> end, T thickness)
        : start_(start), end_(end), half_thickness_(thickness / T(2)) {
        
        line_vec_ = end - start;
        line_length_ = length(line_vec_);
        
        if (line_length_ > T(0)) {
            // Compute perpendicular normal
            normal_ = normalize(perp(line_vec_));
            
            // Compute bounding box
            auto expand = abs(vec2<T>{normal_.x() * half_thickness_, normal_.y() * half_thickness_}) + vec2<T>{T(1), T(1)};
            auto min_pt = point2<T>{min(start_.x, end_.x) - expand[0], min(start_.y, end_.y) - expand[1]};
            auto max_pt = point2<T>{max(start_.x, end_.x) + expand[0], max(start_.y, end_.y) + expand[1]};
            
            x_min_ = static_cast<int>(floor(min_pt.x));
            y_min_ = static_cast<int>(floor(min_pt.y));
            x_max_ = static_cast<int>(ceil(max_pt.x));
            y_max_ = static_cast<int>(ceil(max_pt.y));
            
            current_x_ = x_min_;
            current_y_ = y_min_;
        } else {
            // Degenerate line - just a circle
            x_min_ = static_cast<int>(floor(start.x - half_thickness_ - 1));
            y_min_ = static_cast<int>(floor(start.y - half_thickness_ - 1));
            x_max_ = static_cast<int>(ceil(start.x + half_thickness_ + 1));
            y_max_ = static_cast<int>(ceil(start.y + half_thickness_ + 1));
            
            current_x_ = x_min_;
            current_y_ = y_min_;
        }
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        value_type pixel;
        pixel.pos = {static_cast<T>(current_x_), static_cast<T>(current_y_)};
        
        // Sample at pixel center
        point2<T> p{static_cast<T>(current_x_) + T(0.5), static_cast<T>(current_y_) + T(0.5)};
        T d = distance_to_segment(p);
        
        // Compute coverage based on distance
        pixel.distance = static_cast<float>(d - half_thickness_);
        pixel.coverage = static_cast<float>(clamp(T(1) - (d - half_thickness_), T(0), T(1)));
        
        return pixel;
    }
    
    /**
     * @brief Advance to next pixel
     */
    aa_thick_line_iterator& operator++() {
        // Scan through bounding box, skipping pixels with zero coverage
        do {
            current_x_++;
            if (current_x_ > x_max_) {
                current_x_ = x_min_;
                current_y_++;
                if (current_y_ > y_max_) {
                    this->done_ = true;
                    return *this;
                }
            }
            
            // Check if this pixel has non-zero coverage
            point2<T> p{static_cast<T>(current_x_) + T(0.5), static_cast<T>(current_y_) + T(0.5)};
            T d = distance_to_segment(p);
            if (d <= half_thickness_ + T(1)) {
                break;  // Found a pixel with coverage
            }
        } while (current_y_ <= y_max_);
        
        return *this;
    }
};

/**
 * @brief Scanline-based thick line iterator
 * @tparam T Coordinate type
 * 
 * Generates horizontal spans for filled thick lines.
 * More efficient for very thick lines.
 */
template<typename T>
class thick_line_span_iterator : public dda_iterator_base<thick_line_span_iterator<T>, span, T> {
    using base = dda_iterator_base<thick_line_span_iterator<T>, span, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2<T> start_, end_;
    T half_thickness_;
    vec2<T> line_vec_;
    vec2<T> normal_;
    T line_length_;
    
    int y_min_, y_max_;
    int current_y_;
    
    pair<int, int> compute_span_at_y(int y) const {
        T y_center = static_cast<T>(y) + T(0.5);

        // Find intersection of scanline with thick line
        // The thick line is the set of points within half_thickness_ of the line segment

        int x_min = numeric_limits<int>::max();
        int x_max = numeric_limits<int>::min();

        // Check intersection with line capsule
        if (line_length_ > T(0)) {
            T line_vec_y_sq = line_vec_.y() * line_vec_.y();

            // Handle horizontal lines (line_vec_.y() == 0) separately
            if (line_vec_y_sq > T(0)) {
                // Project scanline onto line direction
                T t1 = ((y_center - start_.y) * line_vec_.y() -
                        half_thickness_ * abs(normal_.y())) / line_vec_y_sq;
                T t2 = ((y_center - start_.y) * line_vec_.y() +
                        half_thickness_ * abs(normal_.y())) / line_vec_y_sq;

                t1 = clamp(t1, T(0), T(1));
                t2 = clamp(t2, T(0), T(1));

                if (t1 <= t2) {
                    auto v1 = vec2<T>(t1 * line_vec_[0], t1 * line_vec_[1]);
                    auto v2 = vec2<T>(t2 * line_vec_[0], t2 * line_vec_[1]);
                    auto p1 = start_ + v1;
                    auto p2 = start_ + v2;

                    // Expand by perpendicular thickness
                    T perp_dist = half_thickness_ * abs(normal_.x());
                    x_min = static_cast<int>(floor(min(p1.x, p2.x) - perp_dist));
                    x_max = static_cast<int>(ceil(max(p1.x, p2.x) + perp_dist));
                }
            } else {
                // Horizontal line: check if scanline is within thickness of the line
                T dy = abs(y_center - start_.y);
                if (dy <= half_thickness_) {
                    // The entire line segment is at this y level (within thickness)
                    x_min = static_cast<int>(floor(min(start_.x, end_.x) - half_thickness_));
                    x_max = static_cast<int>(ceil(max(start_.x, end_.x) + half_thickness_));
                }
            }
        }

        // Check circles at endpoints
        auto check_circle = [&](point2<T> center) {
            T dy = y_center - center.y;
            if (abs(dy) <= half_thickness_) {
                T dx = sqrt(half_thickness_ * half_thickness_ - dy * dy);
                x_min = min(x_min, static_cast<int>(floor(center.x - dx)));
                x_max = max(x_max, static_cast<int>(ceil(center.x + dx)));
            }
        };

        check_circle(start_);
        check_circle(end_);

        return {x_min, x_max};
    }
    
public:
    /**
     * @brief Construct thick line span iterator
     */
    thick_line_span_iterator(point2<T> start, point2<T> end, T thickness)
        : start_(start), end_(end), half_thickness_(thickness / T(2)) {

        line_vec_ = end - start;
        line_length_ = length(line_vec_);

        if (line_length_ > T(0)) {
            normal_ = normalize(perp(line_vec_));
        }

        // Compute vertical bounds
        y_min_ = static_cast<int>(floor(min(start.y, end.y) - half_thickness_));
        y_max_ = static_cast<int>(ceil(max(start.y, end.y) + half_thickness_));
        current_y_ = y_min_;

        // Skip to first valid span
        skip_empty_spans();
    }

    /**
     * @brief Get current span
     */
    value_type operator*() const {
        auto [x_min, x_max] = compute_span_at_y(current_y_);
        return {current_y_, x_min, x_max};
    }

    /**
     * @brief Advance to next span
     */
    thick_line_span_iterator& operator++() {
        current_y_++;
        skip_empty_spans();
        return *this;
    }

private:
    void skip_empty_spans() {
        while (current_y_ <= y_max_) {
            auto [x_min, x_max] = compute_span_at_y(current_y_);
            if (x_min <= x_max) {
                return;  // Found a valid span
            }
            current_y_++;
        }
        this->done_ = true;
    }

};

/**
 * @brief Helper to create thick line iterator
 */
template<typename T>
auto make_thick_line_iterator(point2<T> start, point2<T> end, T thickness) {
    return thick_line_iterator<T>(start, end, thickness);
}

/**
 * @brief Helper to create antialiased thick line iterator
 */
template<typename T>
auto make_aa_thick_line_iterator(point2<T> start, point2<T> end, T thickness) {
    return aa_thick_line_iterator<T>(start, end, thickness);
}

/**
 * @brief Helper to create thick line span iterator
 */
template<typename T>
auto make_thick_line_spans(point2<T> start, point2<T> end, T thickness) {
    return thick_line_span_iterator<T>(start, end, thickness);
}

} // namespace euler::dda