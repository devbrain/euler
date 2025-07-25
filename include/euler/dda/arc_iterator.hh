/**
 * @file arc_iterator.hh
 * @brief Arc and filled arc rasterization extensions
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/dda/ellipse_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/math/trigonometry.hh>
#include <euler/dda/dda_math.hh>
#include <algorithm>
#include <array>

namespace euler::dda {

/**
 * @brief Filled arc iterator for filled circle segments
 * @tparam T Coordinate type
 * 
 * Generates horizontal spans for a filled arc segment.
 */
template<typename T>
class filled_arc_iterator : public dda_iterator_base<filled_arc_iterator<T>, span, T> {
    using base = dda_iterator_base<filled_arc_iterator<T>, span, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int radius_;
    int y_;
    T start_angle_, end_angle_;
    
    bool is_angle_in_arc(T angle) const {
        // Normalize angle to [0, 2π]
        while (angle < 0) angle += T(2 * pi);
        while (angle >= T(2 * pi)) angle -= T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            // Arc crosses 0 degrees
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    std::pair<int, int> compute_x_range(int y_offset) const {
        // For a given y offset from center, compute the x range
        T y = static_cast<T>(y_offset);
        T y_squared = y * y;
        T radius_squared = static_cast<T>(radius_) * static_cast<T>(radius_);
        
        // Check if this y is outside the circle
        if (y_squared > radius_squared) {
            return {-1, -1};
        }
        
        // Compute the x offset for this y
        T x_offset = sqrt(radius_squared - y_squared);
        
        // Full circle endpoints
        int x_left = center_.x - static_cast<int>(x_offset);
        int x_right = center_.x + static_cast<int>(x_offset);
        
        // Now we need to clip to the arc range
        // Check angles at the circle boundaries for this y
        T angle_left = atan2(y, -x_offset);
        T angle_right = atan2(y, x_offset);
        
        // Normalize angles for comparison
        if (angle_left < 0) angle_left += T(2 * pi);
        if (angle_right < 0) angle_right += T(2 * pi);
        
        bool left_in_arc = is_angle_in_arc(angle_left);
        bool right_in_arc = is_angle_in_arc(angle_right);
        
        if (left_in_arc && right_in_arc) {
            // Both endpoints are in the arc, use full span
            return {x_left, x_right};
        }
        
        // We need to find where the arc boundaries intersect this horizontal line
        // The arc boundaries are at start_angle and end_angle
        
        // Compute the x coordinates for start and end angles at this y
        int x_at_start = -1, x_at_end = -1;
        bool start_at_this_y = false, end_at_this_y = false;
        
        // Check if start angle crosses this y
        T start_y = static_cast<T>(radius_) * sin(start_angle_);
        if (abs(start_y - y) < T(0.5)) {
            x_at_start = center_.x + static_cast<int>(round(static_cast<T>(radius_) * cos(start_angle_)));
            start_at_this_y = true;
        }
        
        // Check if end angle crosses this y
        T end_y = static_cast<T>(radius_) * sin(end_angle_);
        if (abs(end_y - y) < T(0.5)) {
            x_at_end = center_.x + static_cast<int>(round(static_cast<T>(radius_) * cos(end_angle_)));
            end_at_this_y = true;
        }
        
        if (!left_in_arc && !right_in_arc) {
            // Neither endpoint is in the arc
            // The span exists only if the arc boundaries cross this y
            if (!start_at_this_y && !end_at_this_y) {
                return {-1, -1};
            }
            
            // If both boundaries are at this y, span is between them
            if (start_at_this_y && end_at_this_y) {
                return {min(x_at_start, x_at_end), max(x_at_start, x_at_end)};
            }
            
            // If only one boundary is at this y, we need to check if the arc
            // wraps around and has another intersection
            if (start_at_this_y || end_at_this_y) {
                // For arcs that wrap around (start > end), check if this y
                // is included in the wrapped portion
                if (start_angle_ > end_angle_) {
                    // Arc wraps around 0 degrees
                    // Check if the y coordinate is in the wrapped region
                    T angle_at_y = asin(y / static_cast<T>(radius_));
                    T angle_at_y_alt = T(pi) - angle_at_y;
                    
                    // Normalize angles
                    if (angle_at_y < 0) angle_at_y += T(2 * pi);
                    if (angle_at_y_alt < 0) angle_at_y_alt += T(2 * pi);
                    
                    bool in_wrapped_region = (angle_at_y >= start_angle_ || angle_at_y <= end_angle_) ||
                                           (angle_at_y_alt >= start_angle_ || angle_at_y_alt <= end_angle_);
                    
                    if (in_wrapped_region) {
                        // Find the other x intersection
                        if (start_at_this_y) {
                            // Start angle is here, find where arc exits on the right
                            return {x_at_start, x_right};
                        } else {
                            // End angle is here, find where arc enters on the left
                            return {x_left, x_at_end};
                        }
                    }
                }
                return {-1, -1};
            }
        }
        
        // One endpoint is in the arc, the other is not
        int x_min = x_left;
        int x_max = x_right;
        
        if (!left_in_arc) {
            // Left endpoint is outside arc
            // Need to find where the arc intersects this horizontal line
            x_min = center_.x;  // Default to center
            
            // The arc could intersect at start_angle or end_angle
            if (start_at_this_y) {
                // Start angle boundary is at this y
                x_min = min(x_min, x_at_start);
            }
            if (end_at_this_y) {
                // End angle boundary is at this y  
                x_min = min(x_min, x_at_end);
            }
            
            // For arcs like 0-90°, the left boundary is often the vertical line at center
            // when neither angle boundary is at this y
            T angle_at_center_x = atan2(y, T(0));
            if (angle_at_center_x < 0) angle_at_center_x += T(2 * pi);
            if (is_angle_in_arc(angle_at_center_x)) {
                x_min = center_.x;
            }
        }
        
        if (!right_in_arc) {
            // Right endpoint is outside arc
            x_max = center_.x;  // Default to center
            
            if (start_at_this_y) {
                x_max = max(x_max, x_at_start);
            }
            if (end_at_this_y) {
                x_max = max(x_max, x_at_end);
            }
            
            // Check if the vertical line at center is in the arc
            T angle_at_center_x = atan2(y, T(0));
            if (angle_at_center_x < 0) angle_at_center_x += T(2 * pi);
            if (is_angle_in_arc(angle_at_center_x)) {
                x_max = center_.x;
            }
        }
        
        if (x_min > x_max) {
            return {-1, -1};
        }
        
        return {x_min, x_max};
    }
    
public:
    /**
     * @brief Construct filled arc iterator
     */
    template<typename Angle>
    filled_arc_iterator(point2<T> center, T radius, 
                       const Angle& start_angle, const Angle& end_angle) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
        }
        
        // Convert angles to radians
        start_angle_ = static_cast<T>(to_radians(start_angle));
        end_angle_ = static_cast<T>(to_radians(end_angle));
        
        // Normalize angles to [0, 2π]
        while (start_angle_ < 0) start_angle_ += T(2 * pi);
        while (end_angle_ < 0) end_angle_ += T(2 * pi);
        while (start_angle_ >= T(2 * pi)) start_angle_ -= T(2 * pi);
        while (end_angle_ >= T(2 * pi)) end_angle_ -= T(2 * pi);
        
        y_ = -radius_;
        
        if (radius_ < 0) {
            this->done_ = true;
        }
        
        // Skip to first valid span
        while (y_ <= radius_) {
            auto [x_min, x_max] = compute_x_range(y_);
            if (x_min != -1 && x_max != -1 && x_min <= x_max) break;
            y_++;
        }
        
        if (y_ > radius_) {
            this->done_ = true;
        }
    }
    
    /**
     * @brief Get current span
     */
    value_type operator*() const {
        auto [x_min, x_max] = compute_x_range(y_);
        // This should never return invalid spans as the constructor and operator++ 
        // should skip them, but let's be safe
        if (x_min == -1 && x_max == -1) {
            // Return a degenerate span at center
            return {center_.y + y_, center_.x, center_.x};
        }
        return {center_.y + y_, x_min, x_max};
    }
    
    /**
     * @brief Advance to next span
     */
    filled_arc_iterator& operator++() {
        do {
            y_++;
            if (y_ > radius_) {
                this->done_ = true;
                return *this;
            }
            
            auto [x_min, x_max] = compute_x_range(y_);
            if (x_min != -1 && x_max != -1 && x_min <= x_max) break;
        } while (y_ <= radius_);
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    filled_arc_iterator operator++(int) {
        filled_arc_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Antialiased circle iterator
 * @tparam T Coordinate type (must be floating point)
 * 
 * Uses Wu's algorithm for antialiased circle rendering.
 */
template<typename T>
class aa_circle_iterator : public dda_iterator_base<aa_circle_iterator<T>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>, 
                  "Antialiased circles require floating point coordinates");
    
    using base = dda_iterator_base<aa_circle_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2<T> center_;
    T radius_;
    
    // For arc support
    bool is_arc_;
    T start_angle_, end_angle_;
    
    // Current position
    T x_, y_;
    
    // Pixel buffer for current position
    std::array<value_type, 16> pixels_;
    int pixel_count_;
    int pixel_index_;
    
    bool is_angle_in_arc(T dx, T dy) const {
        if (!is_arc_) return true;
        
        T angle = atan2(dy, dx);
        if (angle < 0) angle += T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    void add_pixel(T x, T y, T coverage) {
        if (pixel_count_ < static_cast<int>(pixels_.size()) && 
            is_angle_in_arc(x - center_.x, y - center_.y)) {
            auto& p = pixels_[static_cast<size_t>(pixel_count_++)];
            p.pos = point2<T>{x, y};
            p.coverage = coverage;
            p.distance = 0; // Not used for circles
        }
    }
    
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (x_ > y_) {
            this->done_ = true;
            return;
        }
        
        // Wu's algorithm: compute exact distance and coverage
        T exact_y = sqrt(radius_ * radius_ - x_ * x_);
        T floor_y = floor(exact_y);
        T frac_y = exact_y - floor_y;
        
        // Add pixels with appropriate coverage
        // Inner pixel
        add_pixel(center_.x + x_, center_.y + floor_y, 1.0f - frac_y);
        add_pixel(center_.x + x_, center_.y - floor_y, 1.0f - frac_y);
        add_pixel(center_.x - x_, center_.y + floor_y, 1.0f - frac_y);
        add_pixel(center_.x - x_, center_.y - floor_y, 1.0f - frac_y);
        
        // Outer pixel
        if (frac_y > 0.01f) {
            add_pixel(center_.x + x_, center_.y + floor_y + 1, frac_y);
            add_pixel(center_.x + x_, center_.y - floor_y - 1, frac_y);
            add_pixel(center_.x - x_, center_.y + floor_y + 1, frac_y);
            add_pixel(center_.x - x_, center_.y - floor_y - 1, frac_y);
        }
        
        // Also do y-x symmetry
        if (x_ != floor_y) {
            add_pixel(center_.x + floor_y, center_.y + x_, 1.0f - frac_y);
            add_pixel(center_.x + floor_y, center_.y - x_, 1.0f - frac_y);
            add_pixel(center_.x - floor_y, center_.y + x_, 1.0f - frac_y);
            add_pixel(center_.x - floor_y, center_.y - x_, 1.0f - frac_y);
            
            if (frac_y > 0.01f) {
                add_pixel(center_.x + floor_y + 1, center_.y + x_, frac_y);
                add_pixel(center_.x + floor_y + 1, center_.y - x_, frac_y);
                add_pixel(center_.x - floor_y - 1, center_.y + x_, frac_y);
                add_pixel(center_.x - floor_y - 1, center_.y - x_, frac_y);
            }
        }
        
        if (pixel_count_ == 0) {
            // No pixels in arc, advance
            x_ += 1;
            generate_pixels();
        }
    }
    
public:
    /**
     * @brief Construct antialiased circle iterator
     */
    aa_circle_iterator(point2<T> center, T radius)
        : center_(center), radius_(radius), is_arc_(false), x_(0) {
        y_ = radius_;
        generate_pixels();
    }
    
    /**
     * @brief Construct antialiased arc iterator
     */
    template<typename Angle>
    aa_circle_iterator(point2<T> center, T radius,
                      const Angle& start_angle, const Angle& end_angle)
        : center_(center), radius_(radius), is_arc_(true), x_(0) {
        
        // Convert angles to radians
        start_angle_ = static_cast<T>(to_radians(start_angle));
        end_angle_ = static_cast<T>(to_radians(end_angle));
        
        // Normalize angles to [0, 2π]
        while (start_angle_ < 0) start_angle_ += T(2 * pi);
        while (end_angle_ < 0) end_angle_ += T(2 * pi);
        while (start_angle_ >= T(2 * pi)) start_angle_ -= T(2 * pi);
        while (end_angle_ >= T(2 * pi)) end_angle_ -= T(2 * pi);
        
        y_ = radius_;
        generate_pixels();
    }
    
    value_type operator*() const {
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    aa_circle_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            x_ += 1;
            generate_pixels();
        }
        return *this;
    }
    
    aa_circle_iterator operator++(int) {
        aa_circle_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Filled ellipse arc iterator
 * @tparam T Coordinate type
 */
template<typename T>
class filled_ellipse_arc_iterator : public dda_iterator_base<filled_ellipse_arc_iterator<T>, span, T> {
    using base = dda_iterator_base<filled_ellipse_arc_iterator<T>, span, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int a_, b_;  // Semi-major and semi-minor axes
    int y_;
    T start_angle_, end_angle_;
    
    bool is_angle_in_arc(T angle) const {
        // Normalize angle to [0, 2π]
        while (angle < 0) angle += T(2 * pi);
        while (angle >= T(2 * pi)) angle -= T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    std::pair<int, int> compute_x_range(int y_offset) const {
        // For ellipse: x²/a² + y²/b² = 1
        // So x = ±a * sqrt(1 - y²/b²)
        T y_normalized = static_cast<T>(y_offset) / static_cast<T>(b_);
        if (abs(y_normalized) > 1) return {-1, -1};
        
        T x_offset = static_cast<T>(a_) * sqrt(1 - y_normalized * y_normalized);
        int x_left = center_.x - static_cast<int>(x_offset);
        int x_right = center_.x + static_cast<int>(x_offset);
        
        // Check angles at the endpoints
        // For ellipse, we need to use the parametric angle, not the geometric angle
        T angle_left = atan2(static_cast<T>(y_offset) / static_cast<T>(b_), -x_offset / static_cast<T>(a_));
        T angle_right = atan2(static_cast<T>(y_offset) / static_cast<T>(b_), x_offset / static_cast<T>(a_));
        
        // Normalize angles
        if (angle_left < 0) angle_left += T(2 * pi);
        if (angle_right < 0) angle_right += T(2 * pi);
        
        bool left_in_arc = is_angle_in_arc(angle_left);
        bool right_in_arc = is_angle_in_arc(angle_right);
        
        if (left_in_arc && right_in_arc) {
            return {x_left, x_right};
        }
        
        // Compute where arc boundaries intersect this y
        int x_at_start = -1, x_at_end = -1;
        bool start_at_this_y = false, end_at_this_y = false;
        
        // Check if start angle crosses this y
        T start_y = static_cast<T>(b_) * sin(start_angle_);
        if (abs(start_y - static_cast<T>(y_offset)) < T(0.5)) {
            x_at_start = center_.x + static_cast<int>(round(static_cast<T>(a_) * cos(start_angle_)));
            start_at_this_y = true;
        }
        
        // Check if end angle crosses this y
        T end_y = static_cast<T>(b_) * sin(end_angle_);
        if (abs(end_y - static_cast<T>(y_offset)) < T(0.5)) {
            x_at_end = center_.x + static_cast<int>(round(static_cast<T>(a_) * cos(end_angle_)));
            end_at_this_y = true;
        }
        
        if (!left_in_arc && !right_in_arc) {
            if (!start_at_this_y && !end_at_this_y) {
                return {-1, -1};
            }
            
            if (start_at_this_y && end_at_this_y) {
                return {min(x_at_start, x_at_end), max(x_at_start, x_at_end)};
            }
            
            // Handle wrapped arcs
            if (start_at_this_y || end_at_this_y) {
                if (start_angle_ > end_angle_) {
                    // Arc wraps around, check if this y is in wrapped region
                    T param_angle = asin(y_normalized);
                    T param_angle_alt = T(pi) - param_angle;
                    
                    if (param_angle < 0) param_angle += T(2 * pi);
                    if (param_angle_alt < 0) param_angle_alt += T(2 * pi);
                    
                    bool in_wrapped = (param_angle >= start_angle_ || param_angle <= end_angle_) ||
                                    (param_angle_alt >= start_angle_ || param_angle_alt <= end_angle_);
                    
                    if (in_wrapped) {
                        if (start_at_this_y) {
                            return {x_at_start, x_right};
                        } else {
                            return {x_left, x_at_end};
                        }
                    }
                }
                return {-1, -1};
            }
        }
        
        // One endpoint in arc, other not
        int x_min = x_left;
        int x_max = x_right;
        
        if (!left_in_arc) {
            // Left endpoint is outside arc
            x_min = center_.x;  // Default to center
            
            if (start_at_this_y) {
                x_min = min(x_min, x_at_start);
            }
            if (end_at_this_y) {
                x_min = min(x_min, x_at_end);
            }
            
            // Check if vertical line at center is in arc
            T param_angle_at_center = atan2(y_normalized, T(0));
            if (param_angle_at_center < 0) param_angle_at_center += T(2 * pi);
            if (is_angle_in_arc(param_angle_at_center)) {
                x_min = center_.x;
            }
        }
        
        if (!right_in_arc) {
            // Right endpoint is outside arc
            x_max = center_.x;  // Default to center
            
            if (start_at_this_y) {
                x_max = max(x_max, x_at_start);
            }
            if (end_at_this_y) {
                x_max = max(x_max, x_at_end);
            }
            
            // Check if vertical line at center is in arc
            T param_angle_at_center = atan2(y_normalized, T(0));
            if (param_angle_at_center < 0) param_angle_at_center += T(2 * pi);
            if (is_angle_in_arc(param_angle_at_center)) {
                x_max = center_.x;
            }
        }
        
        if (x_min > x_max) {
            return {-1, -1};
        }
        
        return {x_min, x_max};
    }
    
public:
    template<typename Angle>
    filled_ellipse_arc_iterator(point2<T> center, T semi_major, T semi_minor,
                               const Angle& start_angle, const Angle& end_angle) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            a_ = semi_major;
            b_ = semi_minor;
        } else {
            center_ = round(center);
            a_ = static_cast<int>(round(semi_major));
            b_ = static_cast<int>(round(semi_minor));
        }
        
        // Convert angles to radians
        start_angle_ = static_cast<T>(to_radians(start_angle));
        end_angle_ = static_cast<T>(to_radians(end_angle));
        
        // Normalize angles
        while (start_angle_ < 0) start_angle_ += T(2 * pi);
        while (end_angle_ < 0) end_angle_ += T(2 * pi);
        while (start_angle_ >= T(2 * pi)) start_angle_ -= T(2 * pi);
        while (end_angle_ >= T(2 * pi)) end_angle_ -= T(2 * pi);
        
        y_ = -b_;
        
        if (a_ <= 0 || b_ <= 0) {
            this->done_ = true;
        }
        
        // Skip to first valid span
        while (y_ <= b_) {
            auto [x_min, x_max] = compute_x_range(y_);
            if (x_min != -1 && x_max != -1 && x_min <= x_max) break;
            y_++;
        }
        
        if (y_ > b_) {
            this->done_ = true;
        }
    }
    
    value_type operator*() const {
        auto [x_min, x_max] = compute_x_range(y_);
        // This should never return invalid spans as the constructor and operator++ 
        // should skip them, but let's be safe
        if (x_min == -1 && x_max == -1) {
            // Return a degenerate span at center
            return {center_.y + y_, center_.x, center_.x};
        }
        return {center_.y + y_, x_min, x_max};
    }
    
    filled_ellipse_arc_iterator& operator++() {
        do {
            y_++;
            if (y_ > b_) {
                this->done_ = true;
                return *this;
            }
            
            auto [x_min, x_max] = compute_x_range(y_);
            if (x_min != -1 && x_max != -1 && x_min <= x_max) break;
        } while (y_ <= b_);
        
        return *this;
    }
    
    filled_ellipse_arc_iterator operator++(int) {
        filled_ellipse_arc_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Antialiased ellipse arc iterator
 * @tparam T Coordinate type (must be floating point)
 */
template<typename T>
class aa_ellipse_arc_iterator : public dda_iterator_base<aa_ellipse_arc_iterator<T>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>,
                  "Antialiased ellipses require floating point coordinates");
    
    using base = dda_iterator_base<aa_ellipse_arc_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2<T> center_;
    T a_, b_;  // Semi-major and semi-minor axes
    T start_angle_, end_angle_;
    
    // Current angle for parametric generation
    T current_angle_;
    T angle_step_;
    
    // Pixel buffer
    std::array<value_type, 4> pixels_;
    int pixel_count_;
    int pixel_index_;
    
    bool is_angle_in_arc(T angle) const {
        // Normalize angle to [0, 2π]
        while (angle < 0) angle += T(2 * pi);
        while (angle >= T(2 * pi)) angle -= T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (!is_angle_in_arc(current_angle_)) {
            current_angle_ += angle_step_;
            if (current_angle_ > end_angle_ + T(2 * pi)) {
                this->done_ = true;
                return;
            }
            generate_pixels();
            return;
        }
        
        // Parametric ellipse: x = a*cos(t), y = b*sin(t)
        T x = center_.x + a_ * cos(current_angle_);
        T y = center_.y + b_ * sin(current_angle_);
        
        // Get integer coordinates and fractional parts
        T floor_x = floor(x);
        T floor_y = floor(y);
        T frac_x = x - floor_x;
        T frac_y = y - floor_y;
        
        // Add antialiased pixels using coverage based on fractional parts
        auto& p0 = pixels_[static_cast<size_t>(pixel_count_++)];
        p0.pos = point2<T>{floor_x, floor_y};
        p0.coverage = (1 - frac_x) * (1 - frac_y);
        
        if (frac_x > 0.01f) {
            auto& p1 = pixels_[static_cast<size_t>(pixel_count_++)];
            p1.pos = point2<T>{floor_x + 1, floor_y};
            p1.coverage = frac_x * (1 - frac_y);
        }
        
        if (frac_y > 0.01f) {
            auto& p2 = pixels_[static_cast<size_t>(pixel_count_++)];
            p2.pos = point2<T>{floor_x, floor_y + 1};
            p2.coverage = (1 - frac_x) * frac_y;
        }
        
        if (frac_x > 0.01f && frac_y > 0.01f) {
            auto& p3 = pixels_[static_cast<size_t>(pixel_count_++)];
            p3.pos = point2<T>{floor_x + 1, floor_y + 1};
            p3.coverage = frac_x * frac_y;
        }
    }
    
public:
    template<typename Angle>
    aa_ellipse_arc_iterator(point2<T> center, T semi_major, T semi_minor,
                           const Angle& start_angle, const Angle& end_angle)
        : center_(center), a_(semi_major), b_(semi_minor) {
        
        // Convert angles to radians
        start_angle_ = static_cast<T>(to_radians(start_angle));
        end_angle_ = static_cast<T>(to_radians(end_angle));
        
        // Normalize angles
        while (start_angle_ < 0) start_angle_ += T(2 * pi);
        while (end_angle_ < 0) end_angle_ += T(2 * pi);
        while (start_angle_ >= T(2 * pi)) start_angle_ -= T(2 * pi);
        while (end_angle_ >= T(2 * pi)) end_angle_ -= T(2 * pi);
        
        current_angle_ = start_angle_;
        
        // Compute angle step based on curvature
        T max_radius = max(a_, b_);
        angle_step_ = T(1) / max_radius;
        
        generate_pixels();
    }
    
    value_type operator*() const {
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    aa_ellipse_arc_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            current_angle_ += angle_step_;
            generate_pixels();
        }
        return *this;
    }
    
    aa_ellipse_arc_iterator operator++(int) {
        aa_ellipse_arc_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper functions for arc iterators
 */
template<typename T, typename Angle>
auto make_filled_arc_iterator(point2<T> center, T radius,
                             const Angle& start, const Angle& end) {
    return filled_arc_iterator<T>(center, radius, start, end);
}

template<typename T>
auto make_aa_circle_iterator(point2<T> center, T radius) {
    return aa_circle_iterator<T>(center, radius);
}

template<typename T, typename Angle>
auto make_aa_arc_iterator(point2<T> center, T radius,
                         const Angle& start, const Angle& end) {
    return aa_circle_iterator<T>(center, radius, start, end);
}

template<typename T, typename Angle>
auto make_filled_ellipse_arc_iterator(point2<T> center, T semi_major, T semi_minor,
                                     const Angle& start, const Angle& end) {
    return filled_ellipse_arc_iterator<T>(center, semi_major, semi_minor, start, end);
}

template<typename T, typename Angle>
auto make_aa_ellipse_arc_iterator(point2<T> center, T semi_major, T semi_minor,
                                 const Angle& start, const Angle& end) {
    return aa_ellipse_arc_iterator<T>(center, semi_major, semi_minor, start, end);
}

} // namespace euler::dda