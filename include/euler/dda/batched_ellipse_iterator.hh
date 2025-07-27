/**
 * @file batched_ellipse_iterator.hh
 * @brief Batched ellipse rasterization for improved performance
 * @author Euler Library Contributors
 * @date 2024
 * 
 * This file provides batched versions of ellipse rasterization iterators that
 * generate pixels in groups for improved cache utilization. The iterators support
 * both axis-aligned ellipses and elliptical arcs with angle constraints.
 * 
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/ellipse_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <array>
#include <type_traits>

namespace euler::dda {

/**
 * @brief Batched ellipse iterator using midpoint algorithm
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates ellipse pixels in batches for improved performance.
 * It uses the midpoint ellipse algorithm with 4-way symmetry to efficiently
 * generate all pixels on the ellipse perimeter.
 * 
 * Key features:
 * - Generates pixels in batches of up to 16 for cache efficiency
 * - Supports both full ellipses and elliptical arcs with angle constraints
 * - Handles axis-aligned ellipses only (no rotation support)
 * - Avoids duplicate pixels at axis endpoints through special case handling
 * - Uses two-region algorithm for accurate pixel selection
 * 
 * The algorithm divides the ellipse into two regions based on the gradient:
 * - Region 1: Where |dy/dx| < 1 (gradient magnitude less than 1)
 * - Region 2: Where |dy/dx| >= 1 (gradient magnitude greater than or equal to 1)
 * 
 * @note For rotated ellipses, apply a rotation transformation to the output pixels
 * 
 * Example usage:
 * @code
 * auto ellipse = make_batched_ellipse(center, semi_major, semi_minor);
 * while (!ellipse.at_end()) {
 *     const auto& batch = ellipse.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_pixel(batch.pixels[i].pos);
 *     }
 *     ellipse.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_ellipse_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = pixel<int>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // Ellipse parameters
    point2i center_;
    int a_, b_;  // Semi-major and semi-minor axes
    int a2_, b2_;  // a² and b²
    int fa2_, fb2_;  // 4a² and 4b²
    
    // Midpoint algorithm state
    int x_, y_;
    int64_t d1_, d2_;
    bool in_region2_;
    
    // Arc support
    bool is_arc_;
    T start_angle_, end_angle_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // 4-way symmetry points buffer
    std::array<pixel_type, 4> quadrants_;
    int quadrant_count_;
    int quadrant_index_;
    
    // Prefetch distance
    static constexpr int PREFETCH_DISTANCE = 16;
    
    bool is_angle_in_arc(int dx, int dy) const {
        if (!is_arc_) return true;
        
        // Account for ellipse stretching when computing angle
        T angle = atan2(static_cast<T>(dy) * static_cast<T>(a_), 
                       static_cast<T>(dx) * static_cast<T>(b_));
        if (angle < 0) angle += T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    void generate_quadrants() {
        quadrant_count_ = 0;
        quadrant_index_ = 0;
        
        auto add_if_valid = [this](int x, int y) {
            if (is_angle_in_arc(x - center_.x, y - center_.y)) {
                quadrants_[static_cast<size_t>(quadrant_count_++)].pos = {x, y};
            }
        };
        
        // Handle special cases to avoid duplicates
        if (x_ == 0 && y_ == 0) {
            // Center point only (degenerate ellipse)
            add_if_valid(center_.x, center_.y);
        } else if (x_ == 0) {
            // On vertical axis - only 2 points
            add_if_valid(center_.x, center_.y + y_);
            add_if_valid(center_.x, center_.y - y_);
        } else if (y_ == 0) {
            // On horizontal axis - only 2 points
            add_if_valid(center_.x + x_, center_.y);
            add_if_valid(center_.x - x_, center_.y);
        } else {
            // General case - 4-way symmetry
            add_if_valid(center_.x + x_, center_.y + y_);
            add_if_valid(center_.x - x_, center_.y + y_);
            add_if_valid(center_.x + x_, center_.y - y_);
            add_if_valid(center_.x - x_, center_.y - y_);
        }
    }
    
    void advance_to_next_position() {
        if (!in_region2_) {
            // Region 1: gradient > -1
            x_++;
            
            if (d1_ < 0) {
                d1_ += fb2_ * x_ + b2_;
            } else {
                y_--;
                d1_ += fb2_ * x_ - fa2_ * y_ + b2_;
            }
            
            // Check transition to region 2
            if (b2_ * x_ >= a2_ * y_) {
                in_region2_ = true;
                // Initial decision parameter for region 2
                d2_ = b2_ * (x_ + 1/2) * (x_ + 1/2) + 
                      a2_ * (y_ - 1) * (y_ - 1) - a2_ * b2_;
            }
        } else {
            // Region 2: gradient < -1
            y_--;
            
            if (d2_ > 0) {
                d2_ += a2_ - fa2_ * y_;
            } else {
                x_++;
                d2_ += fb2_ * x_ - fa2_ * y_ + a2_;
            }
        }
    }
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Fill batch with pixels
        while (!current_batch_.is_full() && !done_) {
            // Add remaining quadrant points from previous iteration
            while (quadrant_index_ < quadrant_count_ && !current_batch_.is_full()) {
                current_batch_.add(quadrants_[static_cast<size_t>(quadrant_index_++)]);
            }
            
            // If we've processed all quadrants, move to next position
            if (quadrant_index_ >= quadrant_count_) {
                advance_to_next_position();
                if (y_ < 0) {
                    done_ = true;
                    break;
                }
                generate_quadrants();
                if (quadrant_count_ == 0) {
                    done_ = true;
                    break;
                }
            }
        }
        
        // Prefetch memory for next batch if applicable
        if (!done_ && !current_batch_.is_empty()) {
            prefetch_next_batch();
        }
    }
    
    void prefetch_next_batch() {
        // Prefetch control points for future positions
        int future_x = x_ + PREFETCH_DISTANCE;
        if (future_x <= a_) {
            prefetch_hint::prefetch_for_read(&center_);
        }
    }
    
public:
    /**
     * @brief Construct iterator for a full ellipse
     * @param center Center point of the ellipse
     * @param semi_major Length of the semi-major axis (horizontal)
     * @param semi_minor Length of the semi-minor axis (vertical)
     * 
     * Creates an iterator that generates all pixels on the ellipse perimeter.
     * The ellipse is axis-aligned with the semi-major axis along the x-axis
     * and the semi-minor axis along the y-axis.
     * 
     * For floating-point coordinates, values are rounded to nearest integers.
     * 
     * @note If either axis length is <= 0, generates a single pixel at the center
     */
    batched_ellipse_iterator(point2<T> center, T semi_major, T semi_minor) 
        : is_arc_(false), done_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            a_ = semi_major;
            b_ = semi_minor;
        } else {
            center_ = round(center);
            a_ = static_cast<int>(round(semi_major));
            b_ = static_cast<int>(round(semi_minor));
        }
        
        // Handle degenerate cases
        if (a_ <= 0 || b_ <= 0) {
            quadrants_[0].pos = center_;
            quadrant_count_ = 1;
            quadrant_index_ = 0;
            done_ = false;
            fill_batch();
            return;
        }
        
        // Initialize for region 1
        x_ = 0;
        y_ = b_;
        a2_ = a_ * a_;
        b2_ = b_ * b_;
        fa2_ = 4 * a2_;
        fb2_ = 4 * b2_;
        
        // Initial decision parameter for region 1
        d1_ = b2_ - a2_ * b_ + a2_ / 4;
        in_region2_ = false;
        
        // Generate first set of quadrants
        generate_quadrants();
        quadrant_index_ = 0;
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Construct iterator for an elliptical arc
     * @tparam Angle Angle type (can be degrees, radians, or raw T)
     * @param center Center point of the ellipse
     * @param semi_major Length of the semi-major axis (horizontal)
     * @param semi_minor Length of the semi-minor axis (vertical)
     * @param start_angle Starting angle of the arc
     * @param end_angle Ending angle of the arc
     * 
     * Creates an iterator that generates only pixels within the specified
     * angular range. Angles are measured counter-clockwise from the positive
     * x-axis, with the angle adjusted for the ellipse's eccentricity.
     * 
     * The arc is drawn from start_angle to end_angle in the counter-clockwise
     * direction. If start_angle > end_angle, the arc wraps around through 0°.
     * 
     * @note The angle calculation accounts for ellipse stretching to ensure
     *       accurate arc endpoints
     */
    template<typename Angle>
    batched_ellipse_iterator(point2<T> center, T semi_major, T semi_minor,
                            const Angle& start_angle, const Angle& end_angle)
        : is_arc_(true), done_(false) {
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
        if constexpr (std::is_same_v<Angle, T>) {
            // Assume already in radians
            start_angle_ = start_angle;
            end_angle_ = end_angle;
        } else {
            // Convert from angle type to radians
            start_angle_ = static_cast<T>(to_radians(start_angle));
            end_angle_ = static_cast<T>(to_radians(end_angle));
        }
        
        // Normalize angles to [0, 2π)
        while (start_angle_ < 0) start_angle_ += T(2 * pi);
        while (start_angle_ >= T(2 * pi)) start_angle_ -= T(2 * pi);
        while (end_angle_ < 0) end_angle_ += T(2 * pi);
        while (end_angle_ >= T(2 * pi)) end_angle_ -= T(2 * pi);
        
        // Initialize for region 1
        x_ = 0;
        y_ = b_;
        a2_ = a_ * a_;
        b2_ = b_ * b_;
        fa2_ = 4 * a2_;
        fb2_ = 4 * b2_;
        
        // Initial decision parameter for region 1
        d1_ = b2_ - a2_ * b_ + a2_ / 4;
        in_region2_ = false;
        
        // Generate first set of quadrants
        generate_quadrants();
        quadrant_index_ = 0;
        
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
     * Fills the internal batch with the next set of pixels. If the iterator
     * has already generated all pixels, this method clears the current batch.
     */
    batched_ellipse_iterator& next_batch() {
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
 * @brief Batched filled ellipse iterator using horizontal spans
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates horizontal spans that fill an ellipse, producing
 * batches of spans for efficient rendering. Each span represents a horizontal
 * line segment within the ellipse.
 * 
 * The iterator uses the standard ellipse equation (x/a)² + (y/b)² = 1 to
 * calculate the horizontal extent at each y-coordinate, working from top
 * to bottom of the ellipse.
 * 
 * Example usage:
 * @code
 * auto filled = make_batched_filled_ellipse(center, semi_major, semi_minor);
 * while (!filled.at_end()) {
 *     const auto& batch = filled.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         const auto& span = batch.pixels[i];
 *         draw_horizontal_line(span.y, span.x_start, span.x_end);
 *     }
 *     filled.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_filled_ellipse_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using span_type = span;
    using batch_type = pixel_batch<span_type>;
    
private:
    point2i center_;
    int a_, b_;
    int y_;
    
    batch_type current_batch_;
    bool done_;
    
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Generate spans from top to bottom
        while (!current_batch_.is_full() && y_ >= -b_) {
            // Calculate x extent for this y
            double y_ratio = static_cast<double>(y_) / static_cast<double>(b_);
            double x_extent = a_ * sqrt(1.0 - y_ratio * y_ratio);
            int x = static_cast<int>(x_extent);
            
            span_type span;
            span.y = center_.y + y_;
            span.x_start = center_.x - x;
            span.x_end = center_.x + x;
            
            current_batch_.add(span);
            --y_;
        }
        
        if (y_ < -b_) {
            done_ = true;
        }
    }
    
public:
    /**
     * @brief Construct filled ellipse iterator
     */
    batched_filled_ellipse_iterator(point2<T> center, T semi_major, T semi_minor)
        : done_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            a_ = semi_major;
            b_ = semi_minor;
        } else {
            center_ = round(center);
            a_ = static_cast<int>(round(semi_major));
            b_ = static_cast<int>(round(semi_minor));
        }
        
        y_ = b_;
        fill_batch();
    }
    
    const batch_type& current_batch() const { return current_batch_; }
    bool at_end() const { return done_ && current_batch_.is_empty(); }
    
    batched_filled_ellipse_iterator& next_batch() {
        if (!done_) {
            fill_batch();
        } else {
            current_batch_.clear();
        }
        return *this;
    }
    
    /**
     * @brief Process all spans with a callback
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
 * @brief Create a batched iterator for a complete ellipse
 * @tparam T Coordinate type
 * @param center Center point of the ellipse
 * @param semi_major Length of the semi-major axis
 * @param semi_minor Length of the semi-minor axis
 * @return A batched_ellipse_iterator instance
 * 
 * Factory function that creates a batched ellipse iterator. This is the
 * preferred way to create ellipse iterators as it provides type deduction.
 * 
 * @see batched_ellipse_iterator
 */
template<typename T>
auto make_batched_ellipse(point2<T> center, T semi_major, T semi_minor) {
    return batched_ellipse_iterator<T>(center, semi_major, semi_minor);
}

/**
 * @brief Create a batched iterator for an elliptical arc
 * @tparam T Coordinate type
 * @tparam Angle Angle type (degrees, radians, etc.)
 * @param center Center point of the ellipse
 * @param semi_major Length of the semi-major axis
 * @param semi_minor Length of the semi-minor axis
 * @param start_angle Starting angle of the arc
 * @param end_angle Ending angle of the arc
 * @return A batched_ellipse_iterator instance configured for arc generation
 * 
 * Factory function that creates a batched elliptical arc iterator.
 * Angles are automatically converted to radians internally.
 * 
 * @see batched_ellipse_iterator
 */
template<typename T, typename Angle>
auto make_batched_ellipse_arc(point2<T> center, T semi_major, T semi_minor,
                             const Angle& start_angle, const Angle& end_angle) {
    return batched_ellipse_iterator<T>(center, semi_major, semi_minor, start_angle, end_angle);
}

/**
 * @brief Create a batched iterator for a filled ellipse
 * @tparam T Coordinate type
 * @param center Center point of the ellipse
 * @param semi_major Length of the semi-major axis
 * @param semi_minor Length of the semi-minor axis
 * @return A batched_filled_ellipse_iterator instance
 * 
 * Factory function that creates a batched filled ellipse iterator,
 * which generates horizontal spans instead of individual pixels.
 * 
 * @see batched_filled_ellipse_iterator
 */
template<typename T>
auto make_batched_filled_ellipse(point2<T> center, T semi_major, T semi_minor) {
    return batched_filled_ellipse_iterator<T>(center, semi_major, semi_minor);
}

} // namespace euler::dda