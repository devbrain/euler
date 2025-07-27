/**
 * @file batched_circle_iterator.hh
 * @brief Batched circle rasterization for improved performance
 * @author Euler Library Contributors
 * @date 2024
 * 
 * This file provides batched versions of circle rasterization iterators that
 * generate pixels in groups for improved cache utilization and reduced function
 * call overhead. The batched iterators maintain the same algorithmic properties
 * as their single-pixel counterparts while providing significant performance benefits.
 * 
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <array>
#include <type_traits>

namespace euler::dda {

/**
 * @brief Batched circle iterator using midpoint algorithm
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates circle pixels in batches for improved performance compared
 * to single-pixel iteration. It uses the midpoint circle algorithm with 8-way symmetry
 * to efficiently generate all pixels on the circle perimeter.
 * 
 * Key features:
 * - Generates pixels in batches of up to 16 for cache efficiency
 * - Supports both full circles and arcs with angle constraints
 * - Handles degenerate cases (radius 0) correctly
 * - Avoids duplicate pixels at axis endpoints
 * - Compatible with prefetching for improved memory access patterns
 * 
 * @note The iterator produces pixels in a non-sequential order due to symmetry
 *       exploitation. Use a sorting pass if sequential ordering is required.
 * 
 * Example usage:
 * @code
 * auto circle = make_batched_circle(point2f{100, 100}, 50.0f);
 * while (!circle.at_end()) {
 *     const auto& batch = circle.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_pixel(batch.pixels[i].pos);
 *     }
 *     circle.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_circle_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = pixel<int>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // Circle parameters
    point2i center_;
    int radius_;
    
    // Midpoint algorithm state
    int x_, y_;
    int d_;  // Decision parameter
    
    // Arc support
    bool is_arc_;
    T start_angle_, end_angle_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // 8-way symmetry points buffer
    std::array<pixel_type, 8> octants_;
    int octant_count_;
    int octant_index_;
    
    // Prefetch distance
    static constexpr int PREFETCH_DISTANCE = 16;
    
    bool is_angle_in_arc(int dx, int dy) const {
        if (!is_arc_) return true;
        
        T angle = static_cast<T>(atan2(static_cast<double>(dy), static_cast<double>(dx)));
        if (angle < 0) angle += T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            // Arc crosses 0 degrees
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
    void generate_octants() {
        octant_count_ = 0;
        octant_index_ = 0;
        
        auto add_if_valid = [this](int x, int y) {
            if (is_angle_in_arc(x - center_.x, y - center_.y)) {
                octants_[static_cast<size_t>(octant_count_++)].pos = {x, y};
            }
        };
        
        // 8-way symmetry with special case handling
        if (x_ == 0 && y_ == 0) {
            // Center point only (radius 0)
            add_if_valid(center_.x, center_.y);
        } else if (x_ == 0) {
            // On axes - only 4 points
            add_if_valid(center_.x, center_.y + y_);
            add_if_valid(center_.x, center_.y - y_);
            add_if_valid(center_.x + y_, center_.y);
            add_if_valid(center_.x - y_, center_.y);
        } else if (y_ == 0) {
            // On horizontal axis - only 2 points
            add_if_valid(center_.x + x_, center_.y);
            add_if_valid(center_.x - x_, center_.y);
        } else if (x_ == y_) {
            // On diagonal - only 4 points
            add_if_valid(center_.x + x_, center_.y + y_);
            add_if_valid(center_.x - x_, center_.y + y_);
            add_if_valid(center_.x + x_, center_.y - y_);
            add_if_valid(center_.x - x_, center_.y - y_);
        } else {
            // General case - all 8 points
            add_if_valid(center_.x + x_, center_.y + y_);
            add_if_valid(center_.x - x_, center_.y + y_);
            add_if_valid(center_.x + x_, center_.y - y_);
            add_if_valid(center_.x - x_, center_.y - y_);
            add_if_valid(center_.x + y_, center_.y + x_);
            add_if_valid(center_.x - y_, center_.y + x_);
            add_if_valid(center_.x + y_, center_.y - x_);
            add_if_valid(center_.x - y_, center_.y - x_);
        }
    }
    
    void advance_to_next_position() {
        if (d_ < 0) {
            d_ += 2 * x_ + 3;
        } else {
            d_ += 2 * (x_ - y_) + 5;
            --y_;
        }
        ++x_;
    }
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Fill batch with pixels
        while (!current_batch_.is_full() && !done_) {
            // Add remaining octant points from previous iteration
            while (octant_index_ < octant_count_ && !current_batch_.is_full()) {
                current_batch_.add(octants_[static_cast<size_t>(octant_index_++)]);
            }
            
            // If we've processed all octants, move to next position
            if (octant_index_ >= octant_count_) {
                if (x_ <= y_) {
                    advance_to_next_position();
                    if (x_ > y_) {
                        done_ = true;
                        break;
                    }
                    generate_octants();
                } else {
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
        if (future_x <= y_) {
            prefetch_hint::prefetch_for_read(&center_);
        }
    }
    
public:
    /**
     * @brief Construct iterator for a full circle
     * @param center Center point of the circle
     * @param radius Radius of the circle
     * 
     * Creates an iterator that generates all pixels on the circle perimeter.
     * For floating-point coordinates, the center and radius are rounded to
     * the nearest integer values.
     * 
     * @note For radius 0, generates a single pixel at the center
     */
    batched_circle_iterator(point2<T> center, T radius) 
        : is_arc_(false), done_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
        }
        
        // Initialize midpoint algorithm
        x_ = 0;
        y_ = radius_;
        d_ = 1 - radius_;
        
        // Generate first set of octants
        generate_octants();
        octant_index_ = 0;
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Construct iterator for a circular arc
     * @tparam Angle Angle type (can be degrees, radians, or raw T)
     * @param center Center point of the circle
     * @param radius Radius of the circle
     * @param start_angle Starting angle of the arc
     * @param end_angle Ending angle of the arc
     * 
     * Creates an iterator that generates only pixels within the specified
     * angular range. Angles are measured counter-clockwise from the positive
     * x-axis. The arc is drawn from start_angle to end_angle in the 
     * counter-clockwise direction.
     * 
     * @note If start_angle > end_angle, the arc wraps around through 0°
     */
    template<typename Angle>
    batched_circle_iterator(point2<T> center, T radius, 
                           const Angle& start_angle, const Angle& end_angle)
        : is_arc_(true), done_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
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
        
        // Initialize midpoint algorithm
        x_ = 0;
        y_ = radius_;
        d_ = 1 - radius_;
        
        // Generate first set of octants
        generate_octants();
        octant_index_ = 0;
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Get the current batch of pixels
     * @return Const reference to the current pixel batch
     * 
     * Returns the current batch containing up to 16 pixels. The actual
     * number of valid pixels is indicated by the batch's count member.
     * 
     * @note The batch remains valid until next_batch() is called
     */
    const batch_type& current_batch() const {
        return current_batch_;
    }
    
    /**
     * @brief Check if the iterator has finished generating all pixels
     * @return true if no more pixels are available, false otherwise
     * 
     * This method returns true when all circle pixels have been generated
     * and the current batch is empty.
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
    batched_circle_iterator& next_batch() {
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
     * provided callback for each non-empty batch. This is more efficient
     * than manual iteration for processing all pixels.
     * 
     * Example:
     * @code
     * auto circle = make_batched_circle(center, radius);
     * circle.process_all([](const auto& batch) {
     *     for (size_t i = 0; i < batch.count; ++i) {
     *         draw_pixel(batch.pixels[i].pos);
     *     }
     * });
     * @endcode
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
 * @brief Batched filled circle iterator using horizontal spans
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates horizontal spans that fill a circle, producing
 * batches of spans for efficient rendering. Each span represents a horizontal
 * line segment within the circle.
 * 
 * The iterator uses the standard circle equation x² + y² = r² to calculate
 * the horizontal extent at each y-coordinate, working from top to bottom.
 * 
 * Example usage:
 * @code
 * auto filled = make_batched_filled_circle(center, radius);
 * while (!filled.at_end()) {
 *     const auto& batch = filled.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_horizontal_line(batch.pixels[i]);
 *     }
 *     filled.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_filled_circle_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using span_type = span;
    using batch_type = pixel_batch<span_type>;
    
private:
    point2i center_;
    int radius_;
    int y_;
    
    batch_type current_batch_;
    bool done_;
    
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Generate spans from top to bottom
        while (!current_batch_.is_full() && y_ >= -radius_) {
            int x = static_cast<int>(sqrt(static_cast<double>(radius_ * radius_ - y_ * y_)));
            
            span_type span;
            span.y = center_.y + y_;
            span.x_start = center_.x - x;
            span.x_end = center_.x + x;
            
            current_batch_.add(span);
            --y_;
        }
        
        if (y_ < -radius_) {
            done_ = true;
        }
    }
    
public:
    /**
     * @brief Construct filled circle iterator
     */
    batched_filled_circle_iterator(point2<T> center, T radius)
        : done_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
        }
        
        y_ = radius_;
        fill_batch();
    }
    
    const batch_type& current_batch() const { return current_batch_; }
    bool at_end() const { return done_ && current_batch_.is_empty(); }
    
    batched_filled_circle_iterator& next_batch() {
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
 * @brief Create a batched iterator for a complete circle
 * @tparam T Coordinate type
 * @param center Center point of the circle
 * @param radius Radius of the circle
 * @return A batched_circle_iterator instance
 * 
 * Factory function that creates a batched circle iterator. This is the
 * preferred way to create circle iterators as it provides type deduction.
 * 
 * @see batched_circle_iterator
 */
template<typename T>
auto make_batched_circle(point2<T> center, T radius) {
    return batched_circle_iterator<T>(center, radius);
}

/**
 * @brief Create a batched iterator for a circular arc
 * @tparam T Coordinate type
 * @tparam Angle Angle type (degrees, radians, etc.)
 * @param center Center point of the circle
 * @param radius Radius of the circle
 * @param start_angle Starting angle of the arc
 * @param end_angle Ending angle of the arc
 * @return A batched_circle_iterator instance configured for arc generation
 * 
 * Factory function that creates a batched arc iterator. Angles are
 * automatically converted to radians internally.
 * 
 * @see batched_circle_iterator
 */
template<typename T, typename Angle>
auto make_batched_arc(point2<T> center, T radius, 
                     const Angle& start_angle, const Angle& end_angle) {
    return batched_circle_iterator<T>(center, radius, start_angle, end_angle);
}

/**
 * @brief Create a batched iterator for a filled circle
 * @tparam T Coordinate type
 * @param center Center point of the circle
 * @param radius Radius of the circle
 * @return A batched_filled_circle_iterator instance
 * 
 * Factory function that creates a batched filled circle iterator,
 * which generates horizontal spans instead of individual pixels.
 * 
 * @see batched_filled_circle_iterator
 */
template<typename T>
auto make_batched_filled_circle(point2<T> center, T radius) {
    return batched_filled_circle_iterator<T>(center, radius);
}

} // namespace euler::dda