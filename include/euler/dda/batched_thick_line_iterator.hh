/**
 * @file batched_thick_line_iterator.hh
 * @brief Batched thick line rasterization for improved performance
 * @author Euler Library Contributors
 * @date 2024
 * 
 * This file provides batched versions of thick line rasterization iterators
 * that generate pixels in groups for improved cache utilization. The iterators
 * support both pixel-based and span-based approaches for rendering lines with
 * specified thickness.
 * 
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/thick_line_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/dda/circle_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace euler::dda {

/**
 * @brief Batched thick line iterator using brush-based approach
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates pixels for thick lines by sweeping a circular brush
 * along the line path. It produces pixels in batches for improved performance
 * compared to single-pixel iteration.
 * 
 * Key features:
 * - Generates pixels in batches of up to 16 for cache efficiency
 * - Uses circular brush for consistent thickness at all angles
 * - Avoids duplicate pixels through internal tracking
 * - Handles line endpoints with proper capping
 * - Supports fractional thickness values (rounded to nearest pixel)
 * 
 * The thickness parameter represents the diameter of the circular brush.
 * A thickness of 1 produces a standard single-pixel line.
 * 
 * @note This iterator may produce pixels in non-sequential order due to
 *       the brush-based approach. Duplicate pixels are automatically filtered.
 * 
 * Example usage:
 * @code
 * auto thick_line = make_batched_thick_line(start, end, 5.0f);
 * while (!thick_line.at_end()) {
 *     const auto& batch = thick_line.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         draw_pixel(batch.pixels[i].pos);
 *     }
 *     thick_line.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_thick_line_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = pixel<int>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // Line parameters
    line_iterator<T> line_iter_;
    T thickness_;
    int radius_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // Duplicate tracking
    std::unordered_set<point2i, point2i_hash<int>> emitted_pixels_;
    
    // Temporary storage for brush pixels
    std::vector<pixel_type> brush_pixels_;
    
    // Prefetch distance
    static constexpr int PREFETCH_DISTANCE = 8;
    
    void generate_brush_pixels(point2i center) {
        brush_pixels_.clear();
        
        // Use filled circle iterator for the brush
        for (auto span : filled_circle_range<int>(center, radius_)) {
            for (int x = span.x_start; x <= span.x_end; ++x) {
                point2i p{x, span.y};
                if (emitted_pixels_.find(p) == emitted_pixels_.end()) {
                    brush_pixels_.push_back({p});
                    emitted_pixels_.insert(p);
                }
            }
        }
    }
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Fill batch with pixels from multiple brush positions
        while (!current_batch_.is_full() && line_iter_ != line_iterator<T>::end()) {
            auto center = (*line_iter_).pos;
            
            // Generate brush pixels for this position
            generate_brush_pixels(center);
            
            // Add brush pixels to batch
            for (const auto& pixel : brush_pixels_) {
                if (current_batch_.is_full()) {
                    // Save remaining pixels for next batch
                    return;
                }
                current_batch_.add(pixel);
            }
            
            ++line_iter_;
        }
        
        // Check if we're done
        if (line_iter_ == line_iterator<T>::end()) {
            done_ = true;
        }
        
        // Prefetch memory for next batch if applicable
        if (!done_ && !current_batch_.is_empty()) {
            prefetch_next_batch();
        }
    }
    
    void prefetch_next_batch() {
        // Prefetch future line positions
        if (line_iter_ != line_iterator<T>::end()) {
            auto future_pos = (*line_iter_).pos;
            prefetch_hint::prefetch_for_read(&future_pos);
        }
    }
    
public:
    /**
     * @brief Construct batched thick line iterator
     * @param start Starting point of the line
     * @param end Ending point of the line
     * @param thickness Line thickness (diameter of the circular brush)
     * 
     * Creates an iterator that generates pixels for a thick line from start
     * to end. The thickness parameter specifies the diameter of the circular
     * brush used to sweep along the line path.
     * 
     * For floating-point coordinates and thickness, values are rounded to
     * the nearest integer. A thickness less than 1 is treated as 1.
     * 
     * @note The iterator pre-allocates memory based on the expected number
     *       of pixels to improve performance
     */
    batched_thick_line_iterator(point2<T> start, point2<T> end, T thickness)
        : line_iter_(start, end), thickness_(thickness), done_(false) {
        
        radius_ = static_cast<int>(round(thickness / T(2)));
        if (radius_ < 0) radius_ = 0;
        
        // Reserve space for temporary storage
        brush_pixels_.reserve(static_cast<size_t>(pi * static_cast<T>(radius_) * static_cast<T>(radius_)));
        
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
    batched_thick_line_iterator& next_batch() {
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
 * @brief Batched thick line iterator using horizontal spans
 * @tparam T Coordinate type (int or float)
 * 
 * This iterator generates horizontal spans for thick lines, providing
 * a more efficient representation than individual pixels. Each span
 * represents a continuous horizontal segment of the thick line.
 * 
 * The algorithm works by:
 * 1. Moving along the line's central axis
 * 2. Calculating the perpendicular extent at each position
 * 3. Generating horizontal spans that cover the line's thickness
 * 
 * This approach is particularly efficient for:
 * - Hardware-accelerated horizontal line drawing
 * - Fill rate limited rendering systems
 * - Lines with large thickness values
 * 
 * Example usage:
 * @code
 * auto thick_line = make_batched_thick_line_spans(start, end, 10.0f);
 * while (!thick_line.at_end()) {
 *     const auto& batch = thick_line.current_batch();
 *     for (size_t i = 0; i < batch.count; ++i) {
 *         const auto& span = batch.pixels[i];
 *         draw_horizontal_line(span.y, span.x_start, span.x_end);
 *     }
 *     thick_line.next_batch();
 * }
 * @endcode
 */
template<typename T>
class batched_thick_line_span_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using span_type = span;
    using batch_type = pixel_batch<span_type>;
    
private:
    // Line parameters
    point2<T> start_, end_;
    T thickness_;
    T half_thickness_;
    
    // Line direction and perpendicular
    vec2<T> direction_;
    vec2<T> perpendicular_;
    T line_length_;
    
    // Current position along line
    T current_t_;
    T dt_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // Scanline buffer for span generation
    struct scanline_info {
        int y;
        int x_min, x_max;
    };
    std::vector<scanline_info> scanlines_;
    
    void generate_spans_at_position(T t) {
        scanlines_.clear();
        
        // Calculate center point along line
        vec2<T> displacement = direction_ * (t * line_length_);
        point2<T> center = start_ + displacement;
        
        // Generate spans for the rectangular area
        int y_min = static_cast<int>(floor(std::min({center.y - half_thickness_, 
                                                     center.y + half_thickness_})));
        int y_max = static_cast<int>(ceil(std::max({center.y - half_thickness_, 
                                                    center.y + half_thickness_})));
        
        for (int y = y_min; y <= y_max; ++y) {
            // Calculate x extent at this y
            T dy = static_cast<T>(y) - center.y;
            T discriminant = half_thickness_ * half_thickness_ - dy * dy;
            
            if (discriminant >= 0) {
                T max_offset = sqrt(discriminant);
                int x_min = static_cast<int>(floor(center.x - max_offset));
                int x_max = static_cast<int>(ceil(center.x + max_offset));
                scanlines_.push_back({y, x_min, x_max});
            }
        }
    }
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Fill batch with spans
        while (!current_batch_.is_full() && current_t_ <= T(1)) {
            generate_spans_at_position(current_t_);
            
            for (const auto& scanline : scanlines_) {
                if (current_batch_.is_full()) break;
                
                span_type span;
                span.y = scanline.y;
                span.x_start = scanline.x_min;
                span.x_end = scanline.x_max;
                current_batch_.add(span);
            }
            
            current_t_ += dt_;
        }
        
        if (current_t_ > T(1)) {
            done_ = true;
        }
    }
    
public:
    /**
     * @brief Construct batched thick line span iterator
     */
    batched_thick_line_span_iterator(point2<T> start, point2<T> end, T thickness)
        : start_(start), end_(end), thickness_(thickness), 
          half_thickness_(thickness / T(2)), current_t_(0), done_(false) {
        
        // Calculate line direction
        vec2<T> diff = end_ - start_;
        line_length_ = length(diff);
        
        if (line_length_ > T(0)) {
            direction_ = diff / line_length_;
            // Perpendicular vector (rotate 90 degrees)
            perpendicular_ = vec2<T>{-direction_[1], direction_[0]};
            
            // Adaptive step size based on thickness
            dt_ = T(1) / (line_length_ / thickness_ + T(1));
        } else {
            // Degenerate line (single point)
            direction_ = vec2<T>{T(1), T(0)};
            perpendicular_ = vec2<T>{T(0), T(1)};
            dt_ = T(2); // Will finish in one iteration
        }
        
        fill_batch();
    }
    
    const batch_type& current_batch() const { return current_batch_; }
    bool at_end() const { return done_ && current_batch_.is_empty(); }
    
    batched_thick_line_span_iterator& next_batch() {
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
 * @brief Create a batched iterator for a thick line using individual pixels
 * @tparam T Coordinate type
 * @param start Starting point of the line
 * @param end Ending point of the line
 * @param thickness Line thickness (diameter of the circular brush)
 * @return A batched_thick_line_iterator instance
 * 
 * Factory function that creates a batched thick line iterator using the
 * brush-based approach. This generates individual pixels and is best for
 * lines with small to medium thickness.
 * 
 * @see batched_thick_line_iterator
 */
template<typename T>
auto make_batched_thick_line(point2<T> start, point2<T> end, T thickness) {
    return batched_thick_line_iterator<T>(start, end, thickness);
}

/**
 * @brief Create a batched iterator for a thick line using horizontal spans
 * @tparam T Coordinate type
 * @param start Starting point of the line
 * @param end Ending point of the line
 * @param thickness Line thickness
 * @return A batched_thick_line_span_iterator instance
 * 
 * Factory function that creates a batched thick line iterator using the
 * span-based approach. This generates horizontal spans and is more efficient
 * for lines with large thickness values or when hardware acceleration is available.
 * 
 * @see batched_thick_line_span_iterator
 */
template<typename T>
auto make_batched_thick_line_spans(point2<T> start, point2<T> end, T thickness) {
    return batched_thick_line_span_iterator<T>(start, end, thickness);
}

} // namespace euler::dda