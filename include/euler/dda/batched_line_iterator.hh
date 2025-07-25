/**
 * @file batched_line_iterator.hh
 * @brief Batched line rasterization for improved performance
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <algorithm>

namespace euler::dda {

/**
 * @brief Batched line iterator using Bresenham's algorithm
 * @tparam T Coordinate type
 * 
 * Generates pixels in batches for improved cache utilization and
 * potential SIMD processing. Particularly effective for long lines.
 */
template<typename T>
class batched_line_iterator {
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = pixel<int>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // Line endpoints
    point2i start_;
    point2i end_;
    
    // Bresenham state
    point2i current_;
    int dx_, dy_;
    int sx_, sy_;
    int error_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    // Prefetch distance for next batch
    static constexpr int PREFETCH_DISTANCE = 32;
    
    EULER_HOT
    void fill_batch() {
        current_batch_.clear();
        
        if (EULER_UNLIKELY(done_)) return;
        
        // Fill batch with pixels
        // Manual unrolling for better performance without pragma warnings
        while (!current_batch_.is_full() && current_ != end_) {
            // Process up to 4 pixels per iteration for better performance
            if (!current_batch_.is_full() && current_ != end_) {
                current_batch_.add({current_});
                advance_pixel();
            }
            if (!current_batch_.is_full() && current_ != end_) {
                current_batch_.add({current_});
                advance_pixel();
            }
            if (!current_batch_.is_full() && current_ != end_) {
                current_batch_.add({current_});
                advance_pixel();
            }
            if (!current_batch_.is_full() && current_ != end_) {
                current_batch_.add({current_});
                advance_pixel();
            }
        }
        
        // Check if we've reached the end
        if (current_ == end_) {
            // Add final pixel if not already added
            if (current_batch_.is_empty() || 
                current_batch_.pixels[current_batch_.count - 1].pos != end_) {
                current_batch_.add({end_});
            }
            done_ = true;
        }
        
        // Prefetch memory for next batch if applicable
        if (!done_ && current_ != end_) {
            prefetch_next_batch();
        }
    }
    
    EULER_ALWAYS_INLINE
    void advance_pixel() {
        int e2 = 2 * error_;
        
        if (e2 > -dy_) {
            error_ -= dy_;
            current_.x += sx_;
        }
        
        if (e2 < dx_) {
            error_ += dx_;
            current_.y += sy_;
        }
    }
    
    void prefetch_next_batch() {
        // Calculate approximate position after PREFETCH_DISTANCE pixels
        point2i prefetch_pos = current_;
        int temp_error = error_;
        
        for (int i = 0; i < PREFETCH_DISTANCE && prefetch_pos != end_; ++i) {
            int e2 = 2 * temp_error;
            if (e2 > -dy_) {
                temp_error -= dy_;
                prefetch_pos.x += sx_;
            }
            if (e2 < dx_) {
                temp_error += dx_;
                prefetch_pos.y += sy_;
            }
        }
        
        // Prefetch the memory region around the future position
        prefetch_hint::prefetch<point2i, 0, 1>(&prefetch_pos);
    }
    
public:
    /**
     * @brief Construct batched line iterator
     * @param start Start point
     * @param end End point
     */
    batched_line_iterator(point2<T> start, point2<T> end)
        : start_(round(start)), end_(round(end)), current_(start_), done_(false) {
        
        dx_ = abs(end_.x - start_.x);
        dy_ = abs(end_.y - start_.y);
        sx_ = start_.x < end_.x ? 1 : -1;
        sy_ = start_.y < end_.y ? 1 : -1;
        error_ = dx_ - dy_;
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Get current batch of pixels
     */
    const batch_type& current_batch() const {
        return current_batch_;
    }
    
    /**
     * @brief Check if iterator is at end
     */
    bool at_end() const {
        return done_ && current_batch_.is_empty();
    }
    
    /**
     * @brief Advance to next batch
     */
    batched_line_iterator& next_batch() {
        if (!done_) {
            fill_batch();
        } else {
            // Clear the batch when we're done to ensure at_end() works correctly
            current_batch_.clear();
        }
        return *this;
    }
    
    /**
     * @brief Process all pixels with a callback
     */
    template<typename Callback>
    void process_all(Callback&& callback) {
        while (!at_end()) {
            batch_processor<T>::process_pixel_batch(current_batch_, callback);
            next_batch();
        }
    }
};

/**
 * @brief Batched antialiased line iterator
 * @tparam T Coordinate type (must be floating point)
 */
template<typename T>
class batched_aa_line_iterator {
    static_assert(std::is_floating_point_v<T>, 
                  "Antialiased lines require floating point coordinates");
                  
public:
    using coord_type = T;
    using point_type = point2<T>;
    using pixel_type = aa_pixel<T>;
    using batch_type = pixel_batch<pixel_type>;
    
private:
    // Wu's algorithm state
    T x_, y_;
    T x_end_, y_end_;
    T dx_, dy_;
    T gradient_;
    bool steep_;
    int step_;
    int steps_;
    
    // Batch state
    batch_type current_batch_;
    bool done_;
    
    static T fpart(T x) { return x - floor(x); }
    static T rfpart(T x) { return T(1) - fpart(x); }
    
    void swap_if_steep(T& x1, T& y1, T& x2, T& y2) {
        if (abs(y2 - y1) > abs(x2 - x1)) {
            std::swap(x1, y1);
            std::swap(x2, y2);
            steep_ = true;
        } else {
            steep_ = false;
        }
    }
    
    void fill_batch() {
        current_batch_.clear();
        
        if (done_ || step_ >= steps_) {
            done_ = true;
            return;
        }
        
        // Process multiple pixels at once
        while (!current_batch_.is_full() && step_ < steps_) {
            T intery = y_ + gradient_ * static_cast<T>(step_);
            T x_pos = x_ + static_cast<T>(step_);
            
            // Generate two pixels with complementary coverage
            emit_pixel(x_pos, floor(intery), rfpart(intery));
            emit_pixel(x_pos, floor(intery) + 1, fpart(intery));
            
            step_++;
        }
        
        // Prefetch for next batch
        if (step_ < steps_) {
            prefetch_next_batch();
        }
    }
    
    void emit_pixel(T x, T y, T coverage) {
        if (coverage > T(0.001)) {
            pixel_type p;
            if (steep_) {
                p.pos = {static_cast<T>(floor(y)), static_cast<T>(floor(x))};
            } else {
                p.pos = {static_cast<T>(floor(x)), static_cast<T>(floor(y))};
            }
            p.coverage = static_cast<float>(coverage);
            p.distance = 0.0f;
            
            current_batch_.add(p);
        }
    }
    
    void prefetch_next_batch() {
        // Prefetch memory for future pixel positions
        constexpr int PREFETCH_STEPS = 16;
        int future_step = min(step_ + PREFETCH_STEPS, steps_ - 1);
        
        T future_x = x_ + static_cast<T>(future_step);
        T future_y = y_ + gradient_ * static_cast<T>(future_step);
        
        pixel_type future_pixel;
        future_pixel.pos = {future_x, future_y};
        
        prefetch_hint::prefetch_for_read(&future_pixel);
    }
    
public:
    /**
     * @brief Construct batched antialiased line iterator
     */
    batched_aa_line_iterator(point2<T> start, point2<T> end) : done_(false) {
        T x1 = start.x, y1 = start.y;
        T x2 = end.x, y2 = end.y;
        
        swap_if_steep(x1, y1, x2, y2);
        
        if (x1 > x2) {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        
        dx_ = x2 - x1;
        dy_ = y2 - y1;
        gradient_ = (dx_ == 0) ? 1 : dy_ / dx_;
        
        // First endpoint
        T xend = round(x1);
        T yend = y1 + gradient_ * (xend - x1);
        
        x_ = xend;
        y_ = yend;
        x_end_ = round(x2);
        
        steps_ = static_cast<int>(x_end_ - x_) + 1;
        step_ = 0;
        
        // Fill first batch
        fill_batch();
    }
    
    /**
     * @brief Get current batch of pixels
     */
    const batch_type& current_batch() const {
        return current_batch_;
    }
    
    /**
     * @brief Check if iterator is at end
     */
    bool at_end() const {
        return done_ && current_batch_.is_empty();
    }
    
    /**
     * @brief Advance to next batch
     */
    batched_aa_line_iterator& next_batch() {
        if (!done_) {
            fill_batch();
        } else {
            // Clear the batch when we're done to ensure at_end() works correctly
            current_batch_.clear();
        }
        return *this;
    }
    
    /**
     * @brief Process all pixels into a coverage buffer
     */
    void accumulate_coverage(T* coverage_buffer, size_t buffer_width) {
        while (!at_end()) {
            batch_processor<T>::process_aa_batch(current_batch_, 
                                               coverage_buffer, 
                                               buffer_width);
            next_batch();
        }
    }
};

/**
 * @brief Helper functions for creating batched iterators
 */
template<typename T>
auto make_batched_line(point2<T> start, point2<T> end) {
    return batched_line_iterator<T>(start, end);
}

template<typename T>
auto make_batched_aa_line(point2<T> start, point2<T> end) {
    return batched_aa_line_iterator<T>(start, end);
}

} // namespace euler::dda