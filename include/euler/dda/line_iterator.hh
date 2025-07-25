/**
 * @file line_iterator.hh
 * @brief Basic line rasterization using Bresenham's algorithm
 * @ingroup DDAModule
 * 
 * This header provides the fundamental line drawing algorithm for the DDA
 * module. It implements Bresenham's algorithm, which uses only integer
 * arithmetic for optimal performance.
 * 
 * @section Usage
 * @code
 * // Draw a line from (0,0) to (100,50)
 * for (auto pixel : line_pixels(point2i{0, 0}, point2i{100, 50})) {
 *     draw_pixel(pixel.pos.x, pixel.pos.y);
 * }
 * 
 * // Using iterator interface
 * auto line = make_line_iterator(start, end);
 * while (line != line_iterator<int>::end()) {
 *     draw_pixel((*line).pos);
 *     ++line;
 * }
 * @endcode
 * 
 * @see aa_line_iterator.hh for antialiased line drawing
 * @see thick_line_iterator.hh for lines with width
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/dda_math.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>

namespace euler::dda {

/**
 * @brief Line iterator using Bresenham's algorithm
 * @tparam T Coordinate type
 * 
 * Implements efficient line rasterization with integer arithmetic.
 * Specializations exist for pure integer coordinates.
 */
template<typename T>
class line_iterator : public dda_iterator_base<line_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<line_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i current_;
    point2i end_;
    int dx_, dy_;
    int sx_, sy_;
    int error_;
    value_type current_pixel_;
    
    void init(point2i start, point2i end) {
        current_ = start;
        end_ = end;
        dx_ = abs(end.x - start.x);
        dy_ = abs(end.y - start.y);
        sx_ = start.x < end.x ? 1 : -1;
        sy_ = start.y < end.y ? 1 : -1;
        error_ = dx_ - dy_;
        current_pixel_.pos = current_;
        
        // Check if line has zero length
        if (start == end) {
            this->done_ = false;  // Will emit one pixel
        }
    }
    
public:
    /**
     * @brief Construct line iterator from integer endpoints
     */
    line_iterator(point2i start, point2i end) {
        init(start, end);
    }
    
    /**
     * @brief Construct line iterator from floating point endpoints
     */
    template<typename U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    line_iterator(point2<U> start, point2<U> end) {
        // Round to nearest integer coordinates
        init(round(start), round(end));
    }
    
    /**
     * @brief Construct clipped line iterator
     */
    line_iterator(point2<T> start, point2<T> end, 
                  const rectangle<T>& clip_rect) {
        // Clip line first
        if (!clip_rect.clip_line(start, end)) {
            this->done_ = true;
            return;
        }
        
        if constexpr (std::is_integral_v<T>) {
            init(start, end);
        } else {
            init(round(start), round(end));
        }
    }
    
    /**
     * @brief Get current pixel
     */
    constexpr value_type operator*() const {
        return current_pixel_;
    }
    
    /**
     * @brief Advance to next pixel
     */
    line_iterator& operator++() {
        if (current_ == end_) {
            this->done_ = true;
            return *this;
        }
        
        int e2 = 2 * error_;
        if (e2 > -dy_) {
            error_ -= dy_;
            current_.x += sx_;
        }
        if (e2 < dx_) {
            error_ += dx_;
            current_.y += sy_;
        }
        
        current_pixel_.pos = current_;
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    line_iterator operator++(int) {
        line_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    /**
     * @brief Get sentinel for range-based for loops
     */
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Optimized specialization for integer coordinates
 */
template<>
class line_iterator<int> : public dda_iterator_base<line_iterator<int>, pixel<int>, int> {
    using base = dda_iterator_base<line_iterator<int>, pixel<int>, int>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    // Pure integer implementation
    int x_, y_;
    int x2_, y2_;
    int dx_, dy_;
    int sx_, sy_;
    int error_;
    
public:
    /**
     * @brief Construct integer line iterator
     */
    line_iterator(point2i start, point2i end)
        : x_(start.x), y_(start.y), x2_(end.x), y2_(end.y) {
        dx_ = abs(x2_ - x_);
        dy_ = abs(y2_ - y_);
        sx_ = x_ < x2_ ? 1 : -1;
        sy_ = y_ < y2_ ? 1 : -1;
        error_ = dx_ - dy_;
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        return {{x_, y_}};
    }
    
    /**
     * @brief Advance to next pixel
     */
    line_iterator& operator++() {
        if (x_ == x2_ && y_ == y2_) {
            this->done_ = true;
            return *this;
        }
        
        int e2 = 2 * error_;
        if (e2 > -dy_) {
            error_ -= dy_;
            x_ += sx_;
        }
        if (e2 < dx_) {
            error_ += dx_;
            y_ += sy_;
        }
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    line_iterator operator++(int) {
        line_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper function to create line iterator
 */
template<typename T>
auto make_line_iterator(point2<T> start, point2<T> end) {
    return line_iterator<T>(start, end);
}

/**
 * @brief Range wrapper for line iteration
 */
template<typename T>
class line_range {
    point2<T> start_;
    point2<T> end_;
    
public:
    line_range(point2<T> start, point2<T> end) 
        : start_(start), end_(end) {}
    
    auto begin() const { return line_iterator<T>(start_, end_); }
    auto end() const { return dda_sentinel{}; }
};

/**
 * @brief Helper to create line range for range-based for loops
 */
template<typename T>
auto line_pixels(point2<T> start, point2<T> end) {
    return line_range<T>(start, end);
}

} // namespace euler::dda