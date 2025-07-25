/**
 * @file circle_iterator.hh
 * @brief Circle rasterization using midpoint algorithm
 * @ingroup DDAModule
 * 
 * This header provides efficient circle rasterization using the midpoint
 * circle algorithm (also known as Bresenham's circle algorithm). The
 * algorithm exploits 8-way symmetry to minimize calculations.
 * 
 * @section features Features
 * - Integer-only arithmetic for maximum performance
 * - 8-way symmetry reduces computations by 8x
 * - Support for both outline and filled circles
 * - Subpixel center positioning with proper rounding
 * 
 * @section algorithm Algorithm
 * The midpoint algorithm works by:
 * - Starting at (0, radius) and moving clockwise
 * - Using error terms to decide between moving horizontally or diagonally
 * - Generating 8 symmetric points for each calculated point
 * 
 * @section usage Usage
 * @code
 * // Draw circle outline
 * for (auto pixel : circle_pixels(center, radius)) {
 *     draw_pixel(pixel.pos);
 * }
 * 
 * // Draw filled circle
 * auto filled = make_filled_circle_iterator(center, radius);
 * while (filled != filled_circle_iterator<float>::end()) {
 *     auto span = *filled;
 *     draw_horizontal_line(span.y, span.x_start, span.x_end);
 *     ++filled;
 * }
 * @endcode
 * 
 * @see ellipse_iterator.hh for elliptical shapes
 * @see arc_iterator.hh for partial circles
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/dda_math.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/math/trigonometry.hh>
#include <array>

namespace euler::dda {

/**
 * @brief Circle iterator using midpoint algorithm
 * @tparam T Coordinate type
 * 
 * Uses 8-way symmetry for efficient circle rasterization.
 * Can also generate arcs by specifying start and end angles.
 */
template<typename T>
class circle_iterator : public dda_iterator_base<circle_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<circle_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int radius_;
    int x_, y_;
    int d_;  // Decision parameter
    
    // For arc support
    bool is_arc_;
    T start_angle_, end_angle_;
    
    // 8-way symmetry points
    std::array<value_type, 8> octants_;
    int octant_count_;
    int octant_index_;
    
    void generate_octants() {
        octant_count_ = 0;
        octant_index_ = 0;
        
        auto add_if_valid = [this](int x, int y) {
            if (!is_arc_ || is_angle_in_arc(x - center_.x, y - center_.y)) {
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
    
    bool is_angle_in_arc(int dx, int dy) const {
        T angle = static_cast<T>(atan2(static_cast<double>(dy), static_cast<double>(dx)));
        if (angle < 0) angle += T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            // Arc crosses 0 degrees
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
public:
    /**
     * @brief Construct full circle iterator
     */
    circle_iterator(point2<T> center, T radius) 
        : is_arc_(false) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
        }
        
        x_ = 0;
        y_ = radius_;
        d_ = 3 - 2 * radius_;
        
        if (radius_ <= 0) {
            // Single pixel at center for zero radius
            octants_[0].pos = center_;
            octant_count_ = 1;
            octant_index_ = 0;
        } else {
            generate_octants();
        }
    }
    
    /**
     * @brief Construct arc iterator
     */
    template<typename Angle>
    circle_iterator(point2<T> center, T radius, 
                    const Angle& start_angle, const Angle& end_angle)
        : is_arc_(true) {
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
        
        x_ = 0;
        y_ = radius_;
        d_ = 3 - 2 * radius_;
        
        if (radius_ <= 0) {
            octants_[0].pos = center_;
            octant_count_ = 1;
            octant_index_ = 0;
        } else {
            generate_octants();
        }
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        return octants_[static_cast<size_t>(octant_index_)];
    }
    
    /**
     * @brief Advance to next pixel
     */
    circle_iterator& operator++() {
        octant_index_++;
        
        if (octant_index_ >= octant_count_) {
            // Move to next position
            if (x_ >= y_) {
                this->done_ = true;
                return *this;
            }
            
            x_++;
            
            if (d_ > 0) {
                y_--;
                d_ = d_ + 4 * (x_ - y_) + 10;
            } else {
                d_ = d_ + 4 * x_ + 6;
            }
            
            generate_octants();
            
            if (octant_count_ == 0) {
                this->done_ = true;
            }
        }
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    circle_iterator operator++(int) {
        circle_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Filled circle iterator using scanline approach
 * @tparam T Coordinate type
 * 
 * Generates horizontal spans for efficient filling.
 */
template<typename T>
class filled_circle_iterator : public dda_iterator_base<filled_circle_iterator<T>, span, T> {
    using base = dda_iterator_base<filled_circle_iterator<T>, span, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int radius_;
    int y_;
    
public:
    /**
     * @brief Construct filled circle iterator
     */
    filled_circle_iterator(point2<T> center, T radius) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            radius_ = radius;
        } else {
            center_ = round(center);
            radius_ = static_cast<int>(round(radius));
        }
        
        y_ = -radius_;
        
        if (radius_ < 0) {
            this->done_ = true;
        }
    }
    
    /**
     * @brief Get current span
     */
    value_type operator*() const {
        int x_offset = static_cast<int>(sqrt(
            static_cast<T>(radius_ * radius_ - y_ * y_)));
        
        return {
            center_.y + y_,
            center_.x - x_offset,
            center_.x + x_offset
        };
    }
    
    /**
     * @brief Advance to next span
     */
    filled_circle_iterator& operator++() {
        y_++;
        if (y_ > radius_) {
            this->done_ = true;
        }
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    filled_circle_iterator operator++(int) {
        filled_circle_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper to create circle iterator
 */
template<typename T>
auto make_circle_iterator(point2<T> center, T radius) {
    return circle_iterator<T>(center, radius);
}

/**
 * @brief Helper to create arc iterator
 */
template<typename T, typename Angle>
auto make_arc_iterator(point2<T> center, T radius, 
                      const Angle& start, const Angle& end) {
    return circle_iterator<T>(center, radius, start, end);
}

/**
 * @brief Helper to create filled circle iterator
 */
template<typename T>
auto make_filled_circle_iterator(point2<T> center, T radius) {
    return filled_circle_iterator<T>(center, radius);
}

/**
 * @brief Range wrapper for circle pixels
 */
template<typename T>
class circle_range {
    point2<T> center_;
    T radius_;
    
public:
    circle_range(point2<T> center, T radius) 
        : center_(center), radius_(radius) {}
    
    auto begin() const { return circle_iterator<T>(center_, radius_); }
    auto end() const { return dda_sentinel{}; }
};

/**
 * @brief Helper to create circle range
 */
template<typename T>
auto circle_pixels(point2<T> center, T radius) {
    return circle_range<T>(center, radius);
}

} // namespace euler::dda