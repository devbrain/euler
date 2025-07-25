/**
 * @file ellipse_iterator.hh
 * @brief Ellipse rasterization using midpoint algorithm
 * @ingroup DDAModule
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
 * @brief Ellipse iterator using midpoint algorithm
 * @tparam T Coordinate type
 * 
 * Uses 4-way symmetry for efficient ellipse rasterization.
 * Supports axis-aligned ellipses.
 */
template<typename T>
class ellipse_iterator : public dda_iterator_base<ellipse_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<ellipse_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int a_, b_;  // Semi-major and semi-minor axes
    int x_, y_;
    int a2_, b2_;  // a² and b²
    int fa2_, fb2_;  // 4a² and 4b²
    
    // Decision parameters for two regions
    int64_t d1_, d2_;
    bool in_region2_;
    
    // For arc support
    bool is_arc_;
    T start_angle_, end_angle_;
    
    // 4-way symmetry points
    std::array<value_type, 4> quadrants_;
    int quadrant_count_;
    int quadrant_index_;
    
    void generate_quadrants() {
        quadrant_count_ = 0;
        quadrant_index_ = 0;
        
        auto add_if_valid = [this](int x, int y) {
            if (!is_arc_ || is_angle_in_arc(x - center_.x, y - center_.y)) {
                quadrants_[static_cast<size_t>(quadrant_count_++)].pos = {x, y};
            }
        };
        
        // 4-way symmetry
        add_if_valid(center_.x + x_, center_.y + y_);
        add_if_valid(center_.x - x_, center_.y + y_);
        add_if_valid(center_.x + x_, center_.y - y_);
        add_if_valid(center_.x - x_, center_.y - y_);
    }
    
    bool is_angle_in_arc(int dx, int dy) const {
        // Account for ellipse stretching when computing angle
        T angle = atan2(static_cast<T>(dy) * static_cast<T>(a_), static_cast<T>(dx) * static_cast<T>(b_));
        if (angle < 0) angle += T(2 * pi);
        
        if (start_angle_ <= end_angle_) {
            return angle >= start_angle_ && angle <= end_angle_;
        } else {
            return angle >= start_angle_ || angle <= end_angle_;
        }
    }
    
public:
    /**
     * @brief Construct full ellipse iterator
     */
    ellipse_iterator(point2<T> center, T semi_major, T semi_minor) 
        : is_arc_(false) {
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
            in_region2_ = true; // Set this to avoid undefined behavior
            y_ = -1; // This will make operator++ set done_ = true
            return;
        }
        
        // Initialize for region 1 (gradient < -1)
        x_ = 0;
        y_ = b_;
        a2_ = a_ * a_;
        b2_ = b_ * b_;
        fa2_ = 4 * a2_;
        fb2_ = 4 * b2_;
        
        // Initial decision parameter for region 1
        d1_ = b2_ - a2_ * b_ + a2_ / 4;
        in_region2_ = false;
        
        generate_quadrants();
    }
    
    /**
     * @brief Construct arc iterator
     */
    template<typename Angle>
    ellipse_iterator(point2<T> center, T semi_major, T semi_minor,
                     const Angle& start_angle, const Angle& end_angle)
        : is_arc_(true) {
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
        
        if (a_ <= 0 || b_ <= 0) {
            quadrants_[0].pos = center_;
            quadrant_count_ = 1;
            quadrant_index_ = 0;
            in_region2_ = true; // Set this to avoid undefined behavior
            y_ = -1; // This will make operator++ set done_ = true
            return;
        }
        
        x_ = 0;
        y_ = b_;
        a2_ = a_ * a_;
        b2_ = b_ * b_;
        fa2_ = 4 * a2_;
        fb2_ = 4 * b2_;
        d1_ = b2_ - a2_ * b_ + a2_ / 4;
        in_region2_ = false;
        
        generate_quadrants();
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        return quadrants_[static_cast<size_t>(quadrant_index_)];
    }
    
    /**
     * @brief Advance to next pixel
     */
    ellipse_iterator& operator++() {
        quadrant_index_++;
        
        if (quadrant_index_ >= quadrant_count_) {
            // Move to next position
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
                
                if (y_ < 0) {
                    this->done_ = true;
                    return *this;
                }
                
                if (d2_ > 0) {
                    d2_ += a2_ - fa2_ * y_;
                } else {
                    x_++;
                    d2_ += fb2_ * x_ - fa2_ * y_ + a2_;
                }
            }
            
            generate_quadrants();
            
            if (quadrant_count_ == 0) {
                this->done_ = true;
            }
        }
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    ellipse_iterator operator++(int) {
        ellipse_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Filled ellipse iterator using scanline approach
 * @tparam T Coordinate type
 */
template<typename T>
class filled_ellipse_iterator : public dda_iterator_base<filled_ellipse_iterator<T>, span, T> {
    using base = dda_iterator_base<filled_ellipse_iterator<T>, span, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2i center_;
    int a_, b_;  // Semi-major and semi-minor axes
    int y_;
    T a2_, b2_;  // For floating point precision
    
public:
    /**
     * @brief Construct filled ellipse iterator
     */
    filled_ellipse_iterator(point2<T> center, T semi_major, T semi_minor) {
        if constexpr (std::is_integral_v<T>) {
            center_ = center;
            a_ = semi_major;
            b_ = semi_minor;
            a2_ = T(a_ * a_);
            b2_ = T(b_ * b_);
        } else {
            center_ = round(center);
            a_ = static_cast<int>(round(semi_major));
            b_ = static_cast<int>(round(semi_minor));
            a2_ = semi_major * semi_major;
            b2_ = semi_minor * semi_minor;
        }
        
        y_ = -b_;
        
        if (a_ <= 0 || b_ <= 0) {
            this->done_ = true;
        }
    }
    
    /**
     * @brief Get current span
     */
    value_type operator*() const {
        // x²/a² + y²/b² = 1
        // x = a * sqrt(1 - y²/b²)
        T y_ratio = static_cast<T>(y_) / static_cast<T>(b_);
        T x_normalized = sqrt(max(T(0), T(1) - y_ratio * y_ratio));
        int x_offset = static_cast<int>(static_cast<T>(a_) * x_normalized);
        
        return {
            center_.y + y_,
            center_.x - x_offset,
            center_.x + x_offset
        };
    }
    
    /**
     * @brief Advance to next span
     */
    filled_ellipse_iterator& operator++() {
        y_++;
        if (y_ > b_) {
            this->done_ = true;
        }
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    filled_ellipse_iterator operator++(int) {
        filled_ellipse_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Antialiased ellipse iterator
 * @tparam T Coordinate type
 */
template<typename T>
class aa_ellipse_iterator : public dda_iterator_base<aa_ellipse_iterator<T>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>, 
                  "Antialiased ellipses require floating point coordinates");
    
    using base = dda_iterator_base<aa_ellipse_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    point2<T> center_;
    T a_, b_;  // Semi-axes
    T a2_, b2_;
    
    // Current position in parameter space
    T t_;  // Parameter [0, 2π]
    T dt_; // Step size
    
    // Buffer for multiple pixels per step
    std::array<value_type, 4> pixels_;
    int pixel_count_;
    int pixel_index_;
    
    T ellipse_distance(point2<T> p) const {
        // Approximate distance to ellipse boundary
        T dx = (p.x - center_.x) / a_;
        T dy = (p.y - center_.y) / b_;
        T d = dx * dx + dy * dy;
        return abs(sqrt(d) - T(1)) * min(a_, b_);
    }
    
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (t_ >= T(2 * pi)) {
            this->done_ = true;
            return;
        }
        
        // Current point on ellipse
        T cos_t = cos(t_);
        T sin_t = sin(t_);
        point2<T> p{center_.x + a_ * cos_t, center_.y + b_ * sin_t};
        
        // Add main pixel
        point2i pi = round(p);
        auto& main_pixel = pixels_[static_cast<size_t>(pixel_count_++)];
        main_pixel.pos = point2<T>{static_cast<T>(pi.x), static_cast<T>(pi.y)};
        main_pixel.distance = static_cast<float>(ellipse_distance(point2<T>(static_cast<T>(pi.x) + T(0.5), static_cast<T>(pi.y) + T(0.5))));
        main_pixel.coverage = max(0.0f, 1.0f - main_pixel.distance);
        
        // Add neighboring pixels with coverage
        auto check_neighbor = [&](int dx, int dy) {
            if (pixel_count_ >= static_cast<int>(pixels_.size())) return;
            point2i np{pi.x + dx, pi.y + dy};
            T d = ellipse_distance(point2<T>(static_cast<T>(np.x) + T(0.5), static_cast<T>(np.y) + T(0.5)));
            if (d < T(1)) {
                auto& pixel = pixels_[static_cast<size_t>(pixel_count_++)];
                pixel.pos = point2<T>{static_cast<T>(np.x), static_cast<T>(np.y)};
                pixel.distance = static_cast<float>(d);
                pixel.coverage = static_cast<float>(T(1) - d);
            }
        };
        
        // Check 4-connected neighbors
        check_neighbor(1, 0);
        check_neighbor(-1, 0);
        check_neighbor(0, 1);
        check_neighbor(0, -1);
        
        // Adaptive step size based on curvature
        // Higher curvature at the ends of major axis
        T curvature = abs(a_ * b_ / pow(a2_ * sin_t * sin_t + b2_ * cos_t * cos_t, T(1.5)));
        dt_ = min(T(0.1), T(1) / (T(1) + curvature));
        t_ += dt_;
    }
    
public:
    /**
     * @brief Construct antialiased ellipse iterator
     */
    aa_ellipse_iterator(point2<T> center, T semi_major, T semi_minor)
        : center_(center), a_(semi_major), b_(semi_minor) {
        
        a2_ = a_ * a_;
        b2_ = b_ * b_;
        t_ = T(0);
        dt_ = T(0.01);  // Initial step
        
        if (a_ <= 0 || b_ <= 0) {
            this->done_ = true;
            return;
        }
        
        generate_pixels();
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    /**
     * @brief Advance to next pixel
     */
    aa_ellipse_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            generate_pixels();
        }
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    aa_ellipse_iterator operator++(int) {
        aa_ellipse_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper to create ellipse iterator
 */
template<typename T>
auto make_ellipse_iterator(point2<T> center, T semi_major, T semi_minor) {
    return ellipse_iterator<T>(center, semi_major, semi_minor);
}

/**
 * @brief Helper to create ellipse arc iterator
 */
template<typename T, typename Angle>
auto make_ellipse_arc_iterator(point2<T> center, T semi_major, T semi_minor,
                               const Angle& start, const Angle& end) {
    return ellipse_iterator<T>(center, semi_major, semi_minor, start, end);
}

/**
 * @brief Helper to create filled ellipse iterator
 */
template<typename T>
auto make_filled_ellipse_iterator(point2<T> center, T semi_major, T semi_minor) {
    return filled_ellipse_iterator<T>(center, semi_major, semi_minor);
}

/**
 * @brief Helper to create antialiased ellipse iterator
 */
template<typename T>
auto make_aa_ellipse_iterator(point2<T> center, T semi_major, T semi_minor) {
    return aa_ellipse_iterator<T>(center, semi_major, semi_minor);
}

} // namespace euler::dda