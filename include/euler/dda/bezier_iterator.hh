/**
 * @file bezier_iterator.hh
 * @brief Bezier curve rasterization with adaptive stepping
 * @ingroup DDAModule
 * 
 * This header provides comprehensive Bezier curve rasterization support
 * including quadratic, cubic, and general degree Bezier curves. The
 * implementation uses adaptive stepping based on local curvature for
 * optimal quality and performance.
 * 
 * @section Features
 * - Quadratic Bezier curves (3 control points)
 * - Cubic Bezier curves (4 control points)
 * - General Bezier curves (arbitrary control points)
 * - SIMD acceleration for curve evaluation
 * - Adaptive stepping based on curvature
 * - Memoized binomial coefficients for performance
 * 
 * @section Algorithm
 * The iterators use De Casteljau's algorithm with optimizations:
 * - Precomputed binomial coefficients
 * - SIMD evaluation for cubic curves
 * - Adaptive parameter stepping based on local curvature
 * - Line segment fallback for nearly straight sections
 * 
 * @section Usage
 * @code
 * // Cubic Bezier curve
 * auto cubic = make_cubic_bezier_iterator(p0, p1, p2, p3);
 * for (auto pixel : cubic) {
 *     draw_pixel(pixel.pos);
 * }
 * 
 * // General Bezier with 5 control points
 * std::vector<point2f> points = {p0, p1, p2, p3, p4};
 * auto bezier = make_bezier(points, 0.5f); // tolerance = 0.5
 * @endcode
 * 
 * @see batched_bezier_iterator.hh for batched processing
 * @see aa_bezier_iterator.hh for antialiased curves
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/compiler.hh>
#include <euler/core/simd.hh>
#include <euler/dda/aa_simd.hh>
#include <algorithm>
#include <vector>
#include <array>

namespace euler::dda {

/**
 * @brief Quadratic Bezier iterator
 * @tparam T Coordinate type
 * 
 * Rasterizes quadratic Bezier curves using adaptive subdivision.
 */
template<typename T>
class quadratic_bezier_iterator : public dda_iterator_base<quadratic_bezier_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<quadratic_bezier_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
    point2<T> evaluate(T t) const {
#ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd && std::is_floating_point_v<T>) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            T t2 = t * t;
            T one_minus_t = T(1) - t;
            T one_minus_t2 = one_minus_t * one_minus_t;
            
            // Create coefficient arrays
            alignas(simd_alignment_v<T>::value) T coeffs_data[simd_size];
            alignas(simd_alignment_v<T>::value) T x_data[simd_size];
            alignas(simd_alignment_v<T>::value) T y_data[simd_size];
            
            // Initialize with coefficients
            coeffs_data[0] = one_minus_t2;
            coeffs_data[1] = T(2) * one_minus_t * t;
            coeffs_data[2] = t2;
            x_data[0] = p0_.x;
            x_data[1] = p1_.x;
            x_data[2] = p2_.x;
            y_data[0] = p0_.y;
            y_data[1] = p1_.y;
            y_data[2] = p2_.y;
            
            // Fill remaining elements with zero
            for (size_t j = 3; j < simd_size; ++j) {
                coeffs_data[j] = T(0);
                x_data[j] = T(0);
                y_data[j] = T(0);
            }
            
            // Load into SIMD registers
            batch_t coeffs = batch_t::load_aligned(coeffs_data);
            batch_t x_vals = batch_t::load_aligned(x_data);
            batch_t y_vals = batch_t::load_aligned(y_data);
            
            // Compute dot products using SIMD
            T result_x = xsimd::reduce_add(coeffs * x_vals);
            T result_y = xsimd::reduce_add(coeffs * y_vals);
            
            return point2<T>{result_x, result_y};
        } else
#endif
        {
            T t2 = t * t;
            T one_minus_t = T(1) - t;
            T one_minus_t2 = one_minus_t * one_minus_t;
            
            return point2<T>{
                one_minus_t2 * p0_.x + T(2) * one_minus_t * t * p1_.x + t2 * p2_.x,
                one_minus_t2 * p0_.y + T(2) * one_minus_t * t * p1_.y + t2 * p2_.y
            };
        }
    }
    
    point2<T> derivative(T t) const {
        auto d1 = p1_ - p0_;  // This is a vector
        auto d2 = p2_ - p1_;  // This is a vector
        auto deriv = T(2) * ((T(1) - t) * d1 + t * d2);  // This is a vector
        return point2<T>{deriv[0], deriv[1]};
    }
    
    T compute_adaptive_step(T t) const {
        // Estimate curvature at current point
        const T h = T(0.001);
        auto d1 = derivative(t);
        auto d2 = (derivative(t + h) - d1) / h;
        
        T speed = length(d1);
        if (speed < tolerance_ * T(0.01)) {
            return T(0.1);  // Nearly stationary
        }
        
        // Curvature = |v × a| / |v|³
        T cross_z = d1.x * d2[1] - d1.y * d2[0];
        T curvature = abs(cross_z) / (speed * speed * speed);
        
        // Step size inversely proportional to curvature
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.001), T(0.1));
    }
    
private:
    // Control points
    point2<T> p0_, p1_, p2_;
    
    // Current parameter and step
    T t_;
    T dt_;
    T tolerance_;
    
    // Current pixel state
    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;

public:
    /**
     * @brief Construct quadratic Bezier iterator
     * @param p0 Start point
     * @param p1 Control point
     * @param p2 End point
     * @param tolerance Maximum pixel deviation
     */
    quadratic_bezier_iterator(point2<T> p0, point2<T> p1, point2<T> p2,
                             T tolerance = default_tolerance<T>())
        : p0_(p0), p1_(p1), p2_(p2), t_(0), tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        auto start = evaluate(0);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step(0);
    }
    
    /**
     * @brief Get current pixel
     */
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return {last_pixel_};
    }
    
    /**
     * @brief Advance to next pixel
     */
    quadratic_bezier_iterator& operator++() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                return *this;
            }
            using_line_ = false;
            // Don't continue to next bezier point - we already emitted it as the end of the line
            return *this;
        }
        
        if (t_ >= T(1)) {
            this->done_ = true;
            return *this;
        }
        
        // Compute next point
        t_ += dt_;
        if (t_ > T(1)) t_ = T(1);
        
        auto next_point = evaluate(t_);
        point2i next_pixel = round(next_point);
        
        // If pixels are not adjacent, use line iterator
        if (distance_squared(next_pixel, last_pixel_) > 1) {
            line_iter_ = line_iterator<int>(last_pixel_, next_pixel);
            ++line_iter_;  // Skip first pixel (already emitted)
            if (line_iter_ != line_iterator<int>::end()) {
                using_line_ = true;
                // Update last_pixel_ to the target of the line so we continue from there
                last_pixel_ = next_pixel;
                return *this;
            }
        }
        
        last_pixel_ = next_pixel;
        dt_ = compute_adaptive_step(t_);
        
        return *this;
    }
    
    /**
     * @brief Post-increment
     */
    quadratic_bezier_iterator operator++(int) {
        quadratic_bezier_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Cubic Bezier iterator
 * @tparam T Coordinate type
 * 
 * Rasterizes cubic Bezier curves using adaptive subdivision.
 */
template<typename T>
class cubic_bezier_iterator : public dda_iterator_base<cubic_bezier_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<cubic_bezier_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
    EULER_HOT EULER_ALWAYS_INLINE
    point2<T> evaluate(T t) const {
#ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd && std::is_floating_point_v<T>) {
            using batch_t = typename simd_traits<T>::batch_type;
            constexpr size_t simd_size = simd_traits<T>::batch_size;
            
            T t2 = t * t;
            T t3 = t2 * t;
            T one_minus_t = T(1) - t;
            T one_minus_t2 = one_minus_t * one_minus_t;
            T one_minus_t3 = one_minus_t2 * one_minus_t;
            
            // Create coefficient arrays for all 4 control points
            alignas(simd_alignment_v<T>::value) T coeffs_data[simd_size];
            alignas(simd_alignment_v<T>::value) T x_data[simd_size];
            alignas(simd_alignment_v<T>::value) T y_data[simd_size];
            
            // Initialize with coefficients
            coeffs_data[0] = one_minus_t3;
            coeffs_data[1] = T(3) * one_minus_t2 * t;
            coeffs_data[2] = T(3) * one_minus_t * t2;
            coeffs_data[3] = t3;
            x_data[0] = p0_.x;
            x_data[1] = p1_.x;
            x_data[2] = p2_.x;
            x_data[3] = p3_.x;
            y_data[0] = p0_.y;
            y_data[1] = p1_.y;
            y_data[2] = p2_.y;
            y_data[3] = p3_.y;
            
            // Fill remaining elements with zero if needed
            for (size_t j = 4; j < simd_size; ++j) {
                coeffs_data[j] = T(0);
                x_data[j] = T(0);
                y_data[j] = T(0);
            }
            
            // Load into SIMD registers
            batch_t coeffs = batch_t::load_aligned(coeffs_data);
            batch_t x_vals = batch_t::load_aligned(x_data);
            batch_t y_vals = batch_t::load_aligned(y_data);
            
            // Compute dot products using SIMD
            T result_x = xsimd::reduce_add(coeffs * x_vals);
            T result_y = xsimd::reduce_add(coeffs * y_vals);
            
            return point2<T>{result_x, result_y};
        } else
#endif
        {
            T t2 = t * t;
            T t3 = t2 * t;
            T one_minus_t = T(1) - t;
            T one_minus_t2 = one_minus_t * one_minus_t;
            T one_minus_t3 = one_minus_t2 * one_minus_t;
            
            return point2<T>{
                one_minus_t3 * p0_.x + T(3) * one_minus_t2 * t * p1_.x + T(3) * one_minus_t * t2 * p2_.x + t3 * p3_.x,
                one_minus_t3 * p0_.y + T(3) * one_minus_t2 * t * p1_.y + T(3) * one_minus_t * t2 * p2_.y + t3 * p3_.y
            };
        }
    }
    
private:
    // Control points
    point2<T> p0_, p1_, p2_, p3_;
    
    // Current state
    T t_;
    T dt_;
    T tolerance_;
    
    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;

public:
    point2<T> derivative(T t) const {
        T t2 = t * t;
        T one_minus_t = T(1) - t;
        T one_minus_t2 = one_minus_t * one_minus_t;
        
        auto d1 = p1_ - p0_;
        auto d2 = p2_ - p1_;
        auto d3 = p3_ - p2_;
        auto deriv = T(3) * (one_minus_t2 * d1 + T(2) * one_minus_t * t * d2 + t2 * d3);
        return point2<T>{deriv[0], deriv[1]};
    }
    
    T compute_adaptive_step(T t) const {
        const T h = T(0.001);
        auto d1 = derivative(t);
        // Force evaluation to avoid temporary expression issues
        vec2<T> d2 = (derivative(t + h) - d1) / h;
        
        T speed = length(d1);
        if (speed < tolerance_ * T(0.01)) {
            return T(0.1);
        }
        
        T cross_z = d1.x * d2[1] - d1.y * d2[0];
        T curvature = abs(cross_z) / (speed * speed * speed);
        
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.001), T(0.1));
    }
    
public:
    /**
     * @brief Construct cubic Bezier iterator
     */
    cubic_bezier_iterator(point2<T> p0, point2<T> p1, point2<T> p2, point2<T> p3,
                         T tolerance = default_tolerance<T>())
        : p0_(p0), p1_(p1), p2_(p2), p3_(p3), t_(0), tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        auto start = evaluate(0);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step(0);
    }
    
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return {last_pixel_};
    }
    
    cubic_bezier_iterator& operator++() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                return *this;
            }
            using_line_ = false;
            // Don't continue to next bezier point - we already emitted it as the end of the line
            return *this;
        }
        
        if (t_ >= T(1)) {
            this->done_ = true;
            return *this;
        }
        
        t_ += dt_;
        if (t_ > T(1)) t_ = T(1);
        
        auto next_point = evaluate(t_);
        point2i next_pixel = round(next_point);
        
        if (distance_squared(next_pixel, last_pixel_) > 1) {
            line_iter_ = line_iterator<int>(last_pixel_, next_pixel);
            // Don't skip the first pixel - we want to include last_pixel_
            if (line_iter_ != line_iterator<int>::end()) {
                using_line_ = true;
                // Update last_pixel_ to the target of the line so we continue from there
                last_pixel_ = next_pixel;
                return *this;
            }
        }
        
        last_pixel_ = next_pixel;
        dt_ = compute_adaptive_step(t_);
        
        return *this;
    }
    
    cubic_bezier_iterator operator++(int) {
        cubic_bezier_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief General Bezier iterator for arbitrary degree
 * @tparam T Coordinate type
 */
template<typename T>
class bezier_iterator : public dda_iterator_base<bezier_iterator<T>, pixel<int>, T> {
    using base = dda_iterator_base<bezier_iterator<T>, pixel<int>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    std::vector<point2<T>> control_points_;
    std::vector<T> binomial_coeffs_;
    int degree_;
    
    T t_;
    T dt_;
    T tolerance_;
    
    point2i last_pixel_;
    line_iterator<int> line_iter_;
    bool using_line_;
    
    void compute_binomial_coefficients() {
        binomial_coeffs_.resize(static_cast<size_t>(degree_ + 1));
        
        // Use recursive formula: C(n, k+1) = C(n, k) * (n - k) / (k + 1)
        // Start with C(n, 0) = 1
        binomial_coeffs_[0] = T(1);
        
        for (int k = 0; k < degree_; ++k) {
            binomial_coeffs_[static_cast<size_t>(k + 1)] = 
                binomial_coeffs_[static_cast<size_t>(k)] * 
                static_cast<T>(degree_ - k) / static_cast<T>(k + 1);
        }
    }
    
    static T binomial_coefficient(int n, int k) {
        // Use memoization for small values (common case)
        constexpr int MAX_MEMOIZED = 20;
        static std::array<std::array<int, MAX_MEMOIZED + 1>, MAX_MEMOIZED + 1> memo = []() {
            std::array<std::array<int, MAX_MEMOIZED + 1>, MAX_MEMOIZED + 1> table{};
            // Initialize with -1 to indicate uncomputed
            for (auto& row : table) {
                row.fill(-1);
            }
            // Base cases
            for (int i = 0; i <= MAX_MEMOIZED; ++i) {
                table[static_cast<size_t>(i)][0] = 1;
                table[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1;
            }
            return table;
        }();
        
        if (n <= MAX_MEMOIZED && k <= MAX_MEMOIZED) {
            if (k > n) return T(0);
            if (k > n - k) k = n - k;
            
            // Check if already computed
            if (memo[static_cast<size_t>(n)][static_cast<size_t>(k)] != -1) {
                return static_cast<T>(memo[static_cast<size_t>(n)][static_cast<size_t>(k)]);
            }
            
            // Compute using Pascal's triangle
            int result = 1;
            for (int i = 0; i < k; ++i) {
                result *= (n - i);
                result /= (i + 1);
            }
            memo[static_cast<size_t>(n)][static_cast<size_t>(k)] = result;
            return static_cast<T>(result);
        }
        
        // Fall back to direct computation for large values
        if (k > n - k) k = n - k;
        T result = 1;
        for (int i = 0; i < k; ++i) {
            result *= static_cast<T>(n - i);
            result /= static_cast<T>(i + 1);
        }
        return result;
    }
    
    point2<T> evaluate(T t) const {
#ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd && std::is_floating_point_v<T>) {
            return evaluate_simd(t);
        } else {
            return evaluate_scalar(t);
        }
#else
        return evaluate_scalar(t);
#endif
    }
    
private:
    EULER_HOT
    point2<T> evaluate_scalar(T t) const {
        point2<T> result{0, 0};
        T one_minus_t = T(1) - t;
        
        // Precompute powers to avoid repeated calls
        std::vector<T> powers_t(static_cast<size_t>(degree_ + 1));
        std::vector<T> powers_one_minus_t(static_cast<size_t>(degree_ + 1));
        
        powers_t[0] = T(1);
        powers_one_minus_t[0] = T(1);
        
        for (int i = 1; i < degree_ + 1; ++i) {
            powers_t[static_cast<size_t>(i)] = powers_t[static_cast<size_t>(i-1)] * t;
            powers_one_minus_t[static_cast<size_t>(i)] = powers_one_minus_t[static_cast<size_t>(i-1)] * one_minus_t;
        }
        
        EULER_LOOP_VECTORIZE
        for (int i = 0; i < degree_ + 1; ++i) {
            T basis = binomial_coeffs_[static_cast<size_t>(i)] * 
                     powers_one_minus_t[static_cast<size_t>(degree_ - i)] * 
                     powers_t[static_cast<size_t>(i)];
            result.x += basis * control_points_[static_cast<size_t>(i)].x;
            result.y += basis * control_points_[static_cast<size_t>(i)].y;
        }
        
        return result;
    }
    
#ifdef EULER_HAS_XSIMD
    EULER_DISABLE_WARNING_PUSH
    EULER_DISABLE_WARNING_STRICT_OVERFLOW
    point2<T> evaluate_simd(T t) const {
        using batch_t = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = simd_traits<T>::batch_size;
        
        T one_minus_t = T(1) - t;
        
        // Precompute powers to avoid repeated calculations
        std::vector<T> powers_t(static_cast<size_t>(degree_ + 1));
        std::vector<T> powers_one_minus_t(static_cast<size_t>(degree_ + 1));
        
        powers_t[0] = T(1);
        powers_one_minus_t[0] = T(1);
        
        for (int i = 1; i < degree_ + 1; ++i) {
            powers_t[static_cast<size_t>(i)] = powers_t[static_cast<size_t>(i-1)] * t;
            powers_one_minus_t[static_cast<size_t>(i)] = powers_one_minus_t[static_cast<size_t>(i-1)] * one_minus_t;
        }
        
        T result_x = 0;
        T result_y = 0;
        
        // Process control points in SIMD batches when possible
        int i = 0;
        if (degree_ + 1 >= static_cast<int>(simd_size)) {
            alignas(simd_alignment_v<T>::value) T basis_data[simd_size];
            alignas(simd_alignment_v<T>::value) T x_data[simd_size];
            alignas(simd_alignment_v<T>::value) T y_data[simd_size];
            
            const int total_points = degree_ + 1;
            const int simd_step = static_cast<int>(simd_size);
            for (; i <= total_points - simd_step; i += simd_step) {
                // Compute basis functions for this batch
                for (size_t j = 0; j < simd_size; ++j) {
                    int idx = i + static_cast<int>(j);
                    T basis = binomial_coeffs_[static_cast<size_t>(idx)] * 
                             powers_one_minus_t[static_cast<size_t>(degree_ - idx)] * 
                             powers_t[static_cast<size_t>(idx)];
                    basis_data[j] = basis;
                    x_data[j] = control_points_[static_cast<size_t>(idx)].x;
                    y_data[j] = control_points_[static_cast<size_t>(idx)].y;
                }
                
                // Load into SIMD registers
                batch_t basis_batch = batch_t::load_aligned(basis_data);
                batch_t x_batch = batch_t::load_aligned(x_data);
                batch_t y_batch = batch_t::load_aligned(y_data);
                
                // Accumulate using SIMD
                result_x += xsimd::reduce_add(basis_batch * x_batch);
                result_y += xsimd::reduce_add(basis_batch * y_batch);
            }
        }
        
        // Handle remaining control points
        const int remaining = degree_ + 1 - i;
        for (int j = 0; j < remaining; ++j) {
            const int idx = i + j;
            const int power_idx = degree_ - idx;
            T basis = binomial_coeffs_[static_cast<size_t>(idx)] * 
                     powers_one_minus_t[static_cast<size_t>(power_idx)] * 
                     powers_t[static_cast<size_t>(idx)];
            result_x += basis * control_points_[static_cast<size_t>(idx)].x;
            result_y += basis * control_points_[static_cast<size_t>(idx)].y;
        }
        
        return point2<T>{result_x, result_y};
    }
    EULER_DISABLE_WARNING_POP
#endif
    
public:
    
    point2<T> derivative(T t) const {
        if (degree_ == 0) return {0, 0};
        
        point2<T> result{0, 0};
        T one_minus_t = T(1) - t;
        
        for (int i = 0; i < degree_; ++i) {
            T basis = binomial_coeffs_[static_cast<size_t>(i)] * static_cast<T>(degree_) * 
                     static_cast<T>(pow(static_cast<double>(one_minus_t), degree_ - i - 1)) * 
                     static_cast<T>(pow(static_cast<double>(t), i));
            auto diff = control_points_[static_cast<size_t>(i + 1)] - control_points_[static_cast<size_t>(i)];
            result = result + vec2<T>(basis * diff[0], basis * diff[1]);
        }
        
        return result;
    }
    
    T compute_adaptive_step(T t) const {
        const T h = T(0.001);
        auto d1 = derivative(t);
        // Force evaluation to avoid temporary expression issues
        vec2<T> d2 = (derivative(t + h) - d1) / h;
        
        T speed = length(d1);
        if (speed < tolerance_ * T(0.01)) {
            return T(0.1);
        }
        
        T cross_z = d1.x * d2[1] - d1.y * d2[0];
        T curvature = abs(cross_z) / (speed * speed * speed);
        
        T step = tolerance_ / (T(1) + curvature * tolerance_);
        return std::clamp(step, T(0.001), T(0.1));
    }
    
public:
    /**
     * @brief Construct general Bezier iterator
     */
    bezier_iterator(const std::vector<point2<T>>& control_points,
                   T tolerance = default_tolerance<T>())
        : control_points_(control_points), 
          degree_(static_cast<int>(control_points.size()) - 1),
          t_(0), tolerance_(tolerance),
          line_iter_(point2i{0,0}, point2i{0,0}), using_line_(false) {
        
        if (control_points_.empty()) {
            this->done_ = true;
            return;
        }
        
        compute_binomial_coefficients();
        auto start = evaluate(0);
        last_pixel_ = round(start);
        dt_ = compute_adaptive_step(0);
    }
    
    value_type operator*() const {
        if (using_line_) {
            return *line_iter_;
        }
        return {last_pixel_};
    }
    
    bezier_iterator& operator++() {
        if (using_line_) {
            ++line_iter_;
            if (line_iter_ != line_iterator<int>::end()) {
                return *this;
            }
            using_line_ = false;
            // Don't continue to next bezier point - we already emitted it as the end of the line
            return *this;
        }
        
        if (t_ >= T(1)) {
            this->done_ = true;
            return *this;
        }
        
        t_ += dt_;
        if (t_ > T(1)) t_ = T(1);
        
        auto next_point = evaluate(t_);
        point2i next_pixel = round(next_point);
        
        if (distance_squared(next_pixel, last_pixel_) > 1) {
            line_iter_ = line_iterator<int>(last_pixel_, next_pixel);
            // Don't skip the first pixel - we want to include last_pixel_
            if (line_iter_ != line_iterator<int>::end()) {
                using_line_ = true;
                // Update last_pixel_ to the target of the line so we continue from there
                last_pixel_ = next_pixel;
                return *this;
            }
        }
        
        last_pixel_ = next_pixel;
        dt_ = compute_adaptive_step(t_);
        
        return *this;
    }
    
    bezier_iterator operator++(int) {
        bezier_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Antialiased cubic Bezier iterator
 */
template<typename T>
class aa_cubic_bezier_iterator : public dda_iterator_base<aa_cubic_bezier_iterator<T>, aa_pixel<T>, T> {
    static_assert(std::is_floating_point_v<T>,
                  "Antialiased Bezier requires floating point");
    
    using base = dda_iterator_base<aa_cubic_bezier_iterator<T>, aa_pixel<T>, T>;
    
public:
    using typename base::value_type;
    using typename base::coord_type;
    using typename base::point_type;
    
private:
    cubic_bezier_iterator<T> bezier_iter_;
    std::array<value_type, 4> pixels_;
    int pixel_count_;
    int pixel_index_;
    
    // Compute signed distance to curve (approximation)
    T curve_distance(point2<T> p, T& closest_t) const {
        // Binary search for closest point on curve
        // T t_min = 0, t_max = 1; // Not used in current implementation
        T min_dist = std::numeric_limits<T>::max();
        
        // Initial samples
        const int samples = 10;
        for (int i = 0; i <= samples; ++i) {
            T t = static_cast<T>(i) / static_cast<T>(samples);
            auto curve_p = bezier_iter_.evaluate(t);
            T d = distance_squared(p, curve_p);
            if (d < min_dist) {
                min_dist = d;
                closest_t = t;
            }
        }
        
        // Refine with binary search
        for (int iter = 0; iter < 5; ++iter) {
            T t1 = closest_t - T(0.1) / (1 << iter);
            T t2 = closest_t + T(0.1) / (1 << iter);
            t1 = std::clamp(t1, T(0), T(1));
            t2 = std::clamp(t2, T(0), T(1));
            
            auto p1 = bezier_iter_.evaluate(t1);
            auto p2 = bezier_iter_.evaluate(t2);
            T d1 = distance_squared(p, p1);
            T d2 = distance_squared(p, p2);
            
            if (d1 < min_dist) {
                min_dist = d1;
                closest_t = t1;
            }
            if (d2 < min_dist) {
                min_dist = d2;
                closest_t = t2;
            }
        }
        
        return sqrt(min_dist);
    }
    
public:
    aa_cubic_bezier_iterator(point2<T> p0, point2<T> p1, point2<T> p2, point2<T> p3,
                            T tolerance = default_tolerance<T>())
        : bezier_iter_(p0, p1, p2, p3, tolerance) {
        generate_pixels();
    }
    
    void generate_pixels() {
        pixel_count_ = 0;
        pixel_index_ = 0;
        
        if (bezier_iter_ == cubic_bezier_iterator<T>::end()) {
            this->done_ = true;
            return;
        }
        
        auto main_pixel = *bezier_iter_;
        ++bezier_iter_;
        
        // Add main pixel with full coverage
        auto& p0 = pixels_[static_cast<size_t>(pixel_count_++)];
        p0.pos = point2<T>{static_cast<T>(main_pixel.pos.x), static_cast<T>(main_pixel.pos.y)};
        p0.coverage = 1.0f;
        p0.distance = 0.0f;
        
#ifdef EULER_HAS_XSIMD
        if constexpr (simd_traits<T>::has_simd) {
            // Use SIMD to check multiple neighbors at once
            point2<T> neighbor_centers[4] = {
                {static_cast<T>(main_pixel.pos.x + 1) + T(0.5), static_cast<T>(main_pixel.pos.y) + T(0.5)},
                {static_cast<T>(main_pixel.pos.x - 1) + T(0.5), static_cast<T>(main_pixel.pos.y) + T(0.5)},
                {static_cast<T>(main_pixel.pos.x) + T(0.5), static_cast<T>(main_pixel.pos.y + 1) + T(0.5)},
                {static_cast<T>(main_pixel.pos.x) + T(0.5), static_cast<T>(main_pixel.pos.y - 1) + T(0.5)}
            };
            
            T distances[4];
            T closest_ts[4];
            
            // Compute distances for all 4 neighbors
            for (int i = 0; i < 4; ++i) {
                distances[i] = curve_distance(neighbor_centers[i], closest_ts[i]);
            }
            
            // Add pixels with coverage
            for (int i = 0; i < 4 && pixel_count_ < 4; ++i) {
                if (distances[i] < T(1)) {
                    auto& pixel = pixels_[static_cast<size_t>(pixel_count_++)];
                    pixel.pos = point2<T>{
                        floor(neighbor_centers[i].x - T(0.5)),
                        floor(neighbor_centers[i].y - T(0.5))
                    };
                    pixel.distance = static_cast<float>(distances[i]);
                    pixel.coverage = static_cast<float>(T(1) - distances[i]);
                }
            }
        } else
#endif
        {
            // Check neighbors for AA
            auto check_neighbor = [&](int dx, int dy) {
                if (pixel_count_ >= 4) return;
                
                point2i np{main_pixel.pos.x + dx, main_pixel.pos.y + dy};
                point2<T> p{static_cast<T>(np.x) + T(0.5), static_cast<T>(np.y) + T(0.5)};
                
                T closest_t;
                T d = curve_distance(p, closest_t);
                
                if (d < T(1)) {
                    auto& pixel = pixels_[static_cast<size_t>(pixel_count_++)];
                    pixel.pos = point2<T>{static_cast<T>(np.x), static_cast<T>(np.y)};
                    pixel.distance = static_cast<float>(d);
                    pixel.coverage = static_cast<float>(T(1) - d);
                }
            };
            
            check_neighbor(1, 0);
            check_neighbor(-1, 0);
            check_neighbor(0, 1);
            check_neighbor(0, -1);
        }
    }
    
    value_type operator*() const {
        return pixels_[static_cast<size_t>(pixel_index_)];
    }
    
    aa_cubic_bezier_iterator& operator++() {
        pixel_index_++;
        if (pixel_index_ >= pixel_count_) {
            generate_pixels();
        }
        return *this;
    }
    
    aa_cubic_bezier_iterator operator++(int) {
        aa_cubic_bezier_iterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    static constexpr dda_sentinel end() { return {}; }
};

/**
 * @brief Helper functions
 */
template<typename T>
auto make_quadratic_bezier(point2<T> p0, point2<T> p1, point2<T> p2,
                          T tolerance = default_tolerance<T>()) {
    return quadratic_bezier_iterator<T>(p0, p1, p2, tolerance);
}

template<typename T>
auto make_cubic_bezier(point2<T> p0, point2<T> p1, point2<T> p2, point2<T> p3,
                      T tolerance = default_tolerance<T>()) {
    return cubic_bezier_iterator<T>(p0, p1, p2, p3, tolerance);
}

template<typename T>
auto make_bezier(const std::vector<point2<T>>& control_points,
                T tolerance = default_tolerance<T>()) {
    return bezier_iterator<T>(control_points, tolerance);
}

template<typename T>
auto make_aa_cubic_bezier(point2<T> p0, point2<T> p1, point2<T> p2, point2<T> p3,
                         T tolerance = default_tolerance<T>()) {
    return aa_cubic_bezier_iterator<T>(p0, p1, p2, p3, tolerance);
}

} // namespace euler::dda