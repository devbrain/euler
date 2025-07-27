/**
 * @file batched_bezier_iterator.hh
 * @brief Batched Bezier curve rasterization with prefetching
 * @ingroup DDAModule
 */
#pragma once

#include <euler/dda/dda_traits.hh>
#include <euler/dda/bezier_iterator.hh>
#include <euler/dda/pixel_batch.hh>
#include <euler/dda/line_iterator.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/dda/dda_math.hh>
#include <euler/core/simd.hh>
#include <algorithm>
#include <vector>

namespace euler::dda {
    /**
     * @brief Batched cubic Bezier iterator with SIMD and prefetching
     * @tparam T Coordinate type
     *
     * Generates pixels in batches for improved performance.
     * Uses adaptive stepping and SIMD evaluation when available.
     */
    template<typename T>
    class batched_cubic_bezier_iterator {
        public:
            using coord_type = T;
            using point_type = point2 <T>;
            using pixel_type = pixel <int>;
            using batch_type = pixel_batch <pixel_type>;

        private:
            // Control points
            point2 <T> p0_, p1_, p2_, p3_;

            // Parameter state
            T t_;
            T dt_;
            T tolerance_;

            // Batch state
            batch_type current_batch_;
            bool done_;

            // Prefetch distance (in parameter space)
            static constexpr T PREFETCH_T_AHEAD = T(0.1);

            // Precomputed values for SIMD evaluation
            struct simd_context {
                alignas(32) T control_x[8]; // Padded for AVX (256-bit)
                alignas(32) T control_y[8]; // Padded for AVX (256-bit)

                void setup(const point2 <T>& p0, const point2 <T>& p1,
                           const point2 <T>& p2, const point2 <T>& p3) {
                    control_x[0] = p0.x;
                    control_y[0] = p0.y;
                    control_x[1] = p1.x;
                    control_y[1] = p1.y;
                    control_x[2] = p2.x;
                    control_y[2] = p2.y;
                    control_x[3] = p3.x;
                    control_y[3] = p3.y;
                    // Padding for SIMD
                    control_x[4] = control_x[5] = control_x[6] = control_x[7] = T(0);
                    control_y[4] = control_y[5] = control_y[6] = control_y[7] = T(0);
                }
            } simd_ctx_;

            point2 <T> evaluate(T t) const {
#ifdef EULER_HAS_XSIMD
                if constexpr (simd_traits <T>::has_simd && std::is_floating_point_v <T>) {
                    return evaluate_simd(t);
                } else
#endif
                {
                    return evaluate_scalar(t);
                }
            }

            point2 <T> evaluate_scalar(T t) const {
                T t2 = t * t;
                T t3 = t2 * t;
                T one_minus_t = T(1) - t;
                T one_minus_t2 = one_minus_t * one_minus_t;
                T one_minus_t3 = one_minus_t2 * one_minus_t;

                return point2 <T>{
                    one_minus_t3 * p0_.x + T(3) * one_minus_t2 * t * p1_.x +
                    T(3) * one_minus_t * t2 * p2_.x + t3 * p3_.x,
                    one_minus_t3 * p0_.y + T(3) * one_minus_t2 * t * p1_.y +
                    T(3) * one_minus_t * t2 * p2_.y + t3 * p3_.y
                };
            }

#ifdef EULER_HAS_XSIMD
            point2 <T> evaluate_simd(T t) const {
                using batch_t = typename simd_traits <T>::batch_type;

                // Compute basis functions
                T t2 = t * t;
                T t3 = t2 * t;
                T one_minus_t = T(1) - t;
                T one_minus_t2 = one_minus_t * one_minus_t;
                T one_minus_t3 = one_minus_t2 * one_minus_t;

                // Ensure array is large enough for SIMD load (AVX needs 256 bits = 8 floats)
                alignas(32) T basis[8] = {
                    one_minus_t3,
                    T(3) * one_minus_t2 * t,
                    T(3) * one_minus_t * t2,
                    t3,
                    T(0), T(0), T(0), T(0)  // Padding for SIMD
                };

                // Load and compute with SIMD
                batch_t basis_vec = batch_t::load_aligned(basis);
                batch_t x_vec = batch_t::load_aligned(simd_ctx_.control_x);
                batch_t y_vec = batch_t::load_aligned(simd_ctx_.control_y);

                T result_x = xsimd::reduce_add(basis_vec * x_vec);
                T result_y = xsimd::reduce_add(basis_vec * y_vec);

                return point2 <T>{result_x, result_y};
            }
#endif

            point2 <T> derivative(T t) const {
                T t2 = t * t;
                T one_minus_t = T(1) - t;
                T one_minus_t2 = one_minus_t * one_minus_t;

                // These are vectors, not points
                vec2<T> d1 = p1_ - p0_;
                vec2<T> d2 = p2_ - p1_;
                vec2<T> d3 = p3_ - p2_;
                
                // Result is a vector<T, 2>
                vec2<T> deriv = T(3) * (one_minus_t2 * d1 + T(2) * one_minus_t * t * d2 + t2 * d3);
                
                // Convert vector to point
                return point2<T>{deriv[0], deriv[1]};
            }

            T compute_adaptive_step(T t) const {
                // Clamp t to valid range
                t = std::clamp(t, T(0), T(1));
                
                const T h = T(0.001);
                T t_next = std::min(t + h, T(1));
                
                auto d1 = derivative(t);
                auto d1_next = derivative(t_next);
                
                // Compute second derivative approximation
                // d1 and d1_next are points, their difference is a vector
                // Force evaluation to avoid expression template issues
                vec2<T> d2 = (d1_next - d1) * (T(1) / h);

                T speed = length(d1);
                if (speed < tolerance_ * T(0.01)) {
                    return T(0.1);
                }

                // d2 is a vector, access with [0] and [1]
                T cross_z = d1.x * d2[1] - d1.y * d2[0];
                T curvature = abs(cross_z) / (speed * speed * speed);

                T step = tolerance_ / (T(1) + curvature * tolerance_);
                return std::clamp(step, T(0.001), T(0.1));
            }

            void fill_batch() {
                current_batch_.clear();

                if (done_ || t_ >= T(1)) {
                    done_ = true;
                    return;
                }

                // Prefetch control points for future evaluations
                T prefetch_t = std::min(t_ + PREFETCH_T_AHEAD, T(1));
                prefetch_curve_data(prefetch_t);

                // Fill batch with curve pixels
                int iteration_count = 0;
                const int MAX_ITERATIONS = 1000; // Safety limit
                
                while (!current_batch_.is_full() && t_ < T(1) && iteration_count < MAX_ITERATIONS) {
                    iteration_count++;
                    auto pos = evaluate(t_);
                    point2i pixel = round(pos);
                    
                    // Sanity check - skip invalid pixels
                    if (pixel.x < -10000 || pixel.x > 10000 || 
                        pixel.y < -10000 || pixel.y > 10000) {
                        // Skip this pixel and continue
                        t_ += dt_;
                        if (t_ > T(1)) t_ = T(1);
                        dt_ = compute_adaptive_step(t_);
                        continue;
                    }

                    // Check if we need to fill gaps with a line
                    if (!current_batch_.is_empty()) {
                        auto last = current_batch_.pixels[current_batch_.count - 1].pos;
                        if (distance_squared(pixel, last) > 2) {
                            // Fill gap with line pixels
                            fill_line_gap(last, pixel);
                        }
                    }

                    // Only add the pixel if there's room
                    if (!current_batch_.is_full()) {
                        current_batch_.add({pixel});
                        
                        // Advance parameter
                        t_ += dt_;
                        if (t_ > T(1)) t_ = T(1);

                        // Update step size
                        dt_ = compute_adaptive_step(t_);
                    } else {
                        // Batch is full, we'll process this pixel next time
                        break;
                    }
                }

                // Handle final point
                if (t_ >= T(1) && !done_) {
                    auto final_pos = evaluate(T(1));
                    point2i final_pixel = round(final_pos);

                    if (current_batch_.is_empty() ||
                        current_batch_.pixels[current_batch_.count - 1].pos != final_pixel) {
                        current_batch_.add({final_pixel});
                    }
                    done_ = true;
                }
            }

            void fill_line_gap(point2i from, point2i to) {
                // Use line iterator to fill gaps
                line_iterator <int> line_iter(from, to);
                ++line_iter; // Skip first pixel (already in batch)

                while (line_iter != line_iterator <int>::end() && !current_batch_.is_full()) {
                    auto pixel = *line_iter;
                    if (pixel.pos != to) {
                        // Don't duplicate the target pixel
                        current_batch_.add(pixel);
                    }
                    ++line_iter;
                }
            }

            void prefetch_curve_data(T future_t) {
                // Prefetch memory that will be needed for future evaluations
                // TODO: Implement prefetch hints when available
                (void)future_t; // Suppress unused parameter warning
            }

        public:
            /**
             * @brief Construct batched cubic Bezier iterator
             */
            batched_cubic_bezier_iterator(point2 <T> p0, point2 <T> p1,
                                          point2 <T> p2, point2 <T> p3,
                                          T tolerance = default_tolerance <T>())
                : p0_(p0), p1_(p1), p2_(p2), p3_(p3),
                  t_(0), tolerance_(tolerance), done_(false) {
                // Setup SIMD context
                simd_ctx_.setup(p0_, p1_, p2_, p3_);

                // Compute initial step
                dt_ = compute_adaptive_step(0);

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
            batched_cubic_bezier_iterator& next_batch() {
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
                    // Call the callback directly with the current batch
                    callback(current_batch_);
                    next_batch();
                }
            }
    };

    /**
     * @brief Batched general Bezier iterator for arbitrary degree
     * @tparam T Coordinate type
     */
    template<typename T>
    class batched_bezier_iterator {
        public:
            using coord_type = T;
            using point_type = point2 <T>;
            using pixel_type = pixel <int>;
            using batch_type = pixel_batch <pixel_type>;

        private:
            std::vector <point2 <T>> control_points_;
            std::vector <T> binomial_coeffs_;
            int degree_;

            T t_;
            T dt_;
            T tolerance_;

            batch_type current_batch_;
            bool done_;

            // Cache for batch evaluation
            struct eval_cache {
                std::vector <T> powers_t;
                std::vector <T> powers_one_minus_t;

                void resize(int degree) {
                    powers_t.resize(static_cast <size_t>(degree + 1));
                    powers_one_minus_t.resize(static_cast <size_t>(degree + 1));
                }

                void compute(T t, int degree) {
                    T one_minus_t = T(1) - t;

                    powers_t[0] = T(1);
                    powers_one_minus_t[0] = T(1);

                    for (int i = 1; i <= degree; ++i) {
                        powers_t[static_cast <size_t>(i)] =
                            powers_t[static_cast <size_t>(i - 1)] * t;
                        powers_one_minus_t[static_cast <size_t>(i)] =
                            powers_one_minus_t[static_cast <size_t>(i - 1)] * one_minus_t;
                    }
                }
            } cache_;

            void compute_binomial_coefficients() {
                binomial_coeffs_.resize(static_cast <size_t>(degree_ + 1));

                // Use recursive formula: C(n, k+1) = C(n, k) * (n - k) / (k + 1)
                // Start with C(n, 0) = 1
                binomial_coeffs_[0] = T(1);

                for (int k = 0; k < degree_; ++k) {
                    binomial_coeffs_[static_cast <size_t>(k + 1)] =
                        binomial_coeffs_[static_cast <size_t>(k)] *
                        static_cast <T>(degree_ - k) / static_cast <T>(k + 1);
                }
            }

            static T binomial_coefficient(int n, int k) {
                // Use memoization for small values (common case)
                constexpr int MAX_MEMOIZED = 20;
                static std::array <std::array <int, MAX_MEMOIZED + 1>, MAX_MEMOIZED + 1> memo = []() {
                    std::array <std::array <int, MAX_MEMOIZED + 1>, MAX_MEMOIZED + 1> table{};
                    // Initialize with -1 to indicate uncomputed
                    for (auto& row : table) {
                        row.fill(-1);
                    }
                    // Base cases
                    for (int i = 0; i <= MAX_MEMOIZED; ++i) {
                        table[static_cast <size_t>(i)][0] = 1;
                        table[static_cast <size_t>(i)][static_cast <size_t>(i)] = 1;
                    }
                    return table;
                }();

                if (n <= MAX_MEMOIZED && k <= MAX_MEMOIZED) {
                    if (k > n) return T(0);
                    if (k > n - k) k = n - k;

                    // Check if already computed
                    if (memo[static_cast <size_t>(n)][static_cast <size_t>(k)] != -1) {
                        return static_cast <T>(memo[static_cast <size_t>(n)][static_cast <size_t>(k)]);
                    }

                    // Compute using Pascal's triangle
                    int result = 1;
                    for (int i = 0; i < k; ++i) {
                        result *= (n - i);
                        result /= (i + 1);
                    }
                    memo[static_cast <size_t>(n)][static_cast <size_t>(k)] = result;
                    return static_cast <T>(result);
                }

                // Fall back to direct computation for large values
                if (k > n - k) k = n - k;
                T result = 1;
                for (int i = 0; i < k; ++i) {
                    result *= static_cast <T>(n - i);
                    result /= static_cast <T>(i + 1);
                }
                return result;
            }

            point2 <T> evaluate_batch(T t) {
                cache_.compute(t, degree_);

                point2 <T> result{0, 0};

                // Prefetch control points for next evaluation
                if (degree_ > 8) {
                    prefetch_hint::prefetch_range <point2 <T>, 0, 3>(
                        control_points_.data(),
                        control_points_.data() + control_points_.size());
                }

                for (int i = 0; i <= degree_; ++i) {
                    T basis = binomial_coeffs_[static_cast <size_t>(i)] *
                              cache_.powers_one_minus_t[static_cast <size_t>(degree_ - i)] *
                              cache_.powers_t[static_cast <size_t>(i)];

                    result = result + vec2 <T>(
                                 basis * control_points_[static_cast <size_t>(i)].x,
                                 basis * control_points_[static_cast <size_t>(i)].y);
                }

                return result;
            }

            void fill_batch() {
                current_batch_.clear();

                if (done_ || t_ >= T(1)) {
                    done_ = true;
                    return;
                }

                // Process multiple t values in one batch
                constexpr int T_SAMPLES_PER_BATCH = 4;
                std::array <T, T_SAMPLES_PER_BATCH> t_values;
                std::array <point2 <T>, T_SAMPLES_PER_BATCH> positions;

                while (!current_batch_.is_full() && t_ < T(1)) {
                    // Evaluate multiple points at once
                    int samples = 0;
                    for (int i = 0; i < T_SAMPLES_PER_BATCH && t_ < T(1); ++i) {
                        t_values[static_cast <size_t>(i)] = t_;
                        positions[static_cast <size_t>(i)] = evaluate_batch(t_);
                        samples++;

                        t_ += dt_;
                        if (t_ > T(1)) t_ = T(1);
                    }

                    // Convert to pixels and add to batch
                    for (int i = 0; i < samples; ++i) {
                        point2i pixel = round(positions[static_cast <size_t>(i)]);

                        // Fill gaps if needed
                        if (!current_batch_.is_empty()) {
                            auto last = current_batch_.pixels[current_batch_.count - 1].pos;
                            if (distance_squared(pixel, last) > 2) {
                                fill_line_gap(last, pixel);
                            }
                        }

                        current_batch_.add({pixel});
                    }
                }

                if (t_ >= T(1)) {
                    done_ = true;
                }
            }

            void fill_line_gap(point2i from, point2i to) {
                line_iterator <int> line_iter(from, to);
                ++line_iter; // Skip first pixel

                while (line_iter != line_iterator <int>::end() && !current_batch_.is_full()) {
                    auto pixel = *line_iter;
                    if (pixel.pos != to) {
                        current_batch_.add(pixel);
                    }
                    ++line_iter;
                }
            }

        public:
            /**
             * @brief Construct batched general Bezier iterator
             */
            batched_bezier_iterator(const std::vector <point2 <T>>& control_points,
                                    T tolerance = default_tolerance <T>())
                : control_points_(control_points),
                  degree_(static_cast <int>(control_points.size()) - 1),
                  t_(0), dt_(T(0.01)), tolerance_(tolerance), done_(false) {
                if (control_points_.empty()) {
                    done_ = true;
                    return;
                }

                compute_binomial_coefficients();
                cache_.resize(degree_);

                fill_batch();
            }

            const batch_type& current_batch() const { return current_batch_; }
            bool at_end() const { return done_ && current_batch_.is_empty(); }

            batched_bezier_iterator& next_batch() {
                if (!done_) {
                    fill_batch();
                } else {
                    // Clear the batch when we're done to ensure at_end() works correctly
                    current_batch_.clear();
                }
                return *this;
            }
    };

    /**
     * @brief Helper functions
     */
    template<typename T>
    auto make_batched_cubic_bezier(point2 <T> p0, point2 <T> p1,
                                   point2 <T> p2, point2 <T> p3,
                                   T tolerance = default_tolerance <T>()) {
        return batched_cubic_bezier_iterator <T>(p0, p1, p2, p3, tolerance);
    }

    template<typename T>
    auto make_batched_bezier(const std::vector <point2 <T>>& control_points,
                             T tolerance = default_tolerance <T>()) {
        return batched_bezier_iterator <T>(control_points, tolerance);
    }
} // namespace euler::dda
