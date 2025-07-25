#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <cstdint>
#include <limits>
#include <random>
#include <cmath>

namespace euler {

// PCG32 random number generator
// Based on PCG algorithm by Melissa O'Neill (www.pcg-random.org)
class random_generator {
public:
    using result_type = uint32_t;
    
    // Default constructor - seed from random device
    random_generator() : random_generator(std::random_device{}()) {}
    
    // Constructor with seed
    explicit random_generator(uint64_t seed) : seed_(seed) {
        state_ = 0U;
        inc_ = (seed << 1u) | 1u;  // Must be odd
        (*this)();  // Advance once
        state_ += seed;
        (*this)();  // Advance again
    }
    
    // Generate next random number
    result_type operator()() {
        uint64_t oldstate = state_;
        // Advance internal state
        state_ = oldstate * 6364136223846793005ULL + inc_;
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    
    // Min/max values for compatibility
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
    
    // Generate uniform float in [0, 1)
    template<typename T>
    auto uniform() -> std::enable_if_t<std::is_floating_point_v<T>, T> {
        // Generate 32-bit random number and scale to [0, 1)
        return static_cast<T>((*this)()) / static_cast<T>(max() + 1.0);
    }
    
    // Generate uniform integer in full range
    template<typename T>
    auto uniform() -> std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 4, T> {
        if constexpr (std::is_signed_v<T>) {
            return static_cast<T>(static_cast<int32_t>((*this)()));
        } else {
            return static_cast<T>((*this)());
        }
    }
    
    // Generate uniform value in range [min, max]
    template<typename T>
    auto uniform(T min_val, T max_val) -> std::enable_if_t<std::is_floating_point_v<T>, T> {
        EULER_CHECK(min_val <= max_val, error_code::invalid_argument,
                    "random_generator::uniform: min must be <= max");
        return min_val + (max_val - min_val) * uniform<T>();
    }
    
    // Generate uniform integer in range [min, max]
    template<typename T>
    auto uniform(T min_val, T max_val) -> std::enable_if_t<std::is_integral_v<T>, T> {
        EULER_CHECK(min_val <= max_val, error_code::invalid_argument,
                    "random_generator::uniform: min must be <= max");
        
        // Use rejection sampling for unbiased results
        uint64_t range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val) + 1;
        if (range == 0) {
            // Full range of type
            return uniform<T>();
        }
        
        uint64_t max_valid = (static_cast<uint64_t>(max()) + 1) / range * range - 1;
        uint32_t x;
        do {
            x = (*this)();
        } while (x > max_valid);
        
        return static_cast<T>(static_cast<uint64_t>(min_val) + x % range);
    }
    
    // Generate normal/Gaussian distributed value
    template<typename T>
    T normal(T mean = T(0), T stddev = T(1)) {
        static_assert(std::is_floating_point_v<T>, "normal distribution requires floating point type");
        
        // Box-Muller transform with caching
        static thread_local bool has_saved = false;
        static thread_local T saved_value;
        
        if (has_saved) {
            has_saved = false;
            return mean + stddev * saved_value;
        }
        
        // Generate two uniform values in (0, 1]
        T u1, u2;
        do {
            u1 = uniform<T>();
        } while (u1 == T(0));  // Avoid log(0)
        u2 = uniform<T>();
        
        // Box-Muller transform
        T radius = std::sqrt(T(-2) * std::log(u1));
        T theta = T(2) * constants<T>::pi * u2;
        
        T z0 = radius * std::cos(theta);
        T z1 = radius * std::sin(theta);
        
        saved_value = z1;
        has_saved = true;
        
        return mean + stddev * z0;
    }
    
    // Reseed the generator
    void seed(uint64_t s) {
        seed_ = s;
        state_ = 0U;
        inc_ = (s << 1u) | 1u;
        (*this)();
        state_ += s;
        (*this)();
    }
    
    // Get the seed value
    uint64_t get_seed() const { return seed_; }
    
    // Discard n values
    void discard(unsigned long long n) {
        for (unsigned long long i = 0; i < n; ++i) {
            (*this)();
        }
    }
    
private:
    uint64_t state_;  // RNG state
    uint64_t inc_;    // Stream (must be odd)
    uint64_t seed_;   // Original seed
};

// Thread-local default generator
inline random_generator& thread_local_rng() {
    static thread_local random_generator rng;
    return rng;
}

// Convenience functions using thread-local generator
template<typename T>
inline T random_uniform() {
    return thread_local_rng().uniform<T>();
}

template<typename T>
inline T random_uniform(T min_val, T max_val) {
    return thread_local_rng().uniform(min_val, max_val);
}

template<typename T>
inline T random_normal(T mean = T(0), T stddev = T(1)) {
    return thread_local_rng().normal(mean, stddev);
}

} // namespace euler