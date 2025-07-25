#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <cmath>
#include <type_traits>

namespace euler {

// ============================================================================
// Uniform Distribution
// ============================================================================

template<typename T>
class uniform_distribution {
public:
    static_assert(std::is_arithmetic_v<T>, "uniform_distribution requires arithmetic type");
    
    uniform_distribution(T min_val = T(0), T max_val = T(1)) 
        : min_(min_val), max_(max_val) {
        EULER_CHECK(min_val <= max_val, error_code::invalid_argument,
                    "uniform_distribution: min must be <= max");
    }
    
    template<typename Generator>
    T operator()(Generator& g) {
        return g.uniform(min_, max_);
    }
    
    T min() const { return min_; }
    T max() const { return max_; }
    
    void param(T min_val, T max_val) {
        EULER_CHECK(min_val <= max_val, error_code::invalid_argument,
                    "uniform_distribution: min must be <= max");
        min_ = min_val;
        max_ = max_val;
    }
    
private:
    T min_, max_;
};

// ============================================================================
// Normal/Gaussian Distribution
// ============================================================================

template<typename T>
class normal_distribution {
public:
    static_assert(std::is_floating_point_v<T>, "normal_distribution requires floating point type");
    
    normal_distribution(T mean = T(0), T stddev = T(1))
        : mean_(mean), stddev_(stddev), has_saved_(false) {
        EULER_CHECK(stddev > T(0), error_code::invalid_argument,
                    "normal_distribution: stddev must be positive");
    }
    
    template<typename Generator>
    T operator()(Generator& g) {
        if (has_saved_) {
            has_saved_ = false;
            return mean_ + stddev_ * saved_;
        }
        
        // Box-Muller transform
        T u1, u2;
        do {
            u1 = g.template uniform<T>();
        } while (u1 == T(0));
        u2 = g.template uniform<T>();
        
        T radius = std::sqrt(T(-2) * std::log(u1));
        T theta = T(2) * constants<T>::pi * u2;
        
        T z0 = radius * std::cos(theta);
        T z1 = radius * std::sin(theta);
        
        saved_ = z1;
        has_saved_ = true;
        
        return mean_ + stddev_ * z0;
    }
    
    T mean() const { return mean_; }
    T stddev() const { return stddev_; }
    
    void param(T mean, T stddev) {
        EULER_CHECK(stddev > T(0), error_code::invalid_argument,
                    "normal_distribution: stddev must be positive");
        mean_ = mean;
        stddev_ = stddev;
    }
    
    void reset() { has_saved_ = false; }
    
private:
    T mean_, stddev_;
    mutable bool has_saved_;
    mutable T saved_;
};

// ============================================================================
// Exponential Distribution
// ============================================================================

template<typename T>
class exponential_distribution {
public:
    static_assert(std::is_floating_point_v<T>, "exponential_distribution requires floating point type");
    
    explicit exponential_distribution(T lambda = T(1)) : lambda_(lambda) {
        EULER_CHECK(lambda > T(0), error_code::invalid_argument,
                    "exponential_distribution: lambda must be positive");
    }
    
    template<typename Generator>
    T operator()(Generator& g) {
        T u;
        do {
            u = g.template uniform<T>();
        } while (u == T(0));  // Avoid log(0)
        
        return -std::log(u) / lambda_;
    }
    
    T lambda() const { return lambda_; }
    
    void param(T lambda) {
        EULER_CHECK(lambda > T(0), error_code::invalid_argument,
                    "exponential_distribution: lambda must be positive");
        lambda_ = lambda;
    }
    
private:
    T lambda_;
};

// ============================================================================
// Bernoulli Distribution
// ============================================================================

class bernoulli_distribution {
public:
    explicit bernoulli_distribution(double p = 0.5) : p_(p) {
        EULER_CHECK(p >= 0.0 && p <= 1.0, error_code::invalid_argument,
                    "bernoulli_distribution: p must be in [0, 1]");
    }
    
    template<typename Generator>
    bool operator()(Generator& g) {
        return g.template uniform<double>() < p_;
    }
    
    double p() const { return p_; }
    
    void param(double p) {
        EULER_CHECK(p >= 0.0 && p <= 1.0, error_code::invalid_argument,
                    "bernoulli_distribution: p must be in [0, 1]");
        p_ = p;
    }
    
private:
    double p_;
};

// ============================================================================
// Discrete Distribution
// ============================================================================

template<typename T = int>
class discrete_distribution {
public:
    static_assert(std::is_integral_v<T>, "discrete_distribution requires integral type");
    
    // Constructor with weights
    template<typename InputIt>
    discrete_distribution(InputIt first, InputIt last) {
        init_weights(first, last);
    }
    
    // Constructor with initializer list
    discrete_distribution(std::initializer_list<double> weights) {
        init_weights(weights.begin(), weights.end());
    }
    
    template<typename Generator>
    T operator()(Generator& g) {
        double u = g.template uniform<double>() * total_weight_;
        
        // Binary search for the interval
        size_t low = 0, high = cumulative_.size();
        while (low < high) {
            size_t mid = (low + high) / 2;
            if (cumulative_[mid] < u) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        return static_cast<T>(low);
    }
    
    std::vector<double> probabilities() const {
        std::vector<double> probs;
        probs.reserve(cumulative_.size());
        
        double prev = 0.0;
        for (double cum : cumulative_) {
            probs.push_back((cum - prev) / total_weight_);
            prev = cum;
        }
        
        return probs;
    }
    
private:
    template<typename InputIt>
    void init_weights(InputIt first, InputIt last) {
        cumulative_.clear();
        total_weight_ = 0.0;
        
        for (auto it = first; it != last; ++it) {
            double weight = static_cast<double>(*it);
            EULER_CHECK(weight >= 0.0, error_code::invalid_argument,
                        "discrete_distribution: weights must be non-negative");
            total_weight_ += weight;
            cumulative_.push_back(total_weight_);
        }
        
        EULER_CHECK(total_weight_ > 0.0, error_code::invalid_argument,
                    "discrete_distribution: at least one weight must be positive");
    }
    
    std::vector<double> cumulative_;
    double total_weight_;
};

// ============================================================================
// Poisson Distribution
// ============================================================================

template<typename T = int>
class poisson_distribution {
public:
    static_assert(std::is_integral_v<T>, "poisson_distribution requires integral type");
    
    explicit poisson_distribution(double mean = 1.0) : mean_(mean) {
        EULER_CHECK(mean > 0.0, error_code::invalid_argument,
                    "poisson_distribution: mean must be positive");
        update_params();
    }
    
    template<typename Generator>
    T operator()(Generator& g) {
        if (mean_ < 10.0) {
            // Knuth's algorithm for small mean
            double L = exp_neg_mean_;
            T k = 0;
            double p = 1.0;
            
            do {
                ++k;
                p *= g.template uniform<double>();
            } while (p > L);
            
            return k - 1;
        } else {
            // Transformed rejection method for large mean
            // (simplified version)
            normal_distribution<double> norm(mean_, std::sqrt(mean_));
            double x;
            do {
                x = norm(g);
            } while (x < 0.0);
            
            return static_cast<T>(std::round(x));
        }
    }
    
    double mean() const { return mean_; }
    
    void param(double mean) {
        EULER_CHECK(mean > 0.0, error_code::invalid_argument,
                    "poisson_distribution: mean must be positive");
        mean_ = mean;
        update_params();
    }
    
private:
    void update_params() {
        exp_neg_mean_ = std::exp(-mean_);
    }
    
    double mean_;
    double exp_neg_mean_;
};

} // namespace euler