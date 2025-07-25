#include <euler/random/random.hh>
#include <euler/random/distributions.hh>
#include <doctest/doctest.h>
#include <vector>
#include <numeric>
#include <cmath>

using namespace euler;

TEST_CASE("Uniform Distribution") {
    random_generator rng(42);
    
    SUBCASE("Float distribution") {
        uniform_distribution<float> dist(2.0f, 5.0f);
        
        const int N = 10000;
        std::vector<float> samples;
        
        for (int i = 0; i < N; ++i) {
            float val = dist(rng);
            CHECK(val >= 2.0f);
            CHECK(val <= 5.0f);
            samples.push_back(val);
        }
        
        // Check mean
        float mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / static_cast<float>(N);
        CHECK(std::abs(mean - 3.5f) < 0.05f);
        
        // Check range
        CHECK(dist.min() == 2.0f);
        CHECK(dist.max() == 5.0f);
        
        // Test param update
        dist.param(0.0f, 1.0f);
        CHECK(dist.min() == 0.0f);
        CHECK(dist.max() == 1.0f);
    }
    
    SUBCASE("Integer distribution") {
        uniform_distribution<int> dist(10, 20);
        
        std::vector<int> counts(21, 0);
        const int N = 11000;
        
        for (int i = 0; i < N; ++i) {
            int val = dist(rng);
            CHECK(val >= 10);
            CHECK(val <= 20);
            counts[static_cast<size_t>(val)]++;
        }
        
        // Each value should appear roughly equally
        for (int i = 10; i <= 20; ++i) {
            float proportion = static_cast<float>(counts[static_cast<size_t>(i)]) / static_cast<float>(N);
            CHECK(std::abs(proportion - 1.0f/11.0f) < 0.02f);
        }
    }
}

TEST_CASE("Normal Distribution") {
    random_generator rng(42);
    
    SUBCASE("Basic functionality") {
        normal_distribution<float> dist(10.0f, 2.0f);
        
        CHECK(dist.mean() == 10.0f);
        CHECK(dist.stddev() == 2.0f);
        
        const int N = 10000;
        std::vector<float> samples;
        
        for (int i = 0; i < N; ++i) {
            samples.push_back(dist(rng));
        }
        
        // Check mean
        float mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / static_cast<float>(N);
        CHECK(std::abs(mean - 10.0f) < 0.1f);
        
        // Check stddev
        float var = 0.0f;
        for (float x : samples) {
            var += (x - mean) * (x - mean);
        }
        var /= static_cast<float>(N);
        float stddev = std::sqrt(var);
        CHECK(std::abs(stddev - 2.0f) < 0.1f);
    }
    
    SUBCASE("Reset functionality") {
        normal_distribution<double> dist;
        
        // Generate some values
        for (int i = 0; i < 10; ++i) {
            dist(rng);
        }
        
        // Reset should clear saved state
        dist.reset();
        
        // Should still work correctly
        double val = dist(rng);
        CHECK(std::isfinite(val));
    }
    
    SUBCASE("Parameter update") {
        normal_distribution<float> dist;
        
        dist.param(100.0f, 15.0f);
        CHECK(dist.mean() == 100.0f);
        CHECK(dist.stddev() == 15.0f);
        
        // Should work after param change
        float val = dist(rng);
        CHECK(std::isfinite(val));
    }
}

TEST_CASE("Exponential Distribution") {
    random_generator rng(42);
    
    SUBCASE("Basic functionality") {
        exponential_distribution<float> dist(2.0f);
        
        CHECK(dist.lambda() == 2.0f);
        
        const int N = 10000;
        std::vector<float> samples;
        
        for (int i = 0; i < N; ++i) {
            float val = dist(rng);
            CHECK(val >= 0.0f);
            samples.push_back(val);
        }
        
        // Check mean (should be 1/lambda = 0.5)
        float mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / static_cast<float>(N);
        CHECK(std::abs(mean - 0.5f) < 0.02f);
        
        // Check that about 63.2% of values are less than mean
        int below_mean = 0;
        for (float x : samples) {
            if (x < 0.5f) below_mean++;
        }
        float proportion = float(below_mean) / N;
        CHECK(std::abs(proportion - 0.632f) < 0.02f);
    }
    
    SUBCASE("Parameter update") {
        exponential_distribution<double> dist(1.0);
        
        dist.param(0.5);
        CHECK(dist.lambda() == 0.5);
        
        // Mean should now be 2.0
        const int N = 10000;
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += dist(rng);
        }
        double mean = sum / static_cast<double>(N);
        CHECK(std::abs(mean - 2.0) < 0.05);
    }
}

TEST_CASE("Bernoulli Distribution") {
    random_generator rng(42);
    
    SUBCASE("Fair coin") {
        bernoulli_distribution dist(0.5);
        
        CHECK(dist.p() == 0.5);
        
        int successes = 0;
        const int N = 10000;
        
        for (int i = 0; i < N; ++i) {
            if (dist(rng)) successes++;
        }
        
        float proportion = float(successes) / N;
        CHECK(std::abs(proportion - 0.5f) < 0.02f);
    }
    
    SUBCASE("Biased coin") {
        bernoulli_distribution dist(0.7);
        
        int successes = 0;
        const int N = 10000;
        
        for (int i = 0; i < N; ++i) {
            if (dist(rng)) successes++;
        }
        
        float proportion = float(successes) / N;
        CHECK(std::abs(proportion - 0.7f) < 0.02f);
    }
    
    SUBCASE("Edge cases") {
        bernoulli_distribution always_false(0.0);
        bernoulli_distribution always_true(1.0);
        
        for (int i = 0; i < 100; ++i) {
            CHECK(always_false(rng) == false);
            CHECK(always_true(rng) == true);
        }
    }
}

TEST_CASE("Discrete Distribution") {
    random_generator rng(42);
    
    SUBCASE("From weights vector") {
        std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
        discrete_distribution<int> dist(weights.begin(), weights.end());
        
        std::vector<int> counts(4, 0);
        const int N = 10000;
        
        for (int i = 0; i < N; ++i) {
            int val = dist(rng);
            CHECK(val >= 0);
            CHECK(val < 4);
            counts[static_cast<size_t>(val)]++;
        }
        
        // Check proportions
        double total_weight = 10.0;
        for (int i = 0; i < 4; ++i) {
            float expected = static_cast<float>(weights[static_cast<size_t>(i)] / total_weight);
            float actual = static_cast<float>(counts[static_cast<size_t>(i)]) / static_cast<float>(N);
            CHECK(std::abs(actual - expected) < 0.02f);
        }
    }
    
    SUBCASE("From initializer list") {
        discrete_distribution<int> dist{0.1, 0.2, 0.3, 0.4};
        
        auto probs = dist.probabilities();
        CHECK(probs.size() == 4);
        CHECK(std::abs(probs[0] - 0.1) < 1e-6);
        CHECK(std::abs(probs[1] - 0.2) < 1e-6);
        CHECK(std::abs(probs[2] - 0.3) < 1e-6);
        CHECK(std::abs(probs[3] - 0.4) < 1e-6);
    }
    
    SUBCASE("Single weight") {
        discrete_distribution<int> dist{1.0};
        
        // Should always return 0
        for (int i = 0; i < 100; ++i) {
            CHECK(dist(rng) == 0);
        }
    }
}

TEST_CASE("Poisson Distribution") {
    random_generator rng(42);
    
    SUBCASE("Small mean") {
        poisson_distribution<int> dist(3.0);
        
        CHECK(dist.mean() == 3.0);
        
        const int N = 10000;
        std::vector<int> counts(20, 0);
        double sum = 0.0;
        
        for (int i = 0; i < N; ++i) {
            int val = dist(rng);
            CHECK(val >= 0);
            if (val < 20) counts[static_cast<size_t>(val)]++;
            sum += val;
        }
        
        // Check mean
        double mean = sum / static_cast<double>(N);
        CHECK(std::abs(mean - 3.0) < 0.1);
        
        // Check mode is at 2 or 3
        int max_count = 0;
        int mode = 0;
        for (int i = 0; i < 20; ++i) {
            if (counts[static_cast<size_t>(i)] > max_count) {
                max_count = counts[static_cast<size_t>(i)];
                mode = i;
            }
        }
        CHECK((mode == 2 || mode == 3));
    }
    
    SUBCASE("Large mean") {
        poisson_distribution<int> dist(100.0);
        
        const int N = 10000;
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (int i = 0; i < N; ++i) {
            int val = dist(rng);
            CHECK(val >= 0);
            sum += val;
            sum_sq += val * val;
        }
        
        // Check mean
        double mean = sum / static_cast<double>(N);
        CHECK(std::abs(mean - 100.0) < 1.0);
        
        // Check variance (should also be close to mean for Poisson)
        double var = sum_sq / N - mean * mean;
        CHECK(std::abs(var - 100.0) < 5.0);
    }
    
    SUBCASE("Parameter update") {
        poisson_distribution<int> dist(1.0);
        
        dist.param(5.0);
        CHECK(dist.mean() == 5.0);
        
        // Should work after param change
        int val = dist(rng);
        CHECK(val >= 0);
    }
}