#include <euler/random/random.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace euler;

TEST_CASE("Random Generator Basic Operations") {
    SUBCASE("Default construction") {
        random_generator rng1;
        random_generator rng2;
        
        // Different generators should produce different sequences
        bool different = false;
        for (int i = 0; i < 10; ++i) {
            if (rng1() != rng2()) {
                different = true;
                break;
            }
        }
        CHECK(different);
    }
    
    SUBCASE("Seeded construction") {
        random_generator rng1(12345);
        random_generator rng2(12345);
        
        // Same seed should produce same sequence
        for (int i = 0; i < 100; ++i) {
            CHECK(rng1() == rng2());
        }
    }
    
    SUBCASE("Min/max values") {
        CHECK(random_generator::min() == 0);
        CHECK(random_generator::max() == std::numeric_limits<uint32_t>::max());
    }
    
    SUBCASE("Reseed") {
        random_generator rng(12345);
        uint32_t val1 = rng();
        uint32_t val2 = rng();
        
        rng.seed(12345);
        CHECK(rng() == val1);
        CHECK(rng() == val2);
    }
    
    SUBCASE("Discard") {
        random_generator rng1(12345);
        random_generator rng2(12345);
        
        rng1.discard(1000);
        for (int i = 0; i < 1000; ++i) {
            rng2();
        }
        
        // Should be at same position
        CHECK(rng1() == rng2());
    }
}

TEST_CASE("Uniform Distribution - Float") {
    random_generator rng(42);
    const int N = 10000;
    
    SUBCASE("Range [0, 1)") {
        std::vector<float> samples;
        for (int i = 0; i < N; ++i) {
            float val = rng.uniform<float>();
            CHECK(val >= 0.0f);
            CHECK(val < 1.0f);
            samples.push_back(val);
        }
        
        // Check mean and variance
        float mean = 0.0f;
        for (float x : samples) mean += x;
        mean /= static_cast<float>(N);
        CHECK(std::abs(mean - 0.5f) < 0.01f);
        
        float var = 0.0f;
        for (float x : samples) var += (x - mean) * (x - mean);
        var /= static_cast<float>(N);
        CHECK(std::abs(var - 1.0f/12.0f) < 0.01f);  // Variance of uniform[0,1] is 1/12
    }
    
    SUBCASE("Custom range") {
        float min_val = -5.0f;
        float max_val = 10.0f;
        
        std::vector<float> samples;
        for (int i = 0; i < N; ++i) {
            float val = rng.uniform(min_val, max_val);
            CHECK(val >= min_val);
            CHECK(val <= max_val);
            samples.push_back(val);
        }
        
        // Check mean
        float mean = 0.0f;
        for (float x : samples) mean += x;
        mean /= static_cast<float>(N);
        float expected_mean = (min_val + max_val) / 2.0f;
        CHECK(std::abs(mean - expected_mean) < 0.1f);
    }
}

TEST_CASE("Uniform Distribution - Integer") {
    random_generator rng(42);
    const int N = 10000;
    
    SUBCASE("Full range") {
        std::vector<int> samples;
        for (int i = 0; i < 100; ++i) {
            samples.push_back(rng.uniform<int>());
        }
        
        // Should have both positive and negative values
        bool has_positive = false;
        bool has_negative = false;
        for (int x : samples) {
            if (x > 0) has_positive = true;
            if (x < 0) has_negative = true;
        }
        CHECK(has_positive);
        CHECK(has_negative);
    }
    
    SUBCASE("Custom range") {
        int min_val = 1;
        int max_val = 6;  // Dice roll
        
        std::vector<int> counts(7, 0);  // 0-6, but 0 should never occur
        for (int i = 0; i < N; ++i) {
            int val = rng.uniform(min_val, max_val);
            CHECK(val >= min_val);
            CHECK(val <= max_val);
            counts[static_cast<size_t>(val)]++;
        }
        
        CHECK(counts[0] == 0);
        
        // Each value should appear roughly equally
        for (int i = 1; i <= 6; ++i) {
            float proportion = static_cast<float>(counts[static_cast<size_t>(i)]) / static_cast<float>(N);
            CHECK(std::abs(proportion - 1.0f/6.0f) < 0.02f);
        }
    }
}

TEST_CASE("Normal Distribution") {
    random_generator rng(42);
    const int N = 10000;
    
    SUBCASE("Standard normal") {
        std::vector<float> samples;
        for (int i = 0; i < N; ++i) {
            samples.push_back(rng.normal<float>());
        }
        
        // Check mean
        float mean = 0.0f;
        for (float x : samples) mean += x;
        mean /= static_cast<float>(N);
        CHECK(std::abs(mean - 0.0f) < 0.05f);
        
        // Check variance
        float var = 0.0f;
        for (float x : samples) var += x * x;
        var /= static_cast<float>(N);
        CHECK(std::abs(var - 1.0f) < 0.05f);
        
        // Check that ~68% of values are within 1 stddev
        int within_one_sigma = 0;
        for (float x : samples) {
            if (std::abs(x) <= 1.0f) within_one_sigma++;
        }
        float proportion = static_cast<float>(within_one_sigma) / static_cast<float>(N);
        CHECK(std::abs(proportion - 0.6827f) < 0.02f);
    }
    
    SUBCASE("Custom mean and stddev") {
        float target_mean = 100.0f;
        float target_stddev = 15.0f;
        
        std::vector<float> samples;
        for (int i = 0; i < N; ++i) {
            samples.push_back(rng.normal(target_mean, target_stddev));
        }
        
        // Check mean
        float mean = 0.0f;
        for (float x : samples) mean += x;
        mean /= static_cast<float>(N);
        CHECK(std::abs(mean - target_mean) < 0.5f);
        
        // Check stddev
        float var = 0.0f;
        for (float x : samples) var += (x - mean) * (x - mean);
        var /= static_cast<float>(N);
        float stddev = std::sqrt(var);
        CHECK(std::abs(stddev - target_stddev) < 0.5f);
    }
}

TEST_CASE("Thread-local generator") {
    SUBCASE("Basic usage") {
        // Thread-local generator should work
        float val1 = random_uniform<float>();
        CHECK(val1 >= 0.0f);
        CHECK(val1 < 1.0f);
        
        int val2 = random_uniform(1, 10);
        CHECK(val2 >= 1);
        CHECK(val2 <= 10);
        
        float val3 = random_normal(0.0f, 1.0f);
        CHECK(std::isfinite(val3));
    }
    
    SUBCASE("Same thread gets same generator") {
        auto& rng1 = thread_local_rng();
        auto& rng2 = thread_local_rng();
        CHECK(&rng1 == &rng2);
    }
}

TEST_CASE("Statistical properties") {
    random_generator rng(12345);
    
    SUBCASE("Uniformity test - Chi-square") {
        const int bins = 10;
        const int N = 10000;
        std::vector<int> counts(bins, 0);
        
        for (int i = 0; i < N; ++i) {
            float val = rng.uniform<float>();
            int bin = static_cast<int>(val * bins);
            if (bin == bins) bin = bins - 1;  // Handle edge case
            counts[static_cast<size_t>(bin)]++;
        }
        
        // Chi-square test
        float expected = static_cast<float>(N) / static_cast<float>(bins);
        float chi_square = 0.0f;
        for (int count : counts) {
            float diff = static_cast<float>(count) - expected;
            chi_square += (diff * diff) / expected;
        }
        
        // For 9 degrees of freedom, critical value at 95% confidence is 16.919
        CHECK(chi_square < 16.919f);
    }
    
    SUBCASE("Independence test") {
        // Simple lag-1 autocorrelation test
        const int N = 1000;
        std::vector<float> samples;
        for (int i = 0; i < N; ++i) {
            samples.push_back(rng.uniform<float>());
        }
        
        float mean = 0.0f;
        for (float x : samples) mean += x;
        mean /= static_cast<float>(N);
        
        float autocorr = 0.0f;
        float var = 0.0f;
        for (int i = 0; i < N - 1; ++i) {
            autocorr += (samples[static_cast<size_t>(i)] - mean) * (samples[static_cast<size_t>(i + 1)] - mean);
            var += (samples[static_cast<size_t>(i)] - mean) * (samples[static_cast<size_t>(i)] - mean);
        }
        var += (samples[static_cast<size_t>(N-1)] - mean) * (samples[static_cast<size_t>(N-1)] - mean);
        
        autocorr /= static_cast<float>(N - 1);
        var /= static_cast<float>(N);
        
        float correlation = autocorr / var;
        CHECK(std::abs(correlation) < 0.05f);  // Should be near zero
    }
}