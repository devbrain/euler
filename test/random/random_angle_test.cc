#include <euler/random/random_angle.hh>
#include <euler/random/random.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace euler;

TEST_CASE("Random Angle Generation - Uniform Distribution") {
    random_generator rng(12345);
    
    SUBCASE("Degrees - Full circle") {
        const int N = 10000;
        std::vector<float> values;
        values.reserve(N);
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle<float, degree_tag>(rng);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 360.0f);
            values.push_back(angle.value());
        }
        
        // Check uniform distribution
        float mean = 0.0f;
        for (float v : values) mean += v;
        mean /= N;
        CHECK(std::abs(mean - 180.0f) < 5.0f);  // Should be near 180
        
        // Check variance (should be (360²)/12 ≈ 10800)
        float var = 0.0f;
        for (float v : values) {
            var += (v - mean) * (v - mean);
        }
        var /= N;
        CHECK(std::abs(var - 10800.0f) < 500.0f);
        
        // Check distribution across quadrants
        int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
        for (float v : values) {
            if (v < 90.0f) q1++;
            else if (v < 180.0f) q2++;
            else if (v < 270.0f) q3++;
            else q4++;
        }
        
        // Each quadrant should have ~25%
        CHECK(std::abs(q1 - N/4) < N/20);
        CHECK(std::abs(q2 - N/4) < N/20);
        CHECK(std::abs(q3 - N/4) < N/20);
        CHECK(std::abs(q4 - N/4) < N/20);
    }
    
    SUBCASE("Radians - Full circle") {
        const int N = 10000;
        std::vector<double> values;
        values.reserve(N);
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle<double, radian_tag>(rng);
            CHECK(angle.value() >= 0.0);
            CHECK(angle.value() < 2.0 * constants<double>::pi);
            values.push_back(angle.value());
        }
        
        // Check mean (should be π)
        double mean = 0.0;
        for (double v : values) mean += v;
        mean /= N;
        CHECK(std::abs(mean - constants<double>::pi) < 0.1);
    }
    
    SUBCASE("Custom range - Degrees") {
        auto min_angle = degree<float>(30);
        auto max_angle = degree<float>(150);
        
        const int N = 10000;
        std::vector<float> values;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle(rng, min_angle, max_angle);
            CHECK(angle.value() >= 30.0f);
            CHECK(angle.value() <= 150.0f);
            values.push_back(angle.value());
        }
        
        // Check mean (should be 90)
        float mean = 0.0f;
        for (float v : values) mean += v;
        mean /= N;
        CHECK(std::abs(mean - 90.0f) < 2.0f);
    }
    
    SUBCASE("Negative range - Radians") {
        auto min_angle = radian<float>(-constants<float>::pi);
        auto max_angle = radian<float>(constants<float>::pi);
        
        const int N = 10000;
        int positive = 0;
        int negative = 0;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle(rng, min_angle, max_angle);
            CHECK(angle.value() >= -constants<float>::pi);
            CHECK(angle.value() <= constants<float>::pi);
            
            if (angle.value() > 0) positive++;
            else if (angle.value() < 0) negative++;
        }
        
        // Should be roughly 50/50
        CHECK(std::abs(positive - negative) < N/20);
    }
}

TEST_CASE("Random Angle Generation - Normal Distribution") {
    random_generator rng(54321);
    
    SUBCASE("Degrees - Wrapping behavior") {
        auto mean = degree<float>(350);  // Near wraparound
        auto stddev = degree<float>(20);
        
        const int N = 10000;
        int wrapped = 0;  // Count how many wrapped to [0, 360)
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_normal(rng, mean, stddev);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 360.0f);
            
            // Values that wrapped around 360 will be small
            if (angle.value() < 180.0f) {
                wrapped++;
            }
        }
        
        // Some should wrap around
        CHECK(wrapped > 100);
        CHECK(wrapped < N/2);  // But not too many
    }
    
    SUBCASE("Radians - Standard deviation") {
        auto mean = radian<float>(constants<float>::pi);
        auto stddev = radian<float>(0.5f);
        
        const int N = 10000;
        std::vector<float> unwrapped_values;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_normal(rng, mean, stddev);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 2.0f * constants<float>::pi);
            
            // Unwrap values for statistics
            float val = angle.value();
            if (val < constants<float>::pi/2) {
                val += 2.0f * constants<float>::pi;  // Wrapped value
            }
            unwrapped_values.push_back(val);
        }
        
        // Check that ~68% are within 1 stddev
        int within_one_sigma = 0;
        for (float v : unwrapped_values) {
            if (std::abs(v - constants<float>::pi) <= 0.5f) {
                within_one_sigma++;
            }
        }
        
        float proportion = float(within_one_sigma) / N;
        CHECK(std::abs(proportion - 0.6827f) < 0.05f);
    }
    
    SUBCASE("Zero mean - Degrees") {
        auto mean = degree<float>(0);
        auto stddev = degree<float>(30);
        
        const int N = 1000;
        int near_zero = 0;
        int near_360 = 0;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_normal(rng, mean, stddev);
            
            if (angle.value() < 90.0f) near_zero++;
            if (angle.value() > 270.0f) near_360++;
        }
        
        // Should have values on both sides of the wraparound
        CHECK(near_zero > 100);
        CHECK(near_360 > 100);
    }
}

TEST_CASE("Random Angle Generation - Von Mises Distribution") {
    random_generator rng(99999);
    
    SUBCASE("Large kappa - Concentrated distribution") {
        auto mean = degree<float>(90);
        float kappa = 20.0f;  // High concentration
        
        const int N = 1000;
        int near_mean = 0;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_von_mises(rng, mean, kappa);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 360.0f);
            
            // Count values within 30 degrees of mean
            float diff = std::abs(angle.value() - 90.0f);
            if (diff <= 30.0f) {
                near_mean++;
            }
        }
        
        // With high kappa, most should be near mean
        CHECK(near_mean > 900);
    }
    
    SUBCASE("Small kappa - Dispersed distribution") {
        auto mean = radian<float>(0);
        float kappa = 0.5f;  // Low concentration
        
        const int N = 10000;
        std::vector<int> octants(8, 0);
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_von_mises(rng, mean, kappa);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 2.0f * constants<float>::pi);
            
            // Divide circle into 8 parts
            int octant = int(angle.value() * 8.0f / (2.0f * constants<float>::pi));
            if (octant >= 8) octant = 7;
            octants[static_cast<size_t>(octant)]++;
        }
        
        // With low kappa, distribution should be more spread out
        // but still peaked around mean (octant 0)
        CHECK(octants[0] > octants[4]);  // More at 0 than at π
        
        // But not too concentrated
        for (int count : octants) {
            CHECK(count > N/20);  // Each octant should have at least 5%
        }
    }
    
    SUBCASE("Medium kappa - Wraparound behavior") {
        auto mean = degree<float>(350);  // Near wraparound
        float kappa = 5.0f;
        
        const int N = 1000;
        int near_mean_before = 0;  // 320-360
        int near_mean_after = 0;   // 0-30
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_von_mises(rng, mean, kappa);
            
            if (angle.value() >= 320.0f) {
                near_mean_before++;
            }
            if (angle.value() <= 30.0f) {
                near_mean_after++;
            }
        }
        
        // Should have significant counts on both sides
        CHECK(near_mean_before > 200);
        CHECK(near_mean_after > 200);
        CHECK(near_mean_before + near_mean_after > 700);
    }
}

TEST_CASE("Random Angle Generation - Constrained Sum") {
    random_generator rng(11111);
    
    SUBCASE("Sum to full circle - Degrees") {
        auto target_sum = degree<float>(360);
        size_t count = 4;
        
        const int N = 1000;
        for (int i = 0; i < N; ++i) {
            auto angles = random_angles_constrained_sum(rng, count, target_sum);
            
            CHECK(angles.size() == count);
            
            // Check sum
            float sum = 0.0f;
            for (const auto& angle : angles) {
                CHECK(angle.value() >= 0.0f);
                sum += angle.value();
            }
            CHECK(std::abs(sum - 360.0f) < 1e-4f);
        }
    }
    
    SUBCASE("Sum to half circle - Radians") {
        auto target_sum = radian<float>(constants<float>::pi);
        size_t count = 3;
        
        const int N = 1000;
        std::vector<float> first_angle_values;
        
        for (int i = 0; i < N; ++i) {
            auto angles = random_angles_constrained_sum(rng, count, target_sum);
            
            CHECK(angles.size() == count);
            
            // Check sum
            float sum = 0.0f;
            for (const auto& angle : angles) {
                CHECK(angle.value() >= 0.0f);
                CHECK(angle.value() <= constants<float>::pi);
                sum += angle.value();
            }
            CHECK(std::abs(sum - constants<float>::pi) < 1e-4f);
            
            first_angle_values.push_back(angles[0].value());
        }
        
        // Check that first angle has reasonable distribution
        float mean = 0.0f;
        for (float v : first_angle_values) mean += v;
        mean /= N;
        
        // With 3 angles, mean of first should be ~π/3
        CHECK(std::abs(mean - constants<float>::pi/3) < 0.1f);
    }
    
    SUBCASE("Edge cases") {
        // Zero angles
        auto angles0 = random_angles_constrained_sum(
            rng, 0, degree<float>(180));
        CHECK(angles0.empty());
        
        // One angle
        auto angles1 = random_angles_constrained_sum(
            rng, 1, degree<float>(45));
        CHECK(angles1.size() == 1);
        CHECK(angles1[0].value() == 45.0f);
        
        // Many angles
        auto angles_many = random_angles_constrained_sum(
            rng, 100, radian<float>(2 * constants<float>::pi));
        CHECK(angles_many.size() == 100);
        
        float sum = 0.0f;
        for (const auto& angle : angles_many) {
            sum += angle.value();
        }
        CHECK(std::abs(sum - 2 * constants<float>::pi) < 1e-4f);
    }
}

TEST_CASE("Random Angle Generation - Thread Safety") {
    SUBCASE("Thread-local convenience functions") {
        // Test that thread-local functions work
        auto angle1 = random_angle<float, degree_tag>();
        CHECK(angle1.value() >= 0.0f);
        CHECK(angle1.value() < 360.0f);
        
        auto angle2 = random_angle<double, radian_tag>();
        CHECK(angle2.value() >= 0.0);
        CHECK(angle2.value() < 2.0 * constants<double>::pi);
        
        // Test with range
        auto min_deg = degree<float>(10);
        auto max_deg = degree<float>(20);
        auto angle3 = random_angle(min_deg, max_deg);
        CHECK(angle3.value() >= 10.0f);
        CHECK(angle3.value() <= 20.0f);
        
        // Test normal distribution
        auto mean = radian<double>(1.0);
        auto stddev = radian<double>(0.1);
        auto angle4 = random_angle_normal(mean, stddev);
        CHECK(angle4.value() >= 0.0);
        CHECK(angle4.value() < 2.0 * constants<double>::pi);
    }
}

TEST_CASE("Random Angle Generation - Statistical Properties") {
    random_generator rng(77777);
    
    SUBCASE("Circular mean and variance") {
        // Generate angles and compute circular statistics
        const int N = 10000;
        std::vector<complex<float>> unit_vectors;
        
        auto mean_angle = degree<float>(45);
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle_von_mises(rng, mean_angle, 5.0f);
            
            // Convert to unit vector
            auto rad = to_radians(angle);
            unit_vectors.push_back(complex<float>(static_cast<float>(cos(rad)), static_cast<float>(sin(rad))));
        }
        
        // Compute circular mean
        complex<float> mean_vector(0, 0);
        for (const auto& v : unit_vectors) {
            mean_vector = mean_vector + v;
        }
        mean_vector = mean_vector / static_cast<float>(N);
        
        // Extract mean angle
        float computed_mean = static_cast<float>(atan2(mean_vector.imag(), mean_vector.real()));
        if (computed_mean < 0) computed_mean += 2.0f * constants<float>::pi;
        computed_mean *= 180.0f / constants<float>::pi;
        
        CHECK(std::abs(computed_mean - 45.0f) < 2.0f);
        
        // Check concentration (R = |mean_vector|)
        float R = mean_vector.abs();
        CHECK(R > 0.8f);  // Should be concentrated for kappa=5
        CHECK(R < 0.95f); // But not too concentrated
    }
}