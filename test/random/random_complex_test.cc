#include <euler/random/random_complex.hh>
#include <euler/random/random.hh>
#include <euler/complex/complex.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace euler;

TEST_CASE("Random Complex - Unit Circle") {
    random_generator rng(12345);
    
    SUBCASE("Magnitude check") {
        const int N = 10000;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_unit<float>(rng);
            float mag = c.abs();
            CHECK(std::abs(mag - 1.0f) < 1e-6f);
        }
    }
    
    SUBCASE("Angle distribution") {
        const int N = 10000;
        std::vector<float> angles;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_unit<double>(rng);
            double angle = c.arg().value();  // Returns radian
            CHECK(angle >= -constants<double>::pi);
            CHECK(angle <= constants<double>::pi);
            
            // Convert to [0, 2π) for easier analysis
            if (angle < 0) angle += 2.0 * constants<double>::pi;
            angles.push_back(static_cast<float>(angle));
        }
        
        // Check uniform distribution of angles
        int bins[8] = {0};
        for (float angle : angles) {
            int bin = static_cast<int>(angle * 8.0f / (2.0f * constants<float>::pi));
            if (bin >= 8) bin = 7;
            bins[bin]++;
        }
        
        // Each bin should have roughly N/8 values
        for (int i = 0; i < 8; ++i) {
            CHECK(std::abs(bins[i] - N/8) < N/20);
        }
    }
    
    SUBCASE("Mean should be near zero") {
        const int N = 10000;
        complex<float> sum(0, 0);
        
        for (int i = 0; i < N; ++i) {
            sum = sum + random_complex_unit<float>(rng);
        }
        
        complex<float> mean = sum / static_cast<float>(N);
        CHECK(mean.abs() < 0.02f);  // Should be very close to origin
    }
}

TEST_CASE("Random Complex - Rectangle") {
    random_generator rng(54321);
    
    SUBCASE("Square region") {
        const int N = 10000;
        float size = 2.5f;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex<float>(rng, size);
            CHECK(c.real() >= -size);
            CHECK(c.real() <= size);
            CHECK(c.imag() >= -size);
            CHECK(c.imag() <= size);
        }
    }
    
    SUBCASE("Custom rectangle") {
        const int N = 10000;
        double real_min = -3.0, real_max = 5.0;
        double imag_min = -1.0, imag_max = 2.0;
        
        double sum_real = 0.0, sum_imag = 0.0;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex<double>(rng, real_min, real_max, 
                                          imag_min, imag_max);
            CHECK(c.real() >= real_min);
            CHECK(c.real() <= real_max);
            CHECK(c.imag() >= imag_min);
            CHECK(c.imag() <= imag_max);
            
            sum_real += c.real();
            sum_imag += c.imag();
        }
        
        // Check means
        double mean_real = sum_real / static_cast<double>(N);
        double mean_imag = sum_imag / static_cast<double>(N);
        
        CHECK(std::abs(mean_real - (real_min + real_max)/2.0) < 0.1);
        CHECK(std::abs(mean_imag - (imag_min + imag_max)/2.0) < 0.1);
    }
}

TEST_CASE("Random Complex - Disk") {
    random_generator rng(99999);
    
    SUBCASE("Unit disk") {
        const int N = 10000;
        int inside_half = 0;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_disk<float>(rng);
            float mag = c.abs();
            CHECK(mag <= 1.0f);
            
            if (mag <= 0.5f) {
                inside_half++;
            }
        }
        
        // Area ratio: π(0.5)²/π(1)² = 0.25
        float ratio = static_cast<float>(inside_half) / static_cast<float>(N);
        CHECK(std::abs(ratio - 0.25f) < 0.02f);
    }
    
    SUBCASE("Custom radius") {
        const int N = 10000;
        double radius = 3.0;
        
        double max_mag = 0.0;
        double sum_mag_squared = 0.0;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_disk<double>(rng, radius);
            double mag = c.abs();
            CHECK(mag <= radius);
            
            max_mag = std::max(max_mag, mag);
            sum_mag_squared += mag * mag;
        }
        
        // Should get close to the radius
        CHECK(max_mag > radius * 0.99);
        
        // For uniform distribution in disk, E[r²] = R²/2
        auto mean_mag_squared = sum_mag_squared / static_cast<double>(N);
        CHECK(std::abs(mean_mag_squared - radius*radius/2) < 0.1);
    }
}

TEST_CASE("Random Complex - Annulus") {
    random_generator rng(11111);
    
    SUBCASE("Basic annulus") {
        const int N = 10000;
        float inner = 1.0f;
        float outer = 2.0f;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_annulus<float>(rng, inner, outer);
            float mag = c.abs();
            CHECK(mag >= inner);
            CHECK(mag <= outer);
        }
    }
    
    SUBCASE("Area distribution") {
        const int N = 10000;
        float inner = 1.0f;
        float outer = 3.0f;
        auto middle = 2.0;
        
        int below_middle = 0;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_annulus<double>(rng, inner, outer);
            if (c.abs() < middle) {
                below_middle++;
            }
        }
        
        // Area ratio: (π·2² - π·1²)/(π·3² - π·1²) = 3/8 = 0.375
        float ratio = float(below_middle) / N;
        CHECK(std::abs(ratio - 0.375f) < 0.02f);
    }
    
    SUBCASE("Zero inner radius") {
        // Should behave like disk
        const int N = 1000;
        float outer = 2.0f;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_annulus<float>(rng, 0.0f, outer);
            CHECK(c.abs() <= outer);
        }
    }
}

TEST_CASE("Random Complex - Normal Distribution") {
    random_generator rng(22222);
    
    SUBCASE("Standard normal") {
        const unsigned int N = 10000;
        std::vector<float> real_parts, imag_parts;
        
        for (unsigned int i = 0; i < N; ++i) {
            auto c = random_complex_normal<float>(rng);
            real_parts.push_back(c.real());
            imag_parts.push_back(c.imag());
        }
        
        // Check means (should be 0)
        float mean_real = 0.0f, mean_imag = 0.0f;
        for (unsigned int i = 0; i < N; ++i) {
            mean_real += real_parts[i];
            mean_imag += imag_parts[i];
        }
        mean_real /= N;
        mean_imag /= N;
        
        CHECK(std::abs(mean_real) < 0.02f);
        CHECK(std::abs(mean_imag) < 0.02f);
        
        // Check variances (should be 1)
        float var_real = 0.0f, var_imag = 0.0f;
        for (unsigned int i = 0; i < N; ++i) {
            var_real += (real_parts[i] - mean_real) * (real_parts[i] - mean_real);
            var_imag += (imag_parts[i] - mean_imag) * (imag_parts[i] - mean_imag);
        }
        var_real /= N;
        var_imag /= N;
        
        CHECK(std::abs(var_real - 1.0f) < 0.05f);
        CHECK(std::abs(var_imag - 1.0f) < 0.05f);
    }
    
    SUBCASE("Custom mean and stddev") {
        const int N = 10000;
        complex<double> mean(10.0, -5.0);
        double stddev = 2.0;
        
        complex<double> sum(0, 0);
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_normal<double>(rng, mean, stddev);
            sum = sum + c;
        }
        
        complex<double> computed_mean = sum / double(N);
        CHECK(std::abs(computed_mean.real() - mean.real()) < 0.05);
        CHECK(std::abs(computed_mean.imag() - mean.imag()) < 0.05);
    }
    
    SUBCASE("Independent real/imag stddev") {
        const int N = 10000;
        complex<float> mean(0, 0);
        float real_stddev = 1.0f;
        float imag_stddev = 3.0f;
        
        std::vector<float> real_parts, imag_parts;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_normal<float>(rng, mean, real_stddev, imag_stddev);
            real_parts.push_back(c.real());
            imag_parts.push_back(c.imag());
        }
        
        // Compute actual standard deviations
        float mean_real = 0.0f, mean_imag = 0.0f;
        for (auto x : real_parts) mean_real += x;
        for (auto x : imag_parts) mean_imag += x;
        mean_real /= N;
        mean_imag /= N;
        
        float var_real = 0.0f, var_imag = 0.0f;
        for (auto x : real_parts) var_real += (x - mean_real) * (x - mean_real);
        for (auto x : imag_parts) var_imag += (x - mean_imag) * (x - mean_imag);
        var_real /= N;
        var_imag /= N;
        
        float std_real = std::sqrt(var_real);
        float std_imag = std::sqrt(var_imag);
        
        CHECK(std::abs(std_real - real_stddev) < 0.05f);
        CHECK(std::abs(std_imag - imag_stddev) < 0.15f);
    }
}

TEST_CASE("Random Complex - Fixed Magnitude/Phase") {
    random_generator rng(33333);
    
    SUBCASE("Fixed magnitude") {
        const int N = 1000;
        float magnitude = 2.5f;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_fixed_magnitude<float>(rng, magnitude);
            CHECK(std::abs(c.abs() - magnitude) < 1e-6f);
        }
    }
    
    SUBCASE("Fixed phase") {
        const int N = 1000;
        auto phase = radian<double>(constants<double>::pi / 4);  // 45 degrees
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_fixed_phase<double>(rng, phase, 0.5, 2.0);
            
            // Check phase
            auto computed_phase = c.arg();
            CHECK(std::abs(computed_phase.value() - phase.value()) < 1e-6);
            
            // Check magnitude range
            CHECK(c.abs() >= 0.5);
            CHECK(c.abs() <= 2.0);
        }
    }
}

TEST_CASE("Random Complex - Roots of Unity") {
    random_generator rng(44444);
    
    SUBCASE("Fourth roots") {
        const int N = 1000;
        std::vector<int> counts(4, 0);
        
        for (int i = 0; i < N; ++i) {
            auto c = random_root_of_unity<float>(rng, 4);
            
            // Should be one of: 1, i, -1, -i
            float real = c.real();
            float imag = c.imag();
            
            CHECK(std::abs(c.abs() - 1.0f) < 1e-6f);
            
            if (std::abs(real - 1.0f) < 1e-6f && std::abs(imag) < 1e-6f) {
                counts[0]++;  // 1
            } else if (std::abs(real) < 1e-6f && std::abs(imag - 1.0f) < 1e-6f) {
                counts[1]++;  // i
            } else if (std::abs(real + 1.0f) < 1e-6f && std::abs(imag) < 1e-6f) {
                counts[2]++;  // -1
            } else if (std::abs(real) < 1e-6f && std::abs(imag + 1.0f) < 1e-6f) {
                counts[3]++;  // -i
            } else {
                CHECK(false);  // Should be one of the four roots
            }
        }
        
        // Each root should appear roughly equally
        for (int count : counts) {
            CHECK(std::abs(count - N/4) < N/20);
        }
    }
    
    SUBCASE("Large n") {
        int n = 17;  // Prime number
        
        for (int i = 0; i < 100; ++i) {
            auto c = random_root_of_unity<double>(rng, n);
            
            // Check that c^n = 1
            complex<double> power = c;
            for (int j = 1; j < n; ++j) {
                power = power * c;
            }
            
            CHECK(std::abs(power.real() - 1.0) < 1e-10);
            CHECK(std::abs(power.imag()) < 1e-10);
        }
    }
}

TEST_CASE("Random Complex - Log Normal") {
    random_generator rng(55555);
    
    SUBCASE("Basic log normal") {
        const int N = 10000;
        std::vector<float> log_magnitudes;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_log_normal<float>(rng, 0.0f, 0.5f);
            float mag = c.abs();
            CHECK(mag > 0.0f);
            
            log_magnitudes.push_back(std::log(mag));
        }
        
        // Check that log magnitudes are normally distributed
        float mean = 0.0f;
        for (float lm : log_magnitudes) mean += lm;
        mean /= N;
        
        CHECK(std::abs(mean - 0.0f) < 0.05f);
        
        float var = 0.0f;
        for (float lm : log_magnitudes) {
            var += (lm - mean) * (lm - mean);
        }
        var /= N;
        float stddev = std::sqrt(var);
        
        CHECK(std::abs(stddev - 0.5f) < 0.05f);
    }
    
    SUBCASE("Angle distribution") {
        const int N = 1000;
        
        // Angles should still be uniform
        std::vector<double> angles;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_log_normal<double>(rng, 1.0, 0.5);
            double angle = c.arg().value();
            if (angle < 0) angle += 2.0 * constants<double>::pi;
            angles.push_back(angle);
        }
        
        // Check uniformity
        std::sort(angles.begin(), angles.end());
        
        // Check that angles are well-distributed
        for (size_t i = 1; i < angles.size(); ++i) {
            auto gap = angles[i] - angles[i-1];
            CHECK(gap < 0.1);  // No large gaps
        }
    }
}

TEST_CASE("Random Complex - Thread-local Functions") {
    SUBCASE("All convenience functions") {
        // Unit circle
        auto c1 = random_complex_unit<float>();
        CHECK(std::abs(c1.abs() - 1.0f) < 1e-6f);
        
        // Square
        auto c2 = random_complex<double>(3.0);
        CHECK(std::abs(c2.real()) <= 3.0);
        CHECK(std::abs(c2.imag()) <= 3.0);
        
        // Disk
        auto c3 = random_complex_disk<float>(2.0f);
        CHECK(c3.abs() <= 2.0f);
        
        // Normal
        complex<float> mean(1.0f, 2.0f);
        auto c4 = random_complex_normal(mean, 0.5f);
        CHECK(std::isfinite(c4.real()));
        CHECK(std::isfinite(c4.imag()));
    }
}