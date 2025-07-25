#include <euler/euler.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/quaternion/quaternion.hh>
#include <euler/quaternion/quaternion_ops.hh>
#include <euler/complex/complex.hh>
#include <euler/angles/angle.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>

using namespace euler;

TEST_CASE("Random Angles") {
    random_generator rng(42);
    
    SUBCASE("Uniform angle generation") {
        // Test degrees
        const int N = 1000;
        std::vector<float> deg_values;
        
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle<float, degree_tag>(rng);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 360.0f);
            deg_values.push_back(angle.value());
        }
        
        // Check uniform distribution
        float mean = 0.0f;
        for (float x : deg_values) mean += x;
        mean /= N;
        CHECK(std::abs(mean - 180.0f) < 10.0f);
        
        // Test radians
        std::vector<float> rad_values;
        for (int i = 0; i < N; ++i) {
            auto angle = random_angle<float, radian_tag>(rng);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 2.0f * constants<float>::pi);
            rad_values.push_back(angle.value());
        }
    }
    
    SUBCASE("Angle range") {
        auto min_angle = degree<float>(45);
        auto max_angle = degree<float>(135);
        
        for (int i = 0; i < 100; ++i) {
            auto angle = random_angle(rng, min_angle, max_angle);
            CHECK(angle.value() >= 45.0f);
            CHECK(angle.value() <= 135.0f);
        }
    }
    
    SUBCASE("Normal distribution") {
        auto mean = degree<float>(90);
        auto stddev = degree<float>(10);
        
        std::vector<float> values;
        for (int i = 0; i < 1000; ++i) {
            auto angle = random_angle_normal(rng, mean, stddev);
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() < 360.0f);
            values.push_back(angle.value());
        }
        
        // Most values should be near 90 degrees
        int near_mean = 0;
        for (float val : values) {
            if (std::abs(val - 90.0f) < 30.0f) {  // Within 3 sigma
                near_mean++;
            }
        }
        CHECK(near_mean > 900);  // Should be ~997 for normal distribution
    }
}

TEST_CASE("Random Complex Numbers") {
    random_generator rng(42);
    
    SUBCASE("Unit circle") {
        const int N = 1000;
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_unit<float>(rng);
            float mag = c.abs();
            CHECK(std::abs(mag - 1.0f) < 1e-6f);
        }
    }
    
    SUBCASE("Rectangle") {
        for (int i = 0; i < 100; ++i) {
            auto c = random_complex<float>(rng, -2.0f, 3.0f, -1.0f, 4.0f);
            CHECK(c.real() >= -2.0f);
            CHECK(c.real() <= 3.0f);
            CHECK(c.imag() >= -1.0f);
            CHECK(c.imag() <= 4.0f);
        }
    }
    
    SUBCASE("Disk") {
        const int N = 1000;
        int inside_half = 0;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_disk<float>(rng, 2.0f);
            float mag = c.abs();
            CHECK(mag <= 2.0f);
            
            if (mag <= 1.0f) {
                inside_half++;
            }
        }
        
        // Area ratio test: inner circle / outer circle = 1/4
        float ratio = float(inside_half) / N;
        CHECK(std::abs(ratio - 0.25f) < 0.05f);
    }
    
    SUBCASE("Normal distribution") {
        complex<float> mean(10.0f, 20.0f);
        const int N = 1000;
        
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto c = random_complex_normal(rng, mean, 2.0f);
            sum_real += c.real();
            sum_imag += c.imag();
        }
        
        CHECK(std::abs(sum_real / N - 10.0f) < 0.2f);
        CHECK(std::abs(sum_imag / N - 20.0f) < 0.2f);
    }
}

TEST_CASE("Random Vectors") {
    random_generator rng(42);
    
    SUBCASE("Uniform components") {
        for (int i = 0; i < 100; ++i) {
            auto v = random_vector<float, 3>(rng, -1.0f, 1.0f);
            CHECK(v[0] >= -1.0f); CHECK(v[0] <= 1.0f);
            CHECK(v[1] >= -1.0f); CHECK(v[1] <= 1.0f);
            CHECK(v[2] >= -1.0f); CHECK(v[2] <= 1.0f);
        }
    }
    
    SUBCASE("Unit vectors - 2D") {
        const int N = 1000;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<float, 2>(rng);
            CHECK(std::abs(v.length() - 1.0f) < 1e-6f);
            sum_x += v[0];
            sum_y += v[1];
        }
        
        // Should average to zero
        CHECK(std::abs(sum_x / N) < 0.1f);
        CHECK(std::abs(sum_y / N) < 0.1f);
    }
    
    SUBCASE("Unit vectors - 3D") {
        const int N = 1000;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_unit_vector<float, 3>(rng);
            CHECK(std::abs(v.length() - 1.0f) < 1e-6f);
            sum_x += v[0];
            sum_y += v[1];
            sum_z += v[2];
        }
        
        // Should average to zero
        CHECK(std::abs(sum_x / N) < 0.1f);
        CHECK(std::abs(sum_y / N) < 0.1f);
        CHECK(std::abs(sum_z / N) < 0.1f);
    }
    
    SUBCASE("In sphere") {
        const int N = 1000;
        int inside_half = 0;
        
        for (int i = 0; i < N; ++i) {
            auto v = random_in_sphere<float, 3>(rng);
            float len = v.length();
            CHECK(len <= 1.0f);
            
            if (len <= 0.5f) {
                inside_half++;
            }
        }
        
        // Volume ratio test: inner sphere / outer sphere = (0.5)^3 = 0.125
        float ratio = float(inside_half) / N;
        CHECK(std::abs(ratio - 0.125f) < 0.03f);
    }
}

TEST_CASE("Random Quaternions") {
    random_generator rng(42);
    
    SUBCASE("Uniform distribution") {
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion<float>(rng);
            CHECK(std::abs(q.norm() - 1.0f) < 1e-6f);
        }
        
        // Check component statistics
        // Each component should have mean near 0 due to uniform distribution
        float sum_w = 0, sum_x = 0, sum_y = 0, sum_z = 0;
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion<float>(rng);
            sum_w += q.w();
            sum_x += q.x();
            sum_y += q.y();
            sum_z += q.z();
        }
        
        // Means should be near zero
        CHECK(std::abs(sum_w / N) < 0.03f);
        CHECK(std::abs(sum_x / N) < 0.03f);
        CHECK(std::abs(sum_y / N) < 0.03f);
        CHECK(std::abs(sum_z / N) < 0.03f);
    }
    
    SUBCASE("Limited angle") {
        auto max_angle = radian<float>(constants<float>::pi / 4);  // 45 degrees
        
        for (int i = 0; i < 100; ++i) {
            auto q = random_quaternion(rng, max_angle);
            CHECK(q.is_normalized());
            
            auto angle = q.angle();
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() <= max_angle.value() + 1e-6f);
        }
    }
    
    SUBCASE("Small rotations") {
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_small<float>(rng, 0.1f);
            CHECK(q.is_normalized());
            
            auto angle = q.angle();
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() <= 0.1f + 1e-6f);
        }
    }
}

TEST_CASE("Random Matrices") {
    random_generator rng(42);
    
    SUBCASE("Uniform elements") {
        for (int i = 0; i < 10; ++i) {
            auto m = random_matrix<float, 3, 3>(rng, -2.0f, 2.0f);
            
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    CHECK(m(r, c) >= -2.0f);
                    CHECK(m(r, c) <= 2.0f);
                }
            }
        }
    }
    
    SUBCASE("Rotation matrices - 2D") {
        for (int i = 0; i < 100; ++i) {
            auto m = random_rotation_matrix<float, 2>(rng);
            
            // Check orthogonality
            auto mt = transpose(m);
            auto prod = m * mt;
            CHECK(approx_equal(prod, matrix<float, 2, 2>::identity(), 1e-6f));
            
            // Check determinant = 1
            CHECK(std::abs(determinant(m) - 1.0f) < 1e-6f);
        }
    }
    
    SUBCASE("Rotation matrices - 3D") {
        for (int i = 0; i < 100; ++i) {
            auto m = random_rotation_matrix<float, 3>(rng);
            
            // Check orthogonality
            auto mt = transpose(m);
            auto prod = m * mt;
            CHECK(approx_equal(prod, matrix<float, 3, 3>::identity(), 1e-6f));
            
            // Check determinant = 1
            CHECK(std::abs(determinant(m) - 1.0f) < 1e-6f);
        }
    }
    
    SUBCASE("Orthogonal matrices") {
        int positive_det = 0;
        int negative_det = 0;
        
        for (int i = 0; i < 1000; ++i) {
            auto m = random_orthogonal_matrix<float, 3>(rng);
            
            // Check orthogonality
            auto mt = transpose(m);
            auto prod = m * mt;
            CHECK(approx_equal(prod, matrix<float, 3, 3>::identity(), 1e-6f));
            
            // Check determinant = ±1
            float det = determinant(m);
            CHECK(std::abs(std::abs(det) - 1.0f) < 1e-6f);
            
            if (det > 0) positive_det++;
            else negative_det++;
        }
        
        // Should be roughly 50/50
        CHECK(std::abs(positive_det - negative_det) < 100);
    }
    
    SUBCASE("Symmetric matrices") {
        for (int i = 0; i < 10; ++i) {
            auto m = random_symmetric_matrix<float, 3>(rng, -1.0f, 1.0f);
            
            // Check symmetry
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    CHECK(m(r, c) == m(c, r));
                }
            }
        }
    }
}

TEST_CASE("Thread-local convenience functions") {
    SUBCASE("Angles") {
        auto angle1 = random_angle<float, degree_tag>();
        CHECK(angle1.value() >= 0.0f);
        CHECK(angle1.value() < 360.0f);
        
        auto angle2 = random_angle<float, radian_tag>();
        CHECK(angle2.value() >= 0.0f);
        CHECK(angle2.value() < 2.0f * constants<float>::pi);
    }
    
    SUBCASE("Complex") {
        auto c1 = random_complex_unit<float>();
        CHECK(std::abs(c1.abs() - 1.0f) < 1e-6f);
        
        auto c2 = random_complex<float>(2.0f);
        CHECK(std::abs(c2.real()) <= 2.0f);
        CHECK(std::abs(c2.imag()) <= 2.0f);
    }
    
    SUBCASE("Vectors") {
        auto v1 = random_unit_vector<float, 3>();
        CHECK(std::abs(v1.length() - 1.0f) < 1e-6f);
        
        auto v2 = random_vector<float, 4>(-1.0f, 1.0f);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(v2[i] >= -1.0f);
            CHECK(v2[i] <= 1.0f);
        }
    }
    
    SUBCASE("Quaternions") {
        auto q = random_quaternion<float>();
        CHECK(q.is_normalized());
    }
}