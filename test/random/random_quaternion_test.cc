#include <euler/random/random_quaternion.hh>
#include <euler/random/random.hh>
#include <euler/quaternion/quaternion.hh>
#include <euler/quaternion/quaternion_ops.hh>
#include <euler/vector/vector.hh>
#include <euler/angles/angle.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>
#include <vector>
#include <cmath>

using namespace euler;

TEST_CASE("Random Quaternion - Uniform Distribution") {
    random_generator rng(12345);
    
    SUBCASE("Unit quaternions") {
        const int N = 10000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion<float>(rng);
            CHECK(q.is_normalized(1e-6f));
        }
    }
    
    SUBCASE("Uniform coverage of SO(3)") {
        const int N = 10000;
        
        // Test that rotations are well distributed by checking
        // where they map a fixed vector
        vector<double, 3> test_vec(0, 0, 1);  // z-axis
        
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion<double>(rng);
            auto rotated = q.rotate(test_vec);
            
            sum_x += rotated[0];
            sum_y += rotated[1];
            sum_z += rotated[2];
        }
        
        // Mean should be near zero (uniform on sphere)
        CHECK(std::abs(sum_x / N) < 0.02);
        CHECK(std::abs(sum_y / N) < 0.02);
        CHECK(std::abs(sum_z / N) < 0.02);
    }
    
    SUBCASE("Component distribution") {
        const int N = 10000;
        
        // For uniform quaternions, each component should have mean 0
        // and E[w²] = E[x²] = E[y²] = E[z²] = 1/4
        float sum_w = 0.0f, sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        float sum_w2 = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f, sum_z2 = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion<float>(rng);
            
            sum_w += q.w(); sum_w2 += q.w() * q.w();
            sum_x += q.x(); sum_x2 += q.x() * q.x();
            sum_y += q.y(); sum_y2 += q.y() * q.y();
            sum_z += q.z(); sum_z2 += q.z() * q.z();
        }
        
        // Check means (should be 0)
        CHECK(std::abs(sum_w / N) < 0.02f);
        CHECK(std::abs(sum_x / N) < 0.02f);
        CHECK(std::abs(sum_y / N) < 0.02f);
        CHECK(std::abs(sum_z / N) < 0.02f);
        
        // Check second moments (should be 1/4)
        CHECK(std::abs(sum_w2 / N - 0.25f) < 0.01f);
        CHECK(std::abs(sum_x2 / N - 0.25f) < 0.01f);
        CHECK(std::abs(sum_y2 / N - 0.25f) < 0.01f);
        CHECK(std::abs(sum_z2 / N - 0.25f) < 0.01f);
    }
}

TEST_CASE("Random Quaternion - Limited Angle") {
    random_generator rng(54321);
    
    SUBCASE("Small angle limit") {
        auto max_angle = radian<float>(constants<float>::pi / 6);  // 30 degrees
        
        const int N = 1000;
        float max_observed = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion(rng, max_angle);
            CHECK(q.is_normalized(1e-5f));
            
            auto angle = q.angle();
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() <= max_angle.value() + 1e-6f);
            
            max_observed = std::max(max_observed, angle.value());
        }
        
        // Should get close to the maximum
        CHECK(max_observed > max_angle.value() * 0.95f);
    }
    
    SUBCASE("Large angle limit") {
        auto max_angle = radian<double>(constants<double>::pi * 0.9);  // 162 degrees
        
        const int N = 1000;
        int small_angles = 0;
        int large_angles = 0;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion(rng, max_angle);
            auto angle = q.angle();
            
            if (angle.value() < constants<double>::pi / 2) {
                small_angles++;
            } else {
                large_angles++;
            }
        }
        
        // Should have good mix of small and large angles
        CHECK(small_angles > 200);
        CHECK(large_angles > 200);
    }
}

TEST_CASE("Random Quaternion - Small Rotations") {
    random_generator rng(99999);
    
    SUBCASE("Very small angles") {
        float max_angle_rad = 0.01f;  // ~0.57 degrees
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_small<float>(rng, max_angle_rad);
            CHECK(q.is_normalized(1e-5f));
            
            auto angle = q.angle();
            CHECK(angle.value() >= 0.0f);
            CHECK(angle.value() <= max_angle_rad + 1e-6f);
            
            // Should be close to identity
            CHECK(std::abs(q.w()) > 0.99f);
        }
    }
    
    SUBCASE("Axis distribution") {
        float max_angle_rad = 0.1f;
        
        const int N = 10000;
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_small<double>(rng, max_angle_rad);
            auto axis = q.axis();
            
            sum_x += static_cast<float>(axis[0]);
            sum_y += static_cast<float>(axis[1]);
            sum_z += static_cast<float>(axis[2]);
        }
        
        // Axes should be uniformly distributed on sphere
        CHECK(std::abs(sum_x / N) < 0.02f);
        CHECK(std::abs(sum_y / N) < 0.02f);
        CHECK(std::abs(sum_z / N) < 0.02f);
    }
}

TEST_CASE("Random Quaternion - Normal Distribution") {
    random_generator rng(11111);
    
    SUBCASE("Small standard deviation") {
        float stddev_rad = 0.05f;
        
        const int N = 10000;
        std::vector<float> angles;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_normal<float>(rng, stddev_rad);
            CHECK(q.is_normalized(1e-5f));
            
            angles.push_back(q.angle().value());
        }
        
        // Most angles should be small
        float small_count = 0;
        for (float angle : angles) {
            if (angle < 3 * stddev_rad) {  // Within 3 sigma
                small_count++;
            }
        }
        
        CHECK(small_count > 0.97f * N);  // ~99.7% within 3 sigma (relaxed for test stability)
        
        // Check that we get some variation
        float max_angle = *std::max_element(angles.begin(), angles.end());
        CHECK(max_angle > 2 * stddev_rad);
    }
    
    SUBCASE("Exponential map accuracy") {
        float stddev_rad = 0.1f;
        
        const int N = 100;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_normal<double>(rng, stddev_rad);
            
            // For small rotations, should approximately satisfy:
            // q ≈ (1, v/2) where v is the rotation vector
            if (q.angle().value() < 0.1) {
                float expected_w = static_cast<float>(std::cos(q.angle().value() / 2));
                CHECK(std::abs(static_cast<float>(q.w()) - expected_w) < 0.01f);
            }
        }
    }
}

TEST_CASE("Random Quaternion - Fixed Axis") {
    random_generator rng(22222);
    
    SUBCASE("Rotation around z-axis") {
        vector<float, 3> z_axis(0, 0, 1);
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_axis<float>(rng, z_axis);
            CHECK(q.is_normalized(1e-5f));
            
            // Check that axis is indeed z
            auto computed_axis = q.axis();
            float dot_prod = dot(computed_axis, z_axis);
            CHECK(std::abs(std::abs(dot_prod) - 1.0f) < 0.01f);
        }
    }
    
    SUBCASE("Limited angle range") {
        vector<double, 3> axis(1, 1, 1);
        axis = normalize(axis);
        
        auto min_angle = radian<double>(constants<double>::pi / 4);
        auto max_angle = radian<double>(constants<double>::pi / 2);
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_axis(rng, axis, min_angle, max_angle);
            
            auto angle = q.angle();
            CHECK(angle.value() >= min_angle.value() - 1e-6);
            CHECK(angle.value() <= max_angle.value() + 1e-6);
            
            // Check axis
            auto computed_axis = q.axis();
            float dot_prod = static_cast<float>(dot(computed_axis, axis));
            CHECK(std::abs(std::abs(dot_prod) - 1.0f) < 0.01f);
        }
    }
}

TEST_CASE("Random Quaternion - Euler Angles") {
    random_generator rng(33333);
    
    SUBCASE("Limited Euler angles") {
        auto min_roll = degree<float>(-30);
        auto max_roll = degree<float>(30);
        auto min_pitch = degree<float>(-20);
        auto max_pitch = degree<float>(20);
        auto min_yaw = degree<float>(-45);
        auto max_yaw = degree<float>(45);
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_euler(rng,
                min_roll, max_roll,
                min_pitch, max_pitch,
                min_yaw, max_yaw,
                euler_order::XYZ);
            
            CHECK(q.is_normalized(1e-5f));
            
            // Extract Euler angles
            auto euler = q.to_euler(euler_order::XYZ);
            auto roll = to_degrees(euler[0]);
            auto pitch = to_degrees(euler[1]);
            auto yaw = to_degrees(euler[2]);
            
            // Check ranges (with some tolerance for numerical errors)
            CHECK(roll.value() >= min_roll.value() - 1.0f);
            CHECK(roll.value() <= max_roll.value() + 1.0f);
            CHECK(pitch.value() >= min_pitch.value() - 1.0f);
            CHECK(pitch.value() <= max_pitch.value() + 1.0f);
            CHECK(yaw.value() >= min_yaw.value() - 1.0f);
            CHECK(yaw.value() <= max_yaw.value() + 1.0f);
        }
    }
    
    SUBCASE("Different rotation orders") {
        auto min_angle = degree<double>(-45);
        auto max_angle = degree<double>(45);
        
        euler_order orders[] = {
            euler_order::XYZ, euler_order::XZY,
            euler_order::YXZ, euler_order::YZX,
            euler_order::ZXY, euler_order::ZYX
        };
        
        for (auto order : orders) {
            auto q = random_quaternion_euler<double, degree_tag>(rng,
                min_angle, max_angle,
                min_angle, max_angle,
                min_angle, max_angle,
                order);
            
            CHECK(q.is_normalized(1e-5f));
            
            // Should be able to extract angles back
            auto euler = q.to_euler(order);
            CHECK(std::isfinite(euler[0].value()));
            CHECK(std::isfinite(euler[1].value()));
            CHECK(std::isfinite(euler[2].value()));
        }
    }
}

TEST_CASE("Random Quaternion - Constrained") {
    random_generator rng(44444);
    
    SUBCASE("Forward direction constraint") {
        vector<float, 3> target_forward(1, 0, 0);  // X-axis
        float max_deviation_rad = 0.2f;  // ~11.5 degrees
        
        const int N = 1000;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_constrained<float>(rng, 
                                                         target_forward, 
                                                         max_deviation_rad);
            CHECK(q.is_normalized(1e-5f));
            
            // Check that default forward (0,0,1) is rotated close to target
            vector<float, 3> default_forward(0, 0, 1);
            auto rotated_forward = q.rotate(default_forward);
            
            float dot_prod = dot(rotated_forward, target_forward);
            float angle = std::acos(std::min(1.0f, dot_prod));
            
            CHECK(angle <= max_deviation_rad + 1e-5f);
        }
    }
    
    SUBCASE("Distribution of deviations") {
        vector<double, 3> target_forward(0, 1, 0);  // Y-axis
        double max_deviation_rad = 0.5;  // ~28.6 degrees
        
        const int N = 10000;
        std::vector<double> deviations;
        
        for (int i = 0; i < N; ++i) {
            auto q = random_quaternion_constrained<double>(rng, 
                                                          target_forward, 
                                                          max_deviation_rad);
            
            vector<double, 3> default_forward(0, 0, 1);
            auto rotated_forward = q.rotate(default_forward);
            
            double dot_prod = dot(rotated_forward, target_forward);
            double angle = std::acos(std::min(1.0, dot_prod));
            deviations.push_back(angle);
        }
        
        // Should have good coverage of allowed range
        auto max_dev = *std::max_element(deviations.begin(), deviations.end());
        CHECK(max_dev > max_deviation_rad * 0.9);
        
        // But all within limit
        for (double dev : deviations) {
            CHECK(dev <= max_deviation_rad + 1e-5);
        }
    }
}

TEST_CASE("Random Quaternion - Distributed") {
    random_generator rng(55555);
    
    SUBCASE("Uniform distribution (clustering = 1)") {
        auto quats = random_quaternions_distributed<float>(rng, 1000, 1.0f);
        
        CHECK(quats.size() == 1000);
        
        // Should behave like uniform distribution
        float sum_angle = 0.0f;
        for (const auto& q : quats) {
            CHECK(q.is_normalized(1e-5f));
            sum_angle += q.angle().value();
        }
        
        float mean_angle = sum_angle / static_cast<float>(quats.size());
        // Mean angle for uniform quaternions is close to pi
        CHECK(std::abs(mean_angle - constants<float>::pi) < 0.3f);
    }
    
    SUBCASE("Clustered distribution") {
        auto quats = random_quaternions_distributed<double>(rng, 1000, 3.0);
        
        // Should be more clustered around identity
        int small_angles = 0;
        for (const auto& q : quats) {
            if (q.angle().value() < constants<double>::pi / 2) {
                small_angles++;
            }
        }
        
        CHECK(small_angles > 700);  // Most should be small rotations
    }
    
    SUBCASE("Spread distribution") {
        auto quats = random_quaternions_distributed<float>(rng, 1000, 0.5f);
        
        // Should be more spread out
        int large_angles = 0;
        for (const auto& q : quats) {
            if (q.angle().value() > constants<float>::pi / 2) {
                large_angles++;
            }
        }
        
        CHECK(large_angles > 200);  // More large rotations
    }
}

TEST_CASE("Random Quaternion - Thread-local Functions") {
    SUBCASE("All convenience functions") {
        // Uniform
        auto q1 = random_quaternion<float>();
        CHECK(q1.is_normalized());
        
        // Limited angle
        auto q2 = random_quaternion<double>(radian<double>(1.0));
        CHECK(q2.angle().value() <= 1.0);
        
        // Small rotation
        auto q3 = random_quaternion_small<float>(0.05f);
        CHECK(q3.angle().value() <= 0.05f);
        
        // Normal distribution
        auto q4 = random_quaternion_normal<double>(0.1);
        CHECK(q4.is_normalized());
    }
}

TEST_CASE("Random Quaternion - Statistical Properties") {
    random_generator rng(66666);
    
    SUBCASE("Rotation composition") {
        // Test that composition of random rotations increases variance
        const int N = 1000;
        
        std::vector<float> single_angles;
        std::vector<float> composed_angles;
        
        for (int i = 0; i < N; ++i) {
            auto q1 = random_quaternion_small<float>(rng, 0.1f);
            auto q2 = random_quaternion_small<float>(rng, 0.1f);
            
            auto composed = q1 * q2;
            
            single_angles.push_back(q1.angle().value());
            composed_angles.push_back(composed.angle().value());
        }
        
        // Compute variances
        float mean1 = 0.0f, mean2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            mean1 += single_angles[static_cast<size_t>(i)];
            mean2 += composed_angles[static_cast<size_t>(i)];
        }
        mean1 /= N;
        mean2 /= N;
        
        float var1 = 0.0f, var2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            var1 += (single_angles[static_cast<size_t>(i)] - mean1) * (single_angles[static_cast<size_t>(i)] - mean1);
            var2 += (composed_angles[static_cast<size_t>(i)] - mean2) * (composed_angles[static_cast<size_t>(i)] - mean2);
        }
        
        // Composed rotations should have larger variance
        CHECK(var2 > var1);
    }
}

TEST_CASE("Quaternion from_euler XZY bug") {
    using namespace euler;
    
    SUBCASE("XZY order produces non-normalized quaternions") {
        // Test specific angles that demonstrate the bug
        auto roll = radian<float>(constants<float>::pi / 4);   // X rotation
        auto pitch = radian<float>(constants<float>::pi / 3);  // Y rotation
        auto yaw = radian<float>(constants<float>::pi / 6);    // Z rotation
        
        auto q = quaternion<float>::from_euler(roll, pitch, yaw, euler_order::XZY);
        
        // Check if quaternion is normalized
        float norm_sq = q.norm_squared();
        INFO("Quaternion from XZY euler angles: ", q.w(), ", ", q.x(), ", ", q.y(), ", ", q.z());
        INFO("Norm squared: ", norm_sq);
        
        // This should fail, demonstrating the bug
        CHECK(q.is_normalized(1e-5f));
    }
    
    SUBCASE("Compare XZY with composition of individual rotations") {
        // Create individual rotation quaternions
        auto roll = radian<float>(constants<float>::pi / 4);   // X rotation
        auto pitch = radian<float>(constants<float>::pi / 3);  // Y rotation  
        auto yaw = radian<float>(constants<float>::pi / 6);    // Z rotation
        
        // Individual rotations
        auto qx = quaternion<float>::from_axis_angle(vector<float, 3>::unit_x(), roll);
        auto qy = quaternion<float>::from_axis_angle(vector<float, 3>::unit_y(), pitch);
        auto qz = quaternion<float>::from_axis_angle(vector<float, 3>::unit_z(), yaw);
        
        // XZY order means: first X, then Z, then Y
        auto q_composed = qy * qz * qx;  // quaternion multiplication is right-to-left
        
        auto q_euler = quaternion<float>::from_euler(roll, pitch, yaw, euler_order::XZY);
        
        INFO("Composed quaternion: ", q_composed.w(), ", ", q_composed.x(), ", ", q_composed.y(), ", ", q_composed.z());
        INFO("from_euler quaternion: ", q_euler.w(), ", ", q_euler.x(), ", ", q_euler.y(), ", ", q_euler.z());
        
        // The composed quaternion should be normalized
        CHECK(q_composed.is_normalized(1e-5f));
        
        // Check if they produce the same rotation (if normalized)
        auto q_euler_normalized = q_euler.normalized();
        CHECK(approx_equal(q_composed.w(), q_euler_normalized.w(), 1e-5f));
        CHECK(approx_equal(q_composed.x(), q_euler_normalized.x(), 1e-5f));
        CHECK(approx_equal(q_composed.y(), q_euler_normalized.y(), 1e-5f));
        CHECK(approx_equal(q_composed.z(), q_euler_normalized.z(), 1e-5f));
    }
}