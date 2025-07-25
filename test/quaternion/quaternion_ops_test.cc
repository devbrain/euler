#include <euler/quaternion/quaternion.hh>
#include <euler/quaternion/quaternion_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <doctest/doctest.h>
#include <random>
#include <cmath>

using namespace euler;

TEST_CASE("Quaternion multiplication properties") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Generate random normalized quaternions
    auto random_quat = [&]() {
        quatf q(dist(rng), dist(rng), dist(rng), dist(rng));
        return q.normalized();
    };
    
    SUBCASE("Non-commutativity") {
        quatf p = random_quat();
        quatf q = random_quat();
        
        quatf pq = p * q;
        quatf qp = q * p;
        
        // Generally p*q ≠ q*p
        CHECK(!approx_equal(pq, qp, 1e-6f));
    }
    
    SUBCASE("Associativity") {
        quatf p = random_quat();
        quatf q = random_quat();
        quatf r = random_quat();
        
        quatf pq_r = (p * q) * r;
        quatf p_qr = p * (q * r);
        
        CHECK(approx_equal(pq_r, p_qr, 1e-6f));
    }
    
    SUBCASE("Identity element") {
        quatf q = random_quat();
        quatf id = quatf::identity();
        
        CHECK(approx_equal(q * id, q, 1e-6f));
        CHECK(approx_equal(id * q, q, 1e-6f));
    }
    
    SUBCASE("Inverse property") {
        quatf q = random_quat();
        quatf q_inv = inverse(q);
        
        CHECK(approx_equal(q * q_inv, quatf::identity(), 1e-6f));
        CHECK(approx_equal(q_inv * q, quatf::identity(), 1e-6f));
    }
}

TEST_CASE("Quaternion division") {
    SUBCASE("Division is multiplication by inverse") {
        quatf p = quatf::from_axis_angle(vec3f::unit_x(), 30.0_deg);
        quatf q = quatf::from_axis_angle(vec3f::unit_y(), 45.0_deg);
        
        quatf div1 = p / q;
        quatf div2 = p * inverse(q);
        
        CHECK(approx_equal(div1, div2, 1e-6f));
    }
    
    SUBCASE("Self division gives identity") {
        quatf q = quatf::from_axis_angle(vec3f::unit_z(), 60.0_deg);
        quatf result = q / q;
        
        CHECK(approx_equal(result, quatf::identity(), 1e-6f));
    }
}

TEST_CASE("Squad interpolation") {
    SUBCASE("Squad through multiple quaternions") {
        // Create a sequence of rotations
        quatf q0 = quatf::from_axis_angle(vec3f::unit_y(), 0.0_deg);
        quatf q1 = quatf::from_axis_angle(vec3f::unit_y(), 30.0_deg);
        quatf q2 = quatf::from_axis_angle(vec3f::unit_y(), 60.0_deg);
        quatf q3 = quatf::from_axis_angle(vec3f::unit_y(), 90.0_deg);
        
        // Compute intermediate quaternions
        quatf a = squad_intermediate(q0, q1, q2);
        quatf b = squad_intermediate(q1, q2, q3);
        
        // Interpolate with squad
        quatf result = squad(q1, a, b, q2, 0.5f);
        
        CHECK(result.is_normalized());
        
        // Should be smooth interpolation between q1 and q2
        float angle = result.angle().value();
        CHECK(angle > radian<float>(30.0_deg).value());
        CHECK(angle < radian<float>(60.0_deg).value());
    }
}

TEST_CASE("Complex quaternion operations") {
    using biquatf = biquaternion<float>;
    using complexf = complex<float>;
    
    SUBCASE("Biquaternion construction") {
        biquatf bq(
            complexf(1.0f, 0.5f),
            complexf(0.0f, 1.0f),
            complexf(2.0f, -1.0f),
            complexf(0.0f, 0.0f)
        );
        
        CHECK(bq.w().real() == 1.0f);
        CHECK(bq.w().imag() == 0.5f);
        CHECK(bq.x().real() == 0.0f);
        CHECK(bq.x().imag() == 1.0f);
    }
    
    SUBCASE("Biquaternion conjugate") {
        biquatf bq(
            complexf(1.0f, 2.0f),
            complexf(3.0f, 4.0f),
            complexf(5.0f, 6.0f),
            complexf(7.0f, 8.0f)
        );
        
        biquatf conj = conjugate(bq);
        
        CHECK(conj.w() == complexf(1.0f, 2.0f));
        CHECK(conj.x() == complexf(-3.0f, -4.0f));
        CHECK(conj.y() == complexf(-5.0f, -6.0f));
        CHECK(conj.z() == complexf(-7.0f, -8.0f));
    }
}

TEST_CASE("Edge cases and error handling") {
    SUBCASE("Zero quaternion normalization throws") {
        quatf zero(0.0f, 0.0f, 0.0f, 0.0f);
        CHECK_THROWS(zero.normalize());
        CHECK_THROWS(inverse(zero));
        CHECK_THROWS(log(zero));
    }
    
    SUBCASE("Near-identity slerp uses lerp") {
        quatf q1 = quatf::identity();
        quatf q2 = quatf::from_axis_angle(vec3f::unit_y(), 0.001_deg);
        
        // Should use lerp internally for efficiency
        quatf result = slerp(q1, q2, 0.5f);
        CHECK(result.is_normalized());
        
        // Compare with actual lerp
        quatf lerp_result = lerp(q1, q2, 0.5f);
        CHECK(approx_equal(result, lerp_result, 1e-5f));
    }
    
    SUBCASE("Interpolation parameter validation") {
        quatf q1 = quatf::identity();
        quatf q2 = quatf::from_axis_angle(vec3f::unit_x(), 45.0_deg);
        
        CHECK_THROWS(lerp(q1, q2, -0.1f));
        CHECK_THROWS(lerp(q1, q2, 1.1f));
        CHECK_THROWS(slerp(q1, q2, -0.1f));
        CHECK_THROWS(slerp(q1, q2, 1.1f));
    }
}

TEST_CASE("Quaternion numerical stability") {
    SUBCASE("Repeated normalization") {
        quatf q(1.0f, 1.0f, 1.0f, 1.0f);
        q.normalize();
        
        // Normalize many times - should remain stable
        for (int i = 0; i < 100; ++i) {
            q.normalize();
        }
        
        CHECK(q.is_normalized());
        CHECK(approx_equal(q.length(), 1.0f, 1e-6f));
    }
    
    SUBCASE("Small angle axis extraction") {
        // Very small rotation
        quatf q = quatf::from_axis_angle(vec3f::unit_z(), 0.001_deg);
        
        vec3f axis = q.axis();
        radian<float> angle = q.angle();
        
        // Should still extract reasonable values
        CHECK(approx_equal(axis, vec3f::unit_z(), 1e-3f));
        CHECK(approx_equal(angle.value(), radian<float>(0.001_deg).value(), 1e-6f));
    }
    
    SUBCASE("Near 180 degree rotation") {
        // Just under 180 degrees
        quatf q = quatf::from_axis_angle(vec3f::unit_x(), 179.99_deg);
        
        vec3f axis = q.axis();
        radian<float> angle = q.angle();
        
        CHECK(approx_equal(axis, vec3f::unit_x(), 1e-5f));
        CHECK(approx_equal(angle.value(), radian<float>(179.99_deg).value(), 1e-5f));
    }
}

TEST_CASE("Quaternion performance patterns") {
    SUBCASE("Efficient angle check without full extraction") {
        quatf q1 = quatf::from_axis_angle(vec3f::unit_y(), 45.0_deg);
        quatf q2 = quatf::from_axis_angle(vec3f::unit_y(), 46.0_deg);
        
        // Can check if quaternions are similar using dot product
        float d = dot(q1, q2);
        
        // cos(angle between quaternions) ≈ dot product for unit quaternions
        CHECK(d > 0.99f);  // Very similar
    }
    
    SUBCASE("Power of 2 optimization") {
        quatf q = quatf::from_axis_angle(vec3f::unit_z(), 30.0_deg);
        
        // q^2 can be computed as q * q
        quatf q2_pow = pow(q, 2.0f);
        quatf q2_mul = q * q;
        
        CHECK(approx_equal(q2_pow, q2_mul, 1e-6f));
    }
}

TEST_CASE("Gimbal lock avoidance") {
    SUBCASE("Euler angle singularities") {
        // Test gimbal lock configuration (pitch = ±90°)
        quatf q = quatf::from_euler(30.0_deg, 90.0_deg, 45.0_deg, euler_order::XYZ);
        
        // Should still produce valid quaternion
        CHECK(q.is_normalized());
        
        // Rotation should still work correctly
        vec3f v = vec3f::unit_x();
        vec3f rotated = q.rotate(v);
        CHECK(rotated.length_squared() == doctest::Approx(1.0f));
    }
}

TEST_CASE("Rotation composition") {
    SUBCASE("Sequential rotations") {
        // Rotate 45° around X, then 30° around Y
        quatf rx = quatf::from_axis_angle(vec3f::unit_x(), 45.0_deg);
        quatf ry = quatf::from_axis_angle(vec3f::unit_y(), 30.0_deg);
        
        // Combined rotation (note order: ry * rx for Y then X)
        quatf combined = ry * rx;
        
        // Apply to vector
        vec3f v = vec3f::unit_z();
        vec3f result1 = combined.rotate(v);
        
        // Should be same as applying separately
        vec3f temp = rx.rotate(v);
        vec3f result2 = ry.rotate(temp);
        
        CHECK(approx_equal(result1, result2, 1e-6f));
    }
    
    SUBCASE("Rotation accumulation") {
        // Accumulate many small rotations
        quatf accumulated = quatf::identity();
        const int steps = 36;
        degree<float> step_angle = 10.0_deg;
        
        for (int i = 0; i < steps; ++i) {
            quatf step = quatf::from_axis_angle(vec3f::unit_z(), step_angle);
            accumulated = accumulated * step;
        }
        
        // Should be 360° rotation (back to identity)
        CHECK(same_rotation(accumulated, quatf::identity(), 1e-5f));
    }
}