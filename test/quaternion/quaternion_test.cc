#include <euler/quaternion/quaternion.hh>
#include <euler/quaternion/quaternion_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;
using euler::normalize;  // For vector normalize
using mat3f = matrix<float, 3, 3>;
using mat4f = matrix<float, 4, 4>;

TEST_CASE("Quaternion construction and basic properties") {
    SUBCASE("Default constructor creates identity") {
        quatf q;
        CHECK(q.w() == 1.0f);
        CHECK(q.x() == 0.0f);
        CHECK(q.y() == 0.0f);
        CHECK(q.z() == 0.0f);
        CHECK(q.is_normalized());
    }
    
    SUBCASE("Component constructor") {
        quatf q(0.5f, 0.5f, 0.5f, 0.5f);
        CHECK(q.w() == 0.5f);
        CHECK(q.x() == 0.5f);
        CHECK(q.y() == 0.5f);
        CHECK(q.z() == 0.5f);
        CHECK(q.is_normalized());
    }
    
    SUBCASE("Identity factory method") {
        quatf q = quatf::identity();
        CHECK(q.w() == 1.0f);
        CHECK(q.x() == 0.0f);
        CHECK(q.y() == 0.0f);
        CHECK(q.z() == 0.0f);
    }
    
    SUBCASE("Array access") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(q[0] == 1.0f);  // w
        CHECK(q[1] == 2.0f);  // x
        CHECK(q[2] == 3.0f);  // y
        CHECK(q[3] == 4.0f);  // z
    }
    
    SUBCASE("Vector and scalar parts") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(q.scalar() == 1.0f);
        vec3f v = q.vec();
        CHECK(v[0] == 2.0f);
        CHECK(v[1] == 3.0f);
        CHECK(v[2] == 4.0f);
    }
}

TEST_CASE("Quaternion from axis-angle") {
    SUBCASE("90 degree rotation around Y axis") {
        vec3f axis = vec3f::unit_y();
        quatf q = quatf::from_axis_angle(axis, 90.0_deg);
        
        // Expected: w = cos(45°) ≈ 0.707, x = 0, y = sin(45°) ≈ 0.707, z = 0
        CHECK(approx_equal(q.w(), std::cos(constants<float>::pi / 4), 1e-6f));
        CHECK(approx_equal(q.x(), 0.0f, 1e-6f));
        CHECK(approx_equal(q.y(), std::sin(constants<float>::pi / 4), 1e-6f));
        CHECK(approx_equal(q.z(), 0.0f, 1e-6f));
        CHECK(q.is_normalized());
    }
    
    SUBCASE("180 degree rotation around Z axis") {
        vec3f axis = vec3f::unit_z();
        quatf q = quatf::from_axis_angle(axis, radian<float>(constants<float>::pi));
        
        // Expected: w = 0, x = 0, y = 0, z = 1
        CHECK(approx_equal(q.w(), 0.0f, 1e-6f));
        CHECK(approx_equal(q.x(), 0.0f, 1e-6f));
        CHECK(approx_equal(q.y(), 0.0f, 1e-6f));
        CHECK(approx_equal(q.z(), 1.0f, 1e-6f));
    }
    
    SUBCASE("Arbitrary axis and angle") {
        vec3f axis = normalize(vec3f(1.0f, 1.0f, 1.0f));
        quatf q = quatf::from_axis_angle(axis, 120.0_deg);
        
        // Verify it's normalized
        CHECK(q.is_normalized());
        
        // Extract back and verify
        auto [extracted_axis, extracted_angle] = q.to_axis_angle();
        CHECK(approx_equal(extracted_axis, axis, 1e-6f));
        CHECK(approx_equal(extracted_angle.value(), radian<float>(120.0_deg).value(), 1e-6f));
    }
}

TEST_CASE("Quaternion from Euler angles") {
    SUBCASE("Simple XYZ rotation") {
        quatf q = quatf::from_euler(30.0_deg, 45.0_deg, 60.0_deg, euler_order::XYZ);
        CHECK(q.is_normalized());
        
        // Convert back to Euler angles
        vec3<radian<float>> angles = q.to_euler(euler_order::XYZ);
        CHECK(approx_equal(angles[0].value(), radian<float>(30.0_deg).value(), 1e-5f));
        CHECK(approx_equal(angles[1].value(), radian<float>(45.0_deg).value(), 1e-5f));
        CHECK(approx_equal(angles[2].value(), radian<float>(60.0_deg).value(), 1e-5f));
    }
    
    SUBCASE("Different rotation orders") {
        degree<float> roll = 20.0_deg;
        degree<float> pitch = 30.0_deg;
        degree<float> yaw = 40.0_deg;
        
        // Different orders should produce different quaternions
        quatf q_xyz = quatf::from_euler(roll, pitch, yaw, euler_order::XYZ);
        quatf q_zyx = quatf::from_euler(roll, pitch, yaw, euler_order::ZYX);
        
        CHECK(!approx_equal(q_xyz, q_zyx, 1e-6f));
        CHECK(q_xyz.is_normalized());
        CHECK(q_zyx.is_normalized());
    }
}

TEST_CASE("Quaternion from rotation matrix") {
    SUBCASE("Identity matrix") {
        mat3f m = mat3f::identity();
        quatf q = quatf::from_matrix(m);
        CHECK(approx_equal(q, quatf::identity(), 1e-6f));
    }
    
    SUBCASE("90 degree rotation around Y") {
        // Create rotation matrix for 90° around Y
        float c = 0.0f;  // cos(90°)
        float s = 1.0f;  // sin(90°)
        mat3f m;
        m(0, 0) = c;   m(0, 1) = 0;  m(0, 2) = s;
        m(1, 0) = 0;   m(1, 1) = 1;  m(1, 2) = 0;
        m(2, 0) = -s;  m(2, 1) = 0;  m(2, 2) = c;
        
        quatf q = quatf::from_matrix(m);
        
        // Verify by converting back
        mat3f m2 = q.to_matrix3();
        CHECK(approx_equal(m, m2, 1e-6f));
    }
    
    SUBCASE("4x4 matrix extraction") {
        mat4f m4 = mat4f::identity();
        // Set up a rotation in the upper-left 3x3
        m4(0, 0) = 0;  m4(0, 1) = 0;  m4(0, 2) = 1;
        m4(1, 0) = 1;  m4(1, 1) = 0;  m4(1, 2) = 0;
        m4(2, 0) = 0;  m4(2, 1) = 1;  m4(2, 2) = 0;
        
        quatf q = quatf::from_matrix(m4);
        CHECK(q.is_normalized());
    }
}

TEST_CASE("Quaternion from vectors") {
    SUBCASE("Parallel vectors") {
        vec3f from = vec3f::unit_x();
        vec3f to = vec3f::unit_x();
        quatf q = quatf::from_vectors(from, to);
        CHECK(approx_equal(q, quatf::identity(), 1e-6f));
    }
    
    SUBCASE("Anti-parallel vectors") {
        vec3f from = vec3f::unit_x();
        vec3f to = -vec3f::unit_x();
        quatf q = quatf::from_vectors(from, to);
        
        // Should be 180° rotation around any perpendicular axis
        CHECK(approx_equal(q.angle().value(), constants<float>::pi, 1e-6f));
        vec3f rotated = q.rotate(from);
        CHECK(approx_equal(rotated, to, 1e-6f));
    }
    
    SUBCASE("90 degree rotation") {
        vec3f from = vec3f::unit_x();
        vec3f to = vec3f::unit_y();
        quatf q = quatf::from_vectors(from, to);
        
        // Should rotate from X to Y
        vec3f rotated = q.rotate(from);
        CHECK(approx_equal(rotated, to, 1e-6f));
        
        // Angle should be 90°
        CHECK(approx_equal(q.angle().value(), constants<float>::pi / 2, 1e-6f));
    }
}

TEST_CASE("Quaternion normalization") {
    SUBCASE("Normalize non-unit quaternion") {
        quatf q(2.0f, 0.0f, 0.0f, 0.0f);
        CHECK(!q.is_normalized());
        
        quatf q_norm = q.normalized();
        CHECK(q_norm.is_normalized());
        CHECK(approx_equal(q_norm.w(), 1.0f, 1e-6f));
    }
    
    SUBCASE("In-place normalization") {
        quatf q(1.0f, 1.0f, 1.0f, 1.0f);
        q.normalize();
        CHECK(q.is_normalized());
        CHECK(approx_equal(q.length(), 1.0f, 1e-6f));
    }
}

TEST_CASE("Quaternion properties") {
    SUBCASE("Norm and length") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        float expected_norm_sq = 1 + 4 + 9 + 16;  // 30
        CHECK(approx_equal(q.norm_squared(), expected_norm_sq, 1e-6f));
        CHECK(approx_equal(q.length(), sqrt(expected_norm_sq), 1e-6f));
    }
    
    SUBCASE("Pure quaternion check") {
        quatf q1(0.0f, 1.0f, 2.0f, 3.0f);
        CHECK(q1.is_pure());
        
        quatf q2(0.1f, 1.0f, 2.0f, 3.0f);
        CHECK(!q2.is_pure());
    }
}

TEST_CASE("Vector rotation") {
    SUBCASE("90 degree rotation around Y axis") {
        quatf q = quatf::from_axis_angle(vec3f::unit_y(), 90.0_deg);
        vec3f v = vec3f::unit_x();
        vec3f rotated = q.rotate(v);
        
        // X rotated 90° around Y should give -Z
        CHECK(approx_equal(rotated, -vec3f::unit_z(), 1e-6f));
    }
    
    SUBCASE("180 degree rotation around arbitrary axis") {
        vec3f axis = normalize(vec3f(1.0f, 1.0f, 0.0f));
        quatf q = quatf::from_axis_angle(axis, 180.0_deg);
        vec3f v(1.0f, 0.0f, 0.0f);
        vec3f rotated = q.rotate(v);
        
        // Verify the rotation
        vec3f rotated_back = q.rotate(rotated);
        CHECK(approx_equal(rotated_back, v, 1e-6f));
    }
    
    SUBCASE("Operator* for vector rotation") {
        quatf q = quatf::from_axis_angle(vec3f::unit_z(), 45.0_deg);
        vec3f v = vec3f::unit_x();
        vec3f rotated1 = q.rotate(v);
        vec3f rotated2 = q * v;
        
        CHECK(approx_equal(rotated1, rotated2, 1e-6f));
    }
}

TEST_CASE("Matrix conversion") {
    SUBCASE("Identity quaternion to matrix") {
        quatf q = quatf::identity();
        mat3f m3 = q.to_matrix3();
        mat4f m4 = q.to_matrix4();
        
        CHECK(approx_equal(m3, mat3f::identity(), 1e-6f));
        CHECK(approx_equal(m4, mat4f::identity(), 1e-6f));
    }
    
    SUBCASE("Round-trip conversion") {
        quatf q = quatf::from_axis_angle(
            normalize(vec3f(1.0f, 2.0f, 3.0f)), 
            75.0_deg
        );
        
        mat3f m = q.to_matrix3();
        quatf q2 = quatf::from_matrix(m);
        
        // Note: q and -q represent the same rotation
        CHECK(same_rotation(q, q2, 1e-6f));
    }
}

TEST_CASE("Quaternion arithmetic") {
    quatf q1(1.0f, 2.0f, 3.0f, 4.0f);
    quatf q2(5.0f, 6.0f, 7.0f, 8.0f);
    
    SUBCASE("Addition and subtraction") {
        quatf sum = q1 + q2;
        CHECK(sum.w() == 6.0f);
        CHECK(sum.x() == 8.0f);
        CHECK(sum.y() == 10.0f);
        CHECK(sum.z() == 12.0f);
        
        quatf diff = q2 - q1;
        CHECK(diff.w() == 4.0f);
        CHECK(diff.x() == 4.0f);
        CHECK(diff.y() == 4.0f);
        CHECK(diff.z() == 4.0f);
    }
    
    SUBCASE("Scalar multiplication and division") {
        quatf scaled = q1 * 2.0f;
        CHECK(scaled.w() == 2.0f);
        CHECK(scaled.x() == 4.0f);
        CHECK(scaled.y() == 6.0f);
        CHECK(scaled.z() == 8.0f);
        
        quatf divided = q1 / 2.0f;
        CHECK(divided.w() == 0.5f);
        CHECK(divided.x() == 1.0f);
        CHECK(divided.y() == 1.5f);
        CHECK(divided.z() == 2.0f);
    }
    
    SUBCASE("Unary negation") {
        quatf neg = -q1;
        CHECK(neg.w() == -1.0f);
        CHECK(neg.x() == -2.0f);
        CHECK(neg.y() == -3.0f);
        CHECK(neg.z() == -4.0f);
    }
}

TEST_CASE("Quaternion operations") {
    SUBCASE("Conjugate") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf conj = conjugate(q);
        CHECK(conj.w() == 1.0f);
        CHECK(conj.x() == -2.0f);
        CHECK(conj.y() == -3.0f);
        CHECK(conj.z() == -4.0f);
    }
    
    SUBCASE("Inverse") {
        quatf q = quatf::from_axis_angle(vec3f::unit_y(), 45.0_deg);
        quatf inv = inverse(q);
        quatf product = q * inv;
        CHECK(approx_equal(product, quatf::identity(), 1e-6f));
    }
    
    SUBCASE("Hamilton product") {
        // Test with known result
        quatf q1(1.0f, 0.0f, 0.0f, 0.0f);  // Identity
        quatf q2(0.0f, 1.0f, 0.0f, 0.0f);  // Pure i
        quatf product = q1 * q2;
        CHECK(product.w() == 0.0f);
        CHECK(product.x() == 1.0f);
        CHECK(product.y() == 0.0f);
        CHECK(product.z() == 0.0f);
        
        // Test quaternion multiplication properties
        quatf q3 = quatf::from_axis_angle(vec3f::unit_x(), 30.0_deg);
        quatf q4 = quatf::from_axis_angle(vec3f::unit_x(), 60.0_deg);
        quatf combined = q3 * q4;
        
        // Should be equivalent to 90° rotation around X
        quatf expected = quatf::from_axis_angle(vec3f::unit_x(), 90.0_deg);
        CHECK(approx_equal(combined, expected, 1e-6f));
    }
    
    SUBCASE("Dot product") {
        quatf q1(1.0f, 2.0f, 3.0f, 4.0f);
        quatf q2(5.0f, 6.0f, 7.0f, 8.0f);
        float d = dot(q1, q2);
        float expected = 1*5 + 2*6 + 3*7 + 4*8;  // 70
        CHECK(approx_equal(d, expected, 1e-6f));
    }
}

TEST_CASE("Quaternion interpolation") {
    SUBCASE("Linear interpolation (lerp)") {
        quatf q1 = quatf::identity();
        quatf q2 = quatf::from_axis_angle(vec3f::unit_y(), 90.0_deg);
        
        quatf mid = lerp(q1, q2, 0.5f);
        CHECK(mid.is_normalized());
        
        // Check endpoints
        CHECK(approx_equal(lerp(q1, q2, 0.0f), q1, 1e-6f));
        CHECK(approx_equal(lerp(q1, q2, 1.0f), q2, 1e-6f));
    }
    
    SUBCASE("Spherical linear interpolation (slerp)") {
        quatf q1 = quatf::from_axis_angle(vec3f::unit_z(), 0.0_deg);
        quatf q2 = quatf::from_axis_angle(vec3f::unit_z(), 90.0_deg);
        
        // Interpolate to 45°
        quatf mid = slerp(q1, q2, 0.5f);
        CHECK(mid.is_normalized());
        
        // Should be 45° rotation around Z
        radian<float> angle = mid.angle();
        CHECK(approx_equal(angle.value(), constants<float>::pi / 4, 1e-5f));
        
        // Check endpoints
        CHECK(approx_equal(slerp(q1, q2, 0.0f), q1, 1e-6f));
        CHECK(approx_equal(slerp(q1, q2, 1.0f), q2, 1e-6f));
    }
    
    SUBCASE("Slerp shortest path") {
        quatf q1 = quatf::identity();
        quatf q2 = -quatf::identity();  // Same rotation, opposite sign
        
        quatf result = slerp(q1, q2, 0.5f);
        CHECK(result.is_normalized());
        
        // Should interpolate to identity (shortest path)
        CHECK(approx_equal(result, q1, 1e-5f));
    }
}

TEST_CASE("Quaternion angle operations") {
    SUBCASE("Angle between quaternions") {
        quatf q1 = quatf::from_axis_angle(vec3f::unit_y(), 30.0_deg);
        quatf q2 = quatf::from_axis_angle(vec3f::unit_y(), 75.0_deg);
        
        radian<float> angle = angle_between(q1, q2);
        CHECK(approx_equal(angle.value(), radian<float>(45.0_deg).value(), 1e-5f));
    }
    
    SUBCASE("Same rotation check") {
        quatf q1(0.5f, 0.5f, 0.5f, 0.5f);
        quatf q2 = -q1;  // Opposite signs
        
        CHECK(same_rotation(q1, q2));
        CHECK(!approx_equal(q1, q2));  // Different values
    }
}

TEST_CASE("Quaternion exponential and logarithm") {
    SUBCASE("Exp and log of identity") {
        quatf q = quatf::identity();
        quatf log_q = log(q);
        CHECK(approx_equal(log_q.w(), 0.0f, 1e-6f));
        CHECK(approx_equal(log_q.x(), 0.0f, 1e-6f));
        CHECK(approx_equal(log_q.y(), 0.0f, 1e-6f));
        CHECK(approx_equal(log_q.z(), 0.0f, 1e-6f));
        
        quatf exp_log_q = exp(log_q);
        CHECK(approx_equal(exp_log_q, q, 1e-6f));
    }
    
    SUBCASE("Power function") {
        quatf q = quatf::from_axis_angle(vec3f::unit_z(), 60.0_deg);
        
        // q^0 should be identity
        CHECK(approx_equal(pow(q, 0.0f), quatf::identity(), 1e-6f));
        
        // q^1 should be q
        CHECK(approx_equal(pow(q, 1.0f), q, 1e-6f));
        
        // q^2 should be 120° rotation
        quatf q2 = pow(q, 2.0f);
        CHECK(approx_equal(q2.angle().value(), radian<float>(120.0_deg).value(), 1e-5f));
        
        // q^0.5 should be 30° rotation
        quatf q_half = pow(q, 0.5f);
        CHECK(approx_equal(q_half.angle().value(), radian<float>(30.0_deg).value(), 1e-5f));
    }
}