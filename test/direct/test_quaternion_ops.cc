/**
 * @file test_quaternion_ops.cc
 * @brief Unit tests for direct SIMD quaternion operations
 */

#include <doctest/doctest.h>
#include <euler/direct/quaternion_ops.hh>
#include <euler/quaternion/quaternion.hh>
#include <euler/quaternion/quaternion_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/core/approx_equal.hh>
#include <random>

using namespace euler;
using namespace euler::direct;

// Test configuration
constexpr float FLOAT_TOL = 1e-5f;

// Helper to generate random quaternions
template<typename T>
class RandomQuaternionGenerator {
public:
    RandomQuaternionGenerator(T min_val = -1, T max_val = 1) 
        : gen(std::random_device{}()), dist(min_val, max_val) {}
    
    quaternion<T> generate() {
        // Generate random quaternion and normalize it
        quaternion<T> q(dist(gen), dist(gen), dist(gen), dist(gen));
        return q.normalized();
    }
    
    quaternion<T> generate_unnormalized() {
        return quaternion<T>(dist(gen), dist(gen), dist(gen), dist(gen));
    }
    
private:
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
};

// =============================================================================
// Basic Operations Tests
// =============================================================================

TEST_CASE("Direct quaternion addition") {
    RandomQuaternionGenerator<float> rng_f;
    RandomQuaternionGenerator<double> rng_d;
    
    SUBCASE("Basic addition - quatf") {
        quatf a(1.0f, 2.0f, 3.0f, 4.0f);
        quatf b(5.0f, 6.0f, 7.0f, 8.0f);
        quatf result;
        
        add(a, b, result);
        
        CHECK(result.w() == doctest::Approx(6.0f));
        CHECK(result.x() == doctest::Approx(8.0f));
        CHECK(result.y() == doctest::Approx(10.0f));
        CHECK(result.z() == doctest::Approx(12.0f));
    }
    
    SUBCASE("Addition with aliasing - result = a + a") {
        quatf a(1.0f, 2.0f, 3.0f, 4.0f);
        quatf expected(2.0f, 4.0f, 6.0f, 8.0f);
        
        add(a, a, a);  // a = a + a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Addition with aliasing - result = a + b where result is a") {
        quatf a(1.0f, 2.0f, 3.0f, 4.0f);
        quatf b(5.0f, 6.0f, 7.0f, 8.0f);
        quatf expected(6.0f, 8.0f, 10.0f, 12.0f);
        
        add(a, b, a);  // a = a + b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Random addition tests") {
        for (int test = 0; test < 10; ++test) {
            auto a = rng_f.generate_unnormalized();
            auto b = rng_f.generate_unnormalized();
            quatf result;
            quatf expected(a.w() + b.w(), a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
            
            add(a, b, result);
            CHECK(approx_equal(result, expected, FLOAT_TOL));
        }
    }
    
    SUBCASE("Addition - double precision") {
        quatd a(1.0, 2.0, 3.0, 4.0);
        quatd b(5.0, 6.0, 7.0, 8.0);
        quatd result;
        
        add(a, b, result);
        
        CHECK(result.w() == doctest::Approx(6.0));
        CHECK(result.x() == doctest::Approx(8.0));
        CHECK(result.y() == doctest::Approx(10.0));
        CHECK(result.z() == doctest::Approx(12.0));
    }
}

TEST_CASE("Direct quaternion subtraction") {
    SUBCASE("Basic subtraction") {
        quatf a(5.0f, 7.0f, 9.0f, 11.0f);
        quatf b(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result;
        
        sub(a, b, result);
        
        CHECK(result.w() == doctest::Approx(4.0f));
        CHECK(result.x() == doctest::Approx(5.0f));
        CHECK(result.y() == doctest::Approx(6.0f));
        CHECK(result.z() == doctest::Approx(7.0f));
    }
    
    SUBCASE("Subtraction with aliasing - result = a - a") {
        quatf a(1.0f, 2.0f, 3.0f, 4.0f);
        quatf expected(0.0f, 0.0f, 0.0f, 0.0f);
        
        sub(a, a, a);  // a = a - a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

TEST_CASE("Direct quaternion multiplication") {
    SUBCASE("Basic multiplication - identity") {
        quatf identity = quatf::identity();
        quatf q(0.5f, 0.5f, 0.5f, 0.5f);  // Normalized quaternion
        quatf result;
        
        mul(identity, q, result);
        CHECK(approx_equal(result, q, FLOAT_TOL));
        
        mul(q, identity, result);
        CHECK(approx_equal(result, q, FLOAT_TOL));
    }
    
    SUBCASE("Basic multiplication - orthogonal rotations") {
        // 90 degree rotation around X axis
        quatf qx = quatf::from_axis_angle(vec3<float>(1, 0, 0), radian<float>(constants<float>::half_pi));
        // 90 degree rotation around Y axis
        quatf qy = quatf::from_axis_angle(vec3<float>(0, 1, 0), radian<float>(constants<float>::half_pi));
        quatf result;
        
        mul(qx, qy, result);
        
        // Verify the result represents the correct composition
        vec3<float> v(0, 0, 1);
        vec3<float> rotated = rotate(v, result);
        vec3<float> expected(1, 0, 0);  // Z rotated by X then Y becomes X
        
        CHECK(approx_equal(rotated, expected, FLOAT_TOL));
    }
    
    SUBCASE("Multiplication with aliasing") {
        quatf q(0.5f, 0.5f, 0.5f, 0.5f);
        quatf original = q;
        
        mul(q, q, q);  // q = q * q
        
        // Verify it's different from original (unless it's a 180 degree rotation)
        CHECK(norm(q) == doctest::Approx(norm(original)));
    }
    
    SUBCASE("Multiplication preserves unit norm") {
        RandomQuaternionGenerator<float> rng;
        
        for (int test = 0; test < 10; ++test) {
            auto q1 = rng.generate();  // Unit quaternion
            auto q2 = rng.generate();  // Unit quaternion
            quatf result;
            
            mul(q1, q2, result);
            
            CHECK(norm(result) == doctest::Approx(1.0f).epsilon(FLOAT_TOL));
        }
    }
}

TEST_CASE("Direct quaternion scalar operations") {
    SUBCASE("Scale quaternion") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result;
        
        scale(q, 2.0f, result);
        
        CHECK(result.w() == doctest::Approx(2.0f));
        CHECK(result.x() == doctest::Approx(4.0f));
        CHECK(result.y() == doctest::Approx(6.0f));
        CHECK(result.z() == doctest::Approx(8.0f));
    }
    
    SUBCASE("Scale with aliasing") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf expected(0.5f, 1.0f, 1.5f, 2.0f);
        
        scale(q, 0.5f, q);  // q = 0.5 * q
        
        CHECK(approx_equal(q, expected, FLOAT_TOL));
    }
    
    SUBCASE("Multiplication aliases") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result1, result2;
        
        // Test both aliases
        mul(3.0f, q, result1);
        mul(q, 3.0f, result2);
        
        CHECK(approx_equal(result1, result2, FLOAT_TOL));
    }
}

// =============================================================================
// Geometric Operations Tests
// =============================================================================

TEST_CASE("Quaternion conjugate and negate") {
    SUBCASE("Conjugate") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result;
        
        conjugate(q, result);
        
        CHECK(result.w() == doctest::Approx(1.0f));
        CHECK(result.x() == doctest::Approx(-2.0f));
        CHECK(result.y() == doctest::Approx(-3.0f));
        CHECK(result.z() == doctest::Approx(-4.0f));
    }
    
    SUBCASE("Conjugate with aliasing") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        
        conjugate(q, q);  // q = conjugate(q)
        
        CHECK(q.w() == doctest::Approx(1.0f));
        CHECK(q.x() == doctest::Approx(-2.0f));
        CHECK(q.y() == doctest::Approx(-3.0f));
        CHECK(q.z() == doctest::Approx(-4.0f));
    }
    
    SUBCASE("Negate") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result;
        
        negate(q, result);
        
        CHECK(result.w() == doctest::Approx(-1.0f));
        CHECK(result.x() == doctest::Approx(-2.0f));
        CHECK(result.y() == doctest::Approx(-3.0f));
        CHECK(result.z() == doctest::Approx(-4.0f));
    }
    
    SUBCASE("Double negation returns original") {
        quatf q = quatf::from_axis_angle(vec3<float>(1, 0, 0), radian<float>(1.0f));
        quatf result;
        
        negate(q, result);
        negate(result, result);
        
        // q and -q represent the same rotation
        CHECK(approx_equal(result, q, FLOAT_TOL));
    }
}

TEST_CASE("Quaternion dot product and norm") {
    SUBCASE("Dot product of identical quaternions") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        
        float result = euler::direct::dot(q, q);
        float expected = 1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f;
        
        CHECK(result == doctest::Approx(expected));
    }
    
    SUBCASE("Dot product of orthogonal quaternions") {
        quatf q1 = quatf::from_axis_angle(vec3<float>(1, 0, 0), radian<float>(constants<float>::half_pi));
        quatf q2 = quatf::from_axis_angle(vec3<float>(0, 1, 0), radian<float>(constants<float>::half_pi));
        
        float result = euler::direct::dot(q1, q2);
        
        // These quaternions are not orthogonal in 4D space
        // dot product = w1*w2 + x1*x2 + y1*y2 + z1*z2
        // = 0.707*0.707 + 0.707*0 + 0*0.707 + 0*0 = 0.5
        CHECK(result == doctest::Approx(0.5f).epsilon(FLOAT_TOL));
    }
    
    SUBCASE("Norm of unit quaternion") {
        quatf q = quatf::from_axis_angle(vec3<float>(1, 0, 0), radian<float>(1.0f));
        
        float n = norm(q);
        
        CHECK(n == doctest::Approx(1.0f));
    }
    
    SUBCASE("Norm squared") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        
        float n2 = norm_squared(q);
        float expected = 1.0f + 4.0f + 9.0f + 16.0f;
        
        CHECK(n2 == doctest::Approx(expected));
    }
}

TEST_CASE("Quaternion normalization") {
    SUBCASE("Normalize non-unit quaternion") {
        quatf q(3.0f, 4.0f, 0.0f, 0.0f);
        quatf result;
        
        normalize(q, result);
        
        CHECK(norm(result) == doctest::Approx(1.0f));
        CHECK(result.w() == doctest::Approx(0.6f));
        CHECK(result.x() == doctest::Approx(0.8f));
    }
    
    SUBCASE("Normalize with aliasing") {
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        
        normalize(q, q);  // q = normalize(q)
        
        CHECK(norm(q) == doctest::Approx(1.0f));
    }
    
    SUBCASE("Normalize zero quaternion") {
        quatf q(0.0f, 0.0f, 0.0f, 0.0f);
        quatf result;
        
        normalize(q, result);
        
        // Should return identity
        CHECK(approx_equal(result, quatf::identity(), FLOAT_TOL));
    }
}

TEST_CASE("Quaternion inverse") {
    SUBCASE("Inverse of unit quaternion") {
        quatf q = quatf::from_axis_angle(vec3<float>(0, 0, 1), radian<float>(constants<float>::quarter_pi));
        quatf inv;
        
        inverse(q, inv);
        
        // For unit quaternions, inverse equals conjugate
        quatf conj;
        conjugate(q, conj);
        CHECK(approx_equal(inv, conj, FLOAT_TOL));
        
        // q * q^(-1) = identity
        quatf identity_check;
        mul(q, inv, identity_check);
        CHECK(approx_equal(identity_check, quatf::identity(), FLOAT_TOL));
    }
    
    SUBCASE("Inverse of non-unit quaternion") {
        quatf q(2.0f, 0.0f, 0.0f, 0.0f);  // Non-unit quaternion
        quatf inv;
        
        inverse(q, inv);
        
        CHECK(inv.w() == doctest::Approx(0.5f));
        CHECK(inv.x() == doctest::Approx(0.0f));
        CHECK(inv.y() == doctest::Approx(0.0f));
        CHECK(inv.z() == doctest::Approx(0.0f));
        
        // q * q^(-1) = identity
        quatf identity_check;
        mul(q, inv, identity_check);
        CHECK(approx_equal(identity_check, quatf::identity(), FLOAT_TOL));
    }
    
    SUBCASE("Inverse with aliasing") {
        quatf q = quatf::from_axis_angle(vec3<float>(1, 1, 1).normalized(), radian<float>(1.0f));
        quatf original = q;
        
        inverse(q, q);  // q = inverse(q)
        
        // q * original = identity
        quatf identity_check;
        mul(q, original, identity_check);
        CHECK(approx_equal(identity_check, quatf::identity(), FLOAT_TOL));
    }
}

// =============================================================================
// Conversion Operations Tests
// =============================================================================

TEST_CASE("Quaternion to matrix conversions") {
    SUBCASE("Identity quaternion to matrix") {
        quatf q = quatf::identity();
        matrix3<float> m3;
        matrix4<float> m4;
        
        quat_to_mat3(q, m3);
        quat_to_mat4(q, m4);
        
        CHECK(approx_equal(m3, matrix3<float>::identity(), FLOAT_TOL));
        CHECK(approx_equal(m4, matrix4<float>::identity(), FLOAT_TOL));
    }
    
    SUBCASE("90 degree rotation around Z") {
        quatf q = quatf::from_axis_angle(vec3<float>(0, 0, 1), radian<float>(constants<float>::half_pi));
        matrix3<float> m3;
        
        quat_to_mat3(q, m3);
        
        // Check that it rotates X to Y
        vec3<float> x_axis(1, 0, 0);
        vec3<float> rotated = m3 * x_axis;
        vec3<float> expected(0, 1, 0);
        
        CHECK(approx_equal(rotated, expected, FLOAT_TOL));
    }
    
    SUBCASE("Quaternion to 4x4 matrix") {
        quatf q = quatf::from_axis_angle(vec3<float>(1, 0, 0), radian<float>(constants<float>::quarter_pi));
        matrix4<float> m4;
        
        quat_to_mat4(q, m4);
        
        // Check bottom row
        CHECK(m4(3, 0) == doctest::Approx(0.0f));
        CHECK(m4(3, 1) == doctest::Approx(0.0f));
        CHECK(m4(3, 2) == doctest::Approx(0.0f));
        CHECK(m4(3, 3) == doctest::Approx(1.0f));
        
        // Check translation column
        CHECK(m4(0, 3) == doctest::Approx(0.0f));
        CHECK(m4(1, 3) == doctest::Approx(0.0f));
        CHECK(m4(2, 3) == doctest::Approx(0.0f));
    }
}

TEST_CASE("Matrix to quaternion conversions") {
    SUBCASE("Identity matrix to quaternion") {
        matrix3<float> m3 = matrix3<float>::identity();
        quatf q;
        
        mat3_to_quat(m3, q);
        
        CHECK(approx_equal(q, quatf::identity(), FLOAT_TOL));
    }
    
    SUBCASE("Rotation matrix to quaternion") {
        // Create a rotation matrix for 45 degrees around Y axis
        float angle = constants<float>::quarter_pi;
        float c = std::cos(angle);
        float s = std::sin(angle);
        
        matrix3<float> m3;
        m3(0, 0) = c;  m3(0, 1) = 0;  m3(0, 2) = s;
        m3(1, 0) = 0;  m3(1, 1) = 1;  m3(1, 2) = 0;
        m3(2, 0) = -s; m3(2, 1) = 0;  m3(2, 2) = c;
        
        quatf q;
        mat3_to_quat(m3, q);
        
        // Verify by converting back
        matrix3<float> m3_check;
        quat_to_mat3(q, m3_check);
        
        CHECK(approx_equal(m3, m3_check, FLOAT_TOL));
    }
    
    SUBCASE("4x4 matrix to quaternion") {
        quatf original = quatf::from_axis_angle(vec3<float>(1, 1, 1).normalized(), radian<float>(1.0f));
        matrix4<float> m4;
        
        quat_to_mat4(original, m4);
        
        quatf q;
        mat4_to_quat(m4, q);
        
        // Note: q and -q represent the same rotation
        bool same_rotation = approx_equal(q, original, FLOAT_TOL) || 
                           approx_equal(q, -original, FLOAT_TOL);
        CHECK(same_rotation);
    }
    
    SUBCASE("Round-trip conversion preserves rotation") {
        RandomQuaternionGenerator<float> rng;
        
        for (int test = 0; test < 10; ++test) {
            quatf original = rng.generate();
            matrix3<float> m3;
            quatf recovered;
            
            quat_to_mat3(original, m3);
            mat3_to_quat(m3, recovered);
            
            // Check that they represent the same rotation
            // (accounting for double cover: q and -q are the same rotation)
            bool same_rotation = approx_equal(recovered, original, FLOAT_TOL) || 
                               approx_equal(recovered, -original, FLOAT_TOL);
            CHECK(same_rotation);
        }
    }
}

// =============================================================================
// Edge Cases and Special Values
// =============================================================================

TEST_CASE("Edge cases and special values") {
    SUBCASE("Operations with zero quaternion") {
        quatf zero(0.0f, 0.0f, 0.0f, 0.0f);
        quatf q(1.0f, 2.0f, 3.0f, 4.0f);
        quatf result;
        
        add(zero, q, result);
        CHECK(approx_equal(result, q, FLOAT_TOL));
        
        mul(zero, q, result);
        CHECK(approx_equal(result, zero, FLOAT_TOL));
    }
    
    SUBCASE("Very small quaternions") {
        quatf tiny(1e-30f, 1e-30f, 1e-30f, 1e-30f);
        quatf result;
        
        // Should handle denormals gracefully
        normalize(tiny, result);
        inverse(tiny, result);
    }
}

