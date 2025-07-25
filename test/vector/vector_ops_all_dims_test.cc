#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <cmath>

// Helper macro to test an operation for all dimensions
#define TEST_ALL_DIMS(test_name, test_body) \
    SUBCASE("2D " test_name) { \
        using Vec = euler::vec2<float>; \
        [[maybe_unused]] constexpr size_t dim = 2; \
        test_body \
    } \
    SUBCASE("3D " test_name) { \
        using Vec = euler::vec3<float>; \
        [[maybe_unused]] constexpr size_t dim = 3; \
        test_body \
    } \
    SUBCASE("4D " test_name) { \
        using Vec = euler::vec4<float>; \
        [[maybe_unused]] constexpr size_t dim = 4; \
        test_body \
    }

TEST_CASE("euler::vector operations - all dimensions") {
    using namespace euler;
    
    // Test reflect operation for all dimensions
    TEST_ALL_DIMS("reflect", {
        Vec incident = Vec::ones();
        Vec normal = Vec::unit_x();
        
        Vec reflected = reflect(incident, normal);
        
        // Verify reflection formula: r = i - 2 * dot(i, n) * n
        float d = dot(incident, normal);
        Vec expected = incident - 2.0f * d * normal;
        
        for (size_t i = 0; i < dim; ++i) {
            CHECK(reflected[i] == doctest::Approx(expected[i]));
        }
        
        // Test with expression
        // Compute intermediate values to help debug
        Vec normal_plus_y = normal + Vec::unit_y();
        Vec normalized_normal = normalize(normal_plus_y);
        Vec reflected_expr = reflect(incident * 2.0f, normalized_normal);
        CHECK(length(reflected_expr) > 0.0f);
    })
    
    // Test project/reject operations for all dimensions
    TEST_ALL_DIMS("project and reject", {
        Vec a = Vec::ones();
        Vec b = Vec::unit_x();
        
        Vec proj = project(a, b);
        Vec rej = reject(a, b);
        
        // Project should give component along b
        CHECK(proj[0] == doctest::Approx(1.0f));
        for (size_t i = 1; i < dim; ++i) {
            CHECK(proj[i] == doctest::Approx(0.0f));
        }
        
        // Reject should give component perpendicular to b
        CHECK(rej[0] == doctest::Approx(0.0f));
        for (size_t i = 1; i < dim; ++i) {
            CHECK(rej[i] == doctest::Approx(1.0f));
        }
        
        // proj + rej should equal original vector
        Vec sum = proj + rej;
        for (size_t i = 0; i < dim; ++i) {
            CHECK(sum[i] == doctest::Approx(a[i]));
        }
        
        // Test orthogonality
        CHECK(dot(proj, rej) == doctest::Approx(0.0f).epsilon(0.0001f));
    })
    
    // Test distance operations for all dimensions
    TEST_ALL_DIMS("distance", {

        Vec a = Vec::zero();
        Vec b = Vec::ones();
        
        float dist = distance(a, b);
        float dist_sq = distance_squared(a, b);
        
        // Distance from origin to (1,1,...) is sqrt(dim)
        CHECK(dist == doctest::Approx(std::sqrt(static_cast<float>(dim))));
        CHECK(dist_sq == doctest::Approx(static_cast<float>(dim)));
        
        // Test with expressions
        // Create intermediate values to avoid complex expression issues
        Vec a_plus_x = a + Vec::unit_x();
        Vec b_times_2 = b * 2.0f;
        float dist_expr = distance(a_plus_x, b_times_2);
        CHECK(dist_expr > 0.0f);
    })
    
    // Test angle calculation for all dimensions
    TEST_ALL_DIMS("angle", {
        Vec a = Vec::unit_x();
        Vec b = Vec::unit_x();
        
        // Angle between same vectors should be 0
        float angle_same = angle_between(a, b);
        CHECK(angle_same == doctest::Approx(0.0f));
        
        // For 2D and 3D, test perpendicular vectors
        if (dim >= 2) {
            Vec c = Vec::unit_y();
            float angle_perp = angle_between(a, c);
            CHECK(angle_perp == doctest::Approx(constants<float>::half_pi));
        }
        
        // Angle between opposite vectors
        Vec d = -a;
        float angle_opp = angle_between(a, d);
        CHECK(angle_opp == doctest::Approx(constants<float>::pi));
    })
    
    // Test lerp for all dimensions
    TEST_ALL_DIMS("lerp", {
        Vec start = Vec::zero();
        Vec end = Vec::ones();
        
        // Test interpolation at different t values
        Vec mid = lerp(start, end, 0.5f);
        for (size_t i = 0; i < dim; ++i) {
            CHECK(mid[i] == doctest::Approx(0.5f));
        }
        
        Vec quarter = lerp(start, end, 0.25f);
        for (size_t i = 0; i < dim; ++i) {
            CHECK(quarter[i] == doctest::Approx(0.25f));
        }
        
        // Test with expressions
        Vec lerp_expr = lerp(start * 2.0f, end + Vec::ones(), 0.5f);
        CHECK(length(lerp_expr) > 0.0f);
    })
    
    // Test component-wise operations for all dimensions
    TEST_ALL_DIMS("component-wise min/max/abs/clamp", {
        Vec a = Vec::zero();
        Vec b = Vec::ones();
        
        // Initialize a with different values
        for (size_t i = 0; i < dim; ++i) {
            a[i] = static_cast<float>(i) - static_cast<float>(dim) / 2.0f;
        }
        
        // Test min
        Vec min_result = min(a, b);
        for (size_t i = 0; i < dim; ++i) {
            CHECK(min_result[i] == doctest::Approx(std::min(a[i], b[i])));
        }
        
        // Test max
        Vec max_result = max(a, b);
        for (size_t i = 0; i < dim; ++i) {
            CHECK(max_result[i] == doctest::Approx(std::max(a[i], b[i])));
        }
        
        // Test abs
        Vec abs_result = abs(a);
        for (size_t i = 0; i < dim; ++i) {
            CHECK(abs_result[i] == doctest::Approx(std::abs(a[i])));
        }
        
        // Test clamp
        Vec clamped = clamp(a, Vec::zero(), Vec::ones());
        for (size_t i = 0; i < dim; ++i) {
            CHECK(clamped[i] >= 0.0f);
            CHECK(clamped[i] <= 1.0f);
        }
    })
    
    // Test faceforward for all dimensions
    TEST_ALL_DIMS("faceforward", {
        Vec n = Vec::unit_x();
        Vec i = Vec::ones();
        Vec nref = Vec::unit_x();
        
        Vec result = faceforward(n, i, nref);
        
        // Should flip n because dot(i, nref) > 0
        CHECK(result[0] == doctest::Approx(-1.0f));
        
        // Test case where it shouldn't flip
        Vec i2 = -Vec::unit_x();
        Vec result2 = faceforward(n, i2, nref);
        CHECK(result2[0] == doctest::Approx(1.0f));
    })
    
    // Test approx_equal and approx_zero for all dimensions
    TEST_ALL_DIMS("approx utilities", {
        Vec a = Vec::ones();
        Vec b = Vec::ones();
        
        // Test exact equality
        CHECK(approx_equal(a, b));
        
        // Test with small difference
        b[0] += constants<float>::epsilon * 2;
        CHECK(approx_equal(a, b, constants<float>::epsilon * 3));
        CHECK(!approx_equal(a, b, constants<float>::epsilon));
        
        // Test approx_zero
        Vec small = Vec::zero();
        CHECK(approx_zero(small));
        
        for (size_t i = 0; i < dim; ++i) {
            small[i] = constants<float>::epsilon * 0.5f;
        }
        // For a vector with all components = epsilon * 0.5,
        // length_squared = dim * (epsilon * 0.5)^2 = dim * epsilon^2 / 4
        // So we need tolerance > sqrt(dim) * epsilon / 2
        CHECK(approx_zero(small, constants<float>::epsilon * std::sqrt(static_cast<float>(dim))));
    })
    
    // Test smoothstep for all dimensions
    TEST_ALL_DIMS("smoothstep", {
        Vec edge0 = Vec::zero();
        Vec edge1 = Vec::ones();
        Vec x = Vec::ones() * 0.5f;
        
        Vec result = smoothstep(edge0, edge1, x);
        
        // At midpoint, smoothstep should return 0.5
        for (size_t i = 0; i < dim; ++i) {
            CHECK(result[i] == doctest::Approx(0.5f));
        }
        
        // Test at edges
        Vec at_zero = smoothstep(edge0, edge1, edge0);
        Vec at_one = smoothstep(edge0, edge1, edge1);
        
        for (size_t i = 0; i < dim; ++i) {
            CHECK(at_zero[i] == doctest::Approx(0.0f));
            CHECK(at_one[i] == doctest::Approx(1.0f));
        }
    })
}

TEST_CASE("euler::vector dimension-specific operations") {
    using namespace euler;
    
    SUBCASE("2D cross product") {
        vec2<float> a(1.0f, 0.0f);
        vec2<float> b(0.0f, 1.0f);
        
        // 2D cross product returns scalar
        float cross_result = cross(a, b);
        CHECK(cross_result == doctest::Approx(1.0f));
        
        // Test with parallel vectors
        float cross_parallel = cross(a, a);
        CHECK(cross_parallel == doctest::Approx(0.0f));
    }
    
    SUBCASE("3D cross product") {
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(0.0f, 1.0f, 0.0f);
        
        // 3D cross product returns vector
        vec3<float> cross_result = cross(a, b);
        CHECK(cross_result[0] == doctest::Approx(0.0f));
        CHECK(cross_result[1] == doctest::Approx(0.0f));
        CHECK(cross_result[2] == doctest::Approx(1.0f));
        
        // Test magnitude
        CHECK(length(cross_result) == doctest::Approx(1.0f));
        
        // Cross product should be perpendicular to both inputs
        CHECK(dot(cross_result, a) == doctest::Approx(0.0f));
        CHECK(dot(cross_result, b) == doctest::Approx(0.0f));
    }
    
    SUBCASE("Orthonormalization dimension checks") {
        // Each dimension has its specific orthonormalization
        {
            vec2<float> v0(1.0f, 1.0f);
            vec2<float> v1(0.0f, 1.0f);
            orthonormalize(v0, v1);
            CHECK(length(v0) == doctest::Approx(1.0f));
            CHECK(length(v1) == doctest::Approx(1.0f));
            CHECK(dot(v0, v1) == doctest::Approx(0.0f).epsilon(0.001f));
        }
        
        {
            vec3<float> v0(1.0f, 0.0f, 0.0f);
            vec3<float> v1(1.0f, 1.0f, 0.0f);
            vec3<float> v2(1.0f, 1.0f, 1.0f);
            orthonormalize(v0, v1, v2);
            CHECK(length(v0) == doctest::Approx(1.0f));
            CHECK(length(v1) == doctest::Approx(1.0f));
            CHECK(length(v2) == doctest::Approx(1.0f));
        }
        
        {
            vec4<float> v0(1.0f, 0.0f, 0.0f, 0.0f);
            vec4<float> v1(1.0f, 1.0f, 0.0f, 0.0f);
            vec4<float> v2(1.0f, 1.0f, 1.0f, 0.0f);
            vec4<float> v3(1.0f, 1.0f, 1.0f, 1.0f);
            orthonormalize(v0, v1, v2, v3);
            CHECK(length(v0) == doctest::Approx(1.0f));
            CHECK(length(v1) == doctest::Approx(1.0f));
            CHECK(length(v2) == doctest::Approx(1.0f));
            CHECK(length(v3) == doctest::Approx(1.0f));
        }
    }
}

// Test refract operation separately as it has special cases
TEST_CASE("euler::vector refract - all dimensions") {
    using namespace euler;
    
    TEST_ALL_DIMS("refract", {
        Vec incident = normalize(Vec::ones());
        Vec normal = Vec::unit_x();
        float eta = 1.0f / 1.5f; // Air to glass
        
        Vec refracted = refract(incident, normal, eta);
        // Basic check that refraction occurred (non-zero result for normal refraction)
        CHECK(length(refracted) > 0.0f);
        
        // For total internal reflection case
        Vec steep_incident = Vec::unit_y();
        if (dim >= 2) {
            steep_incident[0] = 0.1f;
            steep_incident = normalize(steep_incident);
            Vec tir_result = refract(steep_incident, normal, 1.5f);
            
            // Should return zero vector for total internal reflection
            bool is_tir = true;
            for (size_t i = 0; i < dim; ++i) {
                if (tir_result[i] != 0.0f) {
                    is_tir = false;
                    break;
                }
            }
            CHECK(is_tir);
        }
    })
}