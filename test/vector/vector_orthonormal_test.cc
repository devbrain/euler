#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/vector/vector_expr.hh>

TEST_CASE("euler::vector orthonormalization") {
    using namespace euler;
    
    SUBCASE("2D orthonormalization") {
        vec2<float> v0(1.0f, 1.0f);
        vec2<float> v1(0.0f, 1.0f);
        
        // In-place orthonormalization
        orthonormalize(v0, v1);
        
        // Check orthonormality
        CHECK(length(v0) == doctest::Approx(1.0f));
        CHECK(length(v1) == doctest::Approx(1.0f));
        CHECK(dot(v0, v1) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Expression-based orthonormalization
        vec2<float> a(2.0f, 0.0f);
        vec2<float> b(1.0f, 1.0f);
        auto [r0, r1] = orthonormalize(a + b, b - a);
        
        CHECK(length(r0) == doctest::Approx(1.0f));
        CHECK(length(r1) == doctest::Approx(1.0f));
        CHECK(dot(r0, r1) == doctest::Approx(0.0f).epsilon(0.001f));
    }
    
    SUBCASE("3D orthonormalization") {
        vec3<float> v0(1.0f, 0.0f, 0.0f);
        vec3<float> v1(1.0f, 1.0f, 0.0f);
        vec3<float> v2(1.0f, 1.0f, 1.0f);
        
        // In-place orthonormalization
        orthonormalize(v0, v1, v2);
        
        // Check orthonormality
        CHECK(length(v0) == doctest::Approx(1.0f));
        CHECK(length(v1) == doctest::Approx(1.0f));
        CHECK(length(v2) == doctest::Approx(1.0f));
        CHECK(dot(v0, v1) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v0, v2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v1, v2) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Expression-based orthonormalization
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(1.0f, 1.0f, 0.0f);
        vec3<float> c(1.0f, 1.0f, 1.0f);
        auto [r0, r1, r2] = orthonormalize(normalize(a), b * 0.5f, c - a);
        
        CHECK(length(r0) == doctest::Approx(1.0f));
        CHECK(length(r1) == doctest::Approx(1.0f));
        CHECK(length(r2) == doctest::Approx(1.0f));
        CHECK(dot(r0, r1) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(r0, r2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(r1, r2) == doctest::Approx(0.0f).epsilon(0.001f));
    }
    
    SUBCASE("4D orthonormalization") {
        vec4<float> v0(1.0f, 0.0f, 0.0f, 0.0f);
        vec4<float> v1(1.0f, 1.0f, 0.0f, 0.0f);
        vec4<float> v2(1.0f, 1.0f, 1.0f, 0.0f);
        vec4<float> v3(1.0f, 1.0f, 1.0f, 1.0f);
        
        // In-place orthonormalization
        orthonormalize(v0, v1, v2, v3);
        
        // Check orthonormality
        CHECK(length(v0) == doctest::Approx(1.0f));
        CHECK(length(v1) == doctest::Approx(1.0f));
        CHECK(length(v2) == doctest::Approx(1.0f));
        CHECK(length(v3) == doctest::Approx(1.0f));
        
        // Check all pairs are orthogonal
        CHECK(dot(v0, v1) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v0, v2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v0, v3) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v1, v2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v1, v3) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(v2, v3) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Expression-based orthonormalization
        auto [r0, r1, r2, r3] = orthonormalize(
            vec4<float>::unit_x() + vec4<float>::unit_y(),
            vec4<float>::unit_y() + vec4<float>::unit_z(),
            vec4<float>::unit_z() + vec4<float>::unit_w(),
            vec4<float>::unit_w() + vec4<float>::unit_x()
        );
        
        CHECK(length(r0) == doctest::Approx(1.0f));
        CHECK(length(r1) == doctest::Approx(1.0f));
        CHECK(length(r2) == doctest::Approx(1.0f));
        CHECK(length(r3) == doctest::Approx(1.0f));
    }
}

TEST_CASE("euler::build_orthonormal_basis") {
    using namespace euler;
    
    SUBCASE("2D build_orthonormal_basis") {
        vec2<float> n(0.0f, 1.0f);
        vec2<float> t;
        
        // In-place version
        build_orthonormal_basis(n, t);
        CHECK(length(t) == doctest::Approx(1.0f));
        CHECK(dot(n, t) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Expression version
        auto [normal, tangent] = build_orthonormal_basis(vec2<float>(1.0f, 1.0f));
        CHECK(length(normal) == doctest::Approx(1.0f));
        CHECK(length(tangent) == doctest::Approx(1.0f));
        CHECK(dot(normal, tangent) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // With expression input
        vec2<float> a(3.0f, 4.0f);
        auto [n2, t2] = build_orthonormal_basis(normalize(a));
        CHECK(length(n2) == doctest::Approx(1.0f));
        CHECK(length(t2) == doctest::Approx(1.0f));
        CHECK(dot(n2, t2) == doctest::Approx(0.0f).epsilon(0.001f));
    }
    
    SUBCASE("3D build_orthonormal_basis") {
        vec3<float> n(0.0f, 0.0f, 1.0f);
        vec3<float> t, b;
        
        // In-place version
        build_orthonormal_basis(n, t, b);
        CHECK(length(t) == doctest::Approx(1.0f));
        CHECK(length(b) == doctest::Approx(1.0f));
        CHECK(dot(n, t) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(n, b) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(t, b) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Check right-handed coordinate system
        vec3<float> cross_tb = cross(t, b);
        CHECK(dot(cross_tb, n) == doctest::Approx(1.0f).epsilon(0.001f));
        
        // Expression version
        auto [normal, tangent, bitangent] = build_orthonormal_basis(vec3<float>(1.0f, 0.0f, 0.0f));
        CHECK(length(normal) == doctest::Approx(1.0f));
        CHECK(length(tangent) == doctest::Approx(1.0f));
        CHECK(length(bitangent) == doctest::Approx(1.0f));
        CHECK(dot(normal, tangent) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(normal, bitangent) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(tangent, bitangent) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Test with expression input
        vec3<float> v1(1.0f, 2.0f, 3.0f);
        vec3<float> v2(4.0f, 5.0f, 6.0f);
        auto [n3, t3, b3] = build_orthonormal_basis(cross(v1, v2));
        CHECK(length(n3) == doctest::Approx(1.0f));
        CHECK(length(t3) == doctest::Approx(1.0f));
        CHECK(length(b3) == doctest::Approx(1.0f));
    }
    
    SUBCASE("4D build_orthonormal_basis") {
        vec4<float> n(1.0f, 0.0f, 0.0f, 0.0f);
        vec4<float> t, b, c;
        
        // In-place version
        build_orthonormal_basis(n, t, b, c);
        CHECK(length(t) == doctest::Approx(1.0f));
        CHECK(length(b) == doctest::Approx(1.0f));
        CHECK(length(c) == doctest::Approx(1.0f));
        
        // Check all orthogonal
        CHECK(dot(n, t) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(n, b) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(n, c) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(t, b) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(t, c) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(b, c) == doctest::Approx(0.0f).epsilon(0.001f));
        
        // Expression version
        auto [normal, tangent, bitangent, tritangent] = build_orthonormal_basis(
            vec4<float>(1.0f, 1.0f, 1.0f, 1.0f)
        );
        CHECK(length(normal) == doctest::Approx(1.0f));
        CHECK(length(tangent) == doctest::Approx(1.0f));
        CHECK(length(bitangent) == doctest::Approx(1.0f));
        CHECK(length(tritangent) == doctest::Approx(1.0f));
        
        // With expression input
        vec4<float> v(2.0f, 3.0f, 4.0f, 5.0f);
        auto [n4, t4, b4, c4] = build_orthonormal_basis(normalize(v));
        CHECK(length(n4) == doctest::Approx(1.0f));
        CHECK(length(t4) == doctest::Approx(1.0f));
        CHECK(length(b4) == doctest::Approx(1.0f));
        CHECK(length(c4) == doctest::Approx(1.0f));
    }
}