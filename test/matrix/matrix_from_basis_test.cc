#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>

TEST_CASE("euler::matrix from orthonormal basis") {
    using namespace euler;
    
    SUBCASE("2x2 matrix from build_orthonormal_basis") {
        vec2<float> n(1.0f, 1.0f);
        auto basis = build_orthonormal_basis(n);
        
        // Create matrix from the basis
        matrix<float, 2, 2> m(basis);
        
        // Check that the columns are the basis vectors
        vec2<float> col0(m(0, 0), m(1, 0));
        vec2<float> col1(m(0, 1), m(1, 1));
        
        CHECK(col0[0] == doctest::Approx(std::get<0>(basis)[0]));
        CHECK(col0[1] == doctest::Approx(std::get<0>(basis)[1]));
        CHECK(col1[0] == doctest::Approx(std::get<1>(basis)[0]));
        CHECK(col1[1] == doctest::Approx(std::get<1>(basis)[1]));
        
        // Check orthonormality
        CHECK(length(col0) == doctest::Approx(1.0f));
        CHECK(length(col1) == doctest::Approx(1.0f));
        CHECK(dot(col0, col1) == doctest::Approx(0.0f).epsilon(0.001f));
    }
    
    SUBCASE("3x3 matrix from build_orthonormal_basis") {
        vec3<float> n(0.0f, 0.0f, 1.0f);
        auto basis = build_orthonormal_basis(n);
        
        // Create matrix from the basis
        matrix<float, 3, 3> m(basis);
        
        // Check that the columns are the basis vectors
        vec3<float> col0(m(0, 0), m(1, 0), m(2, 0));
        vec3<float> col1(m(0, 1), m(1, 1), m(2, 1));
        vec3<float> col2(m(0, 2), m(1, 2), m(2, 2));
        
        CHECK(col0[0] == doctest::Approx(std::get<0>(basis)[0]));
        CHECK(col0[1] == doctest::Approx(std::get<0>(basis)[1]));
        CHECK(col0[2] == doctest::Approx(std::get<0>(basis)[2]));
        
        // Check orthonormality
        CHECK(length(col0) == doctest::Approx(1.0f));
        CHECK(length(col1) == doctest::Approx(1.0f));
        CHECK(length(col2) == doctest::Approx(1.0f));
        CHECK(dot(col0, col1) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(col0, col2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(col1, col2) == doctest::Approx(0.0f).epsilon(0.001f));
    }
    
    SUBCASE("4x4 matrix from build_orthonormal_basis") {
        vec4<float> n(1.0f, 0.0f, 0.0f, 0.0f);
        auto basis = build_orthonormal_basis(n);
        
        // Create matrix from the basis
        matrix<float, 4, 4> m(basis);
        
        // Check that the columns are the basis vectors
        vec4<float> col0(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
        vec4<float> col1(m(0, 1), m(1, 1), m(2, 1), m(3, 1));
        vec4<float> col2(m(0, 2), m(1, 2), m(2, 2), m(3, 2));
        vec4<float> col3(m(0, 3), m(1, 3), m(2, 3), m(3, 3));
        
        CHECK(col0[0] == doctest::Approx(std::get<0>(basis)[0]));
        CHECK(col0[1] == doctest::Approx(std::get<0>(basis)[1]));
        CHECK(col0[2] == doctest::Approx(std::get<0>(basis)[2]));
        CHECK(col0[3] == doctest::Approx(std::get<0>(basis)[3]));
        
        // Check orthonormality
        CHECK(length(col0) == doctest::Approx(1.0f));
        CHECK(length(col1) == doctest::Approx(1.0f));
        CHECK(length(col2) == doctest::Approx(1.0f));
        CHECK(length(col3) == doctest::Approx(1.0f));
    }
    
    SUBCASE("Direct assignment syntax") {
        // Test the desired syntax: m = build_orthonormal_basis(v)
        vec3<float> v(1.0f, 2.0f, 3.0f);
        matrix<float, 3, 3> m = build_orthonormal_basis(v);
        
        // Check that it's a valid rotation matrix (orthonormal columns)
        vec3<float> col0(m(0, 0), m(1, 0), m(2, 0));
        vec3<float> col1(m(0, 1), m(1, 1), m(2, 1));
        vec3<float> col2(m(0, 2), m(1, 2), m(2, 2));
        
        CHECK(length(col0) == doctest::Approx(1.0f));
        CHECK(length(col1) == doctest::Approx(1.0f));
        CHECK(length(col2) == doctest::Approx(1.0f));
        CHECK(dot(col0, col1) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(col0, col2) == doctest::Approx(0.0f).epsilon(0.001f));
        CHECK(dot(col1, col2) == doctest::Approx(0.0f).epsilon(0.001f));
    }
}