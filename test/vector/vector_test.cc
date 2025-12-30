#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/core/types.hh>

TEST_CASE("euler::vector construction") {
    using namespace euler;

    SUBCASE("scalar constructor") {
        vector3 v(5.0f);
        CHECK(v[0] == 5.0f);
        CHECK(v[1] == 5.0f);
        CHECK(v[2] == 5.0f);
    }
    
    SUBCASE("component constructor") {
        vector3 v(1.0f, 2.0f, 3.0f);
        CHECK(v[0] == 1.0f);
        CHECK(v[1] == 2.0f);
        CHECK(v[2] == 3.0f);
        
        // Named access
        CHECK(v.x() == 1.0f);
        CHECK(v.y() == 2.0f);
        CHECK(v.z() == 3.0f);
    }
    
    SUBCASE("factory methods") {
        auto zero = vector3::zero();
        CHECK(zero[0] == 0.0f);
        CHECK(zero[1] == 0.0f);
        CHECK(zero[2] == 0.0f);
        
        auto ones = vector3::ones();
        CHECK(ones[0] == 1.0f);
        CHECK(ones[1] == 1.0f);
        CHECK(ones[2] == 1.0f);
        
        auto unit_x = vector3::unit_x();
        CHECK(unit_x.x() == 1.0f);
        CHECK(unit_x.y() == 0.0f);
        CHECK(unit_x.z() == 0.0f);
    }
}

TEST_CASE("euler::vector layout") {
    using namespace euler;
    
    SUBCASE("default vector follows system layout") {
        vector3 v(1.0f, 2.0f, 3.0f);
        
        // The default vector type adapts to the system layout
        // but we can't test specifics without knowing the build configuration
        // Instead, we verify the interface works correctly
        CHECK(v[0] == 1.0f);
        CHECK(v[1] == 2.0f);
        CHECK(v[2] == 3.0f);
        CHECK(v.x() == 1.0f);
        CHECK(v.y() == 2.0f);
        CHECK(v.z() == 3.0f);
        CHECK(v.size == 3);
    }
    
    SUBCASE("explicit column vector") {
        column_vector<float, 3> cv;
        cv[0] = 1.0f;
        cv[1] = 2.0f;
        cv[2] = 3.0f;
        CHECK(cv.rows == 3);
        CHECK(cv.cols == 1);
        
        // Should be stored contiguously
        CHECK(cv[0] == 1.0f);
        CHECK(cv[1] == 2.0f);
        CHECK(cv[2] == 3.0f);
    }
    
    SUBCASE("explicit row vector") {
        row_vector<float, 3> rv;
        rv[0] = 1.0f;
        rv[1] = 2.0f;
        rv[2] = 3.0f;
        CHECK(rv.rows == 1);
        CHECK(rv.cols == 3);
        
        // Should be stored contiguously
        CHECK(rv[0] == 1.0f);
        CHECK(rv[1] == 2.0f);
        CHECK(rv[2] == 3.0f);
    }
}

TEST_CASE("euler::vector operations") {
    using namespace euler;
    
    SUBCASE("length and normalization") {
        vector3 v(3.0f, 4.0f, 0.0f);
        CHECK(v.length_squared() == doctest::Approx(25.0f));
        CHECK(v.length() == doctest::Approx(5.0f));
        
        auto normalized = v.normalized();
        CHECK(normalized.length() == doctest::Approx(1.0f));
        CHECK(normalized.x() == doctest::Approx(0.6f));
        CHECK(normalized.y() == doctest::Approx(0.8f));
        CHECK(normalized.z() == doctest::Approx(0.0f));
    }
    
    SUBCASE("arithmetic operations") {
        vector3 a(1.0f, 2.0f, 3.0f);
        vector3 b(4.0f, 5.0f, 6.0f);
        
        vector3 sum = a + b;
        CHECK(sum.x() == 5.0f);
        CHECK(sum.y() == 7.0f);
        CHECK(sum.z() == 9.0f);
        
        vector3 diff = b - a;
        CHECK(diff.x() == 3.0f);
        CHECK(diff.y() == 3.0f);
        CHECK(diff.z() == 3.0f);
        
        vector3 scaled = a * 2.0f;
        CHECK(scaled.x() == 2.0f);
        CHECK(scaled.y() == 4.0f);
        CHECK(scaled.z() == 6.0f);
    }
    
    SUBCASE("dot product") {
        vector3 a(1.0f, 2.0f, 3.0f);
        vector3 b(4.0f, 5.0f, 6.0f);
        
        float d = dot(a, b);
        CHECK(d == doctest::Approx(32.0f));  // 1*4 + 2*5 + 3*6
    }
    
    SUBCASE("cross product") {
        vector3 x = vector3::unit_x();
        vector3 y = vector3::unit_y();
        vector3 z = cross(x, y);
        
        CHECK(z.x() == doctest::Approx(0.0f));
        CHECK(z.y() == doctest::Approx(0.0f));
        CHECK(z.z() == doctest::Approx(1.0f));
        
        // 2D cross product
        vec2f a(1.0f, 0.0f);
        vec2f b(0.0f, 1.0f);
        float cross2d = cross(a, b);
        CHECK(cross2d == doctest::Approx(1.0f));
    }
}

TEST_CASE("euler::vector component access") {
    using namespace euler;
    
    SUBCASE("vec2 components") {
        vec2f v(1.0f, 2.0f);
        CHECK(v.x() == 1.0f);
        CHECK(v.y() == 2.0f);
        
        // RGB aliases
        CHECK(v.r() == 1.0f);
        CHECK(v.g() == 2.0f);
        
        // Modification
        v.x() = 10.0f;
        CHECK(v[0] == 10.0f);
    }
    
    SUBCASE("vec3 components") {
        vec3f v(1.0f, 2.0f, 3.0f);
        CHECK(v.x() == 1.0f);
        CHECK(v.y() == 2.0f);
        CHECK(v.z() == 3.0f);
        
        // RGB aliases
        CHECK(v.r() == 1.0f);
        CHECK(v.g() == 2.0f);
        CHECK(v.b() == 3.0f);
    }
    
    SUBCASE("vec4 components") {
        vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
        CHECK(v.x() == 1.0f);
        CHECK(v.y() == 2.0f);
        CHECK(v.z() == 3.0f);
        CHECK(v.w() == 4.0f);
        
        // RGBA aliases
        CHECK(v.r() == 1.0f);
        CHECK(v.g() == 2.0f);
        CHECK(v.b() == 3.0f);
        CHECK(v.a() == 4.0f);
    }
}

TEST_CASE("euler::vector advanced operations") {
    using namespace euler;
    
    SUBCASE("reflection") {
        vector3 incident(0.0f, -1.0f, 0.0f);  // Straight down
        vector3 normal(0.0f, 1.0f, 0.0f);     // Up normal
        
        vector3 reflected = reflect(incident, normal);
        
        // Should reflect straight up
        CHECK(reflected.x() == doctest::Approx(0.0f));
        CHECK(reflected.y() == doctest::Approx(1.0f));
        CHECK(reflected.z() == doctest::Approx(0.0f));
    }
    
    SUBCASE("projection") {
        vector3 a(5.0f, 0.0f, 0.0f);
        vector3 b(1.0f, 1.0f, 0.0f);
        
        vector3 proj = project(a, b);
        CHECK(proj.x() == doctest::Approx(2.5f));
        CHECK(proj.y() == doctest::Approx(2.5f));
        CHECK(proj.z() == doctest::Approx(0.0f));
    }
    
    SUBCASE("interpolation") {
        vector3 a(0.0f, 0.0f, 0.0f);
        vector3 b(10.0f, 10.0f, 10.0f);
        
        vector3 mid = lerp(a, b, 0.5f);
        CHECK(mid.x() == doctest::Approx(5.0f));
        CHECK(mid.y() == doctest::Approx(5.0f));
        CHECK(mid.z() == doctest::Approx(5.0f));
    }
    
    SUBCASE("component-wise operations") {
        vector3 a(1.0f, 5.0f, 3.0f);
        vector3 b(3.0f, 2.0f, 4.0f);
        
        vector3 min_v = min(a, b);
        CHECK(min_v.x() == 1.0f);
        CHECK(min_v.y() == 2.0f);
        CHECK(min_v.z() == 3.0f);
        
        vector3 max_v = max(a, b);
        CHECK(max_v.x() == 3.0f);
        CHECK(max_v.y() == 5.0f);
        CHECK(max_v.z() == 4.0f);
        
        vector3 v(-1.0f, 2.0f, -3.0f);
        vector3 abs_v = abs(v);
        CHECK(abs_v.x() == 1.0f);
        CHECK(abs_v.y() == 2.0f);
        CHECK(abs_v.z() == 3.0f);
    }
}

TEST_CASE("euler::vector type traits") {
    using namespace euler;
    
    SUBCASE("is_vector trait") {
        CHECK(is_vector_v<vector3>);
        CHECK(is_vector_v<vec2f>);
        CHECK(is_vector_v<column_vector<float, 4>>);
        CHECK(is_vector_v<row_vector<double, 3>>);
        CHECK(!is_vector_v<float>);
        CHECK(!is_vector_v<matrix<float, 3, 3>>);
    }
    
    SUBCASE("vector_dimension trait") {
        CHECK(vector_dimension_v<vec2f> == 2);
        CHECK(vector_dimension_v<vector3> == 3);
        CHECK(vector_dimension_v<vec4d> == 4);
        CHECK(vector_dimension_v<column_vector<float, 5>> == 5);
        CHECK(vector_dimension_v<row_vector<int, 7>> == 7);
        CHECK(vector_dimension_v<float> == 0);
    }
}

TEST_CASE("euler::vector row/column compatibility") {
    using namespace euler;
    
    SUBCASE("dot product between row and column vectors") {
        column_vector<float, 3> cv;
        cv[0] = 1.0f;
        cv[1] = 2.0f;
        cv[2] = 3.0f;
        
        row_vector<float, 3> rv;
        rv[0] = 4.0f;
        rv[1] = 5.0f;
        rv[2] = 6.0f;
        
        // Should work seamlessly
        float d1 = dot(cv, rv);
        float d2 = dot(rv, cv);
        CHECK(d1 == doctest::Approx(32.0f));  // 1*4 + 2*5 + 3*6
        CHECK(d2 == doctest::Approx(32.0f));  // Commutative
        
        // Also with regular vectors
        vector3 v(1.0f, 2.0f, 3.0f);
        float d3 = dot(v, cv);
        float d4 = dot(v, rv);
        CHECK(d3 == doctest::Approx(14.0f));  // 1*1 + 2*2 + 3*3
        CHECK(d4 == doctest::Approx(32.0f));
    }
    
    SUBCASE("constructor argument validation") {
        // This should compile - exactly 3 arguments
        vector3 v1(1.0f, 2.0f, 3.0f);
        
        // These should not compile (uncomment to test)
        // vector3 v2(1.0f, 2.0f);           // Too few
        // vector3 v3(1.0f, 2.0f, 3.0f, 4.0f); // Too many
    }
    
    SUBCASE("efficient conversion") {
        // Create vectors and verify conversions work
        column_vector<float, 3> cv;
        cv[0] = 1.0f;
        cv[1] = 2.0f;
        cv[2] = 3.0f;
        
        // Convert to row vector
        row_vector<float, 3> rv(cv);
        CHECK(rv[0] == 1.0f);
        CHECK(rv[1] == 2.0f);
        CHECK(rv[2] == 3.0f);
        
        // Convert to generic vector
        vector3 v(rv);
        CHECK(v[0] == 1.0f);
        CHECK(v[1] == 2.0f);
        CHECK(v[2] == 3.0f);
    }
}