#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/core/types.hh>

// Test basic vector functionality for all dimensions
TEST_CASE("euler::vector basic operations - all dimensions") {
    using namespace euler;
    
    SUBCASE("2D vector basics") {
        // Construction
        vec2<float> v1;
        vec2<float> v2(5.0f);
        vec2<float> v3(1.0f, 2.0f);
        
        // Scalar constructor
        CHECK(v2[0] == 5.0f);
        CHECK(v2[1] == 5.0f);
        
        // Component constructor
        CHECK(v3[0] == 1.0f);
        CHECK(v3[1] == 2.0f);
        
        // Named access
        CHECK(v3.x() == 1.0f);
        CHECK(v3.y() == 2.0f);
        
        // Factory methods
        auto zero = vec2<float>::zero();
        CHECK(zero.x() == 0.0f);
        CHECK(zero.y() == 0.0f);
        
        auto ones = vec2<float>::ones();
        CHECK(ones.x() == 1.0f);
        CHECK(ones.y() == 1.0f);
        
        auto unit_x = vec2<float>::unit_x();
        CHECK(unit_x.x() == 1.0f);
        CHECK(unit_x.y() == 0.0f);
        
        auto unit_y = vec2<float>::unit_y();
        CHECK(unit_y.x() == 0.0f);
        CHECK(unit_y.y() == 1.0f);
        
        // Vector operations
        CHECK(v3.length_squared() == 5.0f);
        CHECK(v3.length() == doctest::Approx(std::sqrt(5.0f)));
        
        // Arithmetic
        vec2<float> sum = v3 + vec2<float>(3.0f, 4.0f);
        CHECK(sum.x() == 4.0f);
        CHECK(sum.y() == 6.0f);
        
        vec2<float> diff = v3 - vec2<float>(0.5f, 1.0f);
        CHECK(diff.x() == 0.5f);
        CHECK(diff.y() == 1.0f);
        
        vec2<float> scaled = v3 * 2.0f;
        CHECK(scaled.x() == 2.0f);
        CHECK(scaled.y() == 4.0f);
        
        // Normalization
        vec2<float> normalized = v3.normalized();
        CHECK(normalized.length() == doctest::Approx(1.0f));
    }
    
    SUBCASE("3D vector basics") {
        // Construction
        vec3<float> v1;
        vec3<float> v2(5.0f);
        vec3<float> v3(1.0f, 2.0f, 3.0f);
        
        // Scalar constructor
        CHECK(v2[0] == 5.0f);
        CHECK(v2[1] == 5.0f);
        CHECK(v2[2] == 5.0f);
        
        // Component constructor
        CHECK(v3[0] == 1.0f);
        CHECK(v3[1] == 2.0f);
        CHECK(v3[2] == 3.0f);
        
        // Named access
        CHECK(v3.x() == 1.0f);
        CHECK(v3.y() == 2.0f);
        CHECK(v3.z() == 3.0f);
        
        // Alternative names (RGB)
        CHECK(v3.r() == 1.0f);
        CHECK(v3.g() == 2.0f);
        CHECK(v3.b() == 3.0f);
        
        // Factory methods
        auto zero = vec3<float>::zero();
        CHECK(zero.x() == 0.0f);
        CHECK(zero.y() == 0.0f);
        CHECK(zero.z() == 0.0f);
        
        auto ones = vec3<float>::ones();
        CHECK(ones.x() == 1.0f);
        CHECK(ones.y() == 1.0f);
        CHECK(ones.z() == 1.0f);
        
        auto unit_x = vec3<float>::unit_x();
        CHECK(unit_x.x() == 1.0f);
        CHECK(unit_x.y() == 0.0f);
        CHECK(unit_x.z() == 0.0f);
        
        auto unit_y = vec3<float>::unit_y();
        CHECK(unit_y.x() == 0.0f);
        CHECK(unit_y.y() == 1.0f);
        CHECK(unit_y.z() == 0.0f);
        
        auto unit_z = vec3<float>::unit_z();
        CHECK(unit_z.x() == 0.0f);
        CHECK(unit_z.y() == 0.0f);
        CHECK(unit_z.z() == 1.0f);
        
        // Vector operations
        CHECK(v3.length_squared() == 14.0f);
        CHECK(v3.length() == doctest::Approx(std::sqrt(14.0f)));
        
        // Arithmetic
        vec3<float> sum = v3 + vec3<float>(3.0f, 4.0f, 5.0f);
        CHECK(sum.x() == 4.0f);
        CHECK(sum.y() == 6.0f);
        CHECK(sum.z() == 8.0f);
        
        vec3<float> diff = v3 - vec3<float>(0.5f, 1.0f, 1.5f);
        CHECK(diff.x() == 0.5f);
        CHECK(diff.y() == 1.0f);
        CHECK(diff.z() == 1.5f);
        
        vec3<float> scaled = v3 * 2.0f;
        CHECK(scaled.x() == 2.0f);
        CHECK(scaled.y() == 4.0f);
        CHECK(scaled.z() == 6.0f);
        
        // Normalization
        vec3<float> normalized = v3.normalized();
        CHECK(normalized.length() == doctest::Approx(1.0f));
    }
    
    SUBCASE("4D vector basics") {
        // Construction
        vec4<float> v1;
        vec4<float> v2(5.0f);
        vec4<float> v3(1.0f, 2.0f, 3.0f, 4.0f);
        
        // Scalar constructor
        CHECK(v2[0] == 5.0f);
        CHECK(v2[1] == 5.0f);
        CHECK(v2[2] == 5.0f);
        CHECK(v2[3] == 5.0f);
        
        // Component constructor
        CHECK(v3[0] == 1.0f);
        CHECK(v3[1] == 2.0f);
        CHECK(v3[2] == 3.0f);
        CHECK(v3[3] == 4.0f);
        
        // Named access
        CHECK(v3.x() == 1.0f);
        CHECK(v3.y() == 2.0f);
        CHECK(v3.z() == 3.0f);
        CHECK(v3.w() == 4.0f);
        
        // Alternative names (RGBA)
        CHECK(v3.r() == 1.0f);
        CHECK(v3.g() == 2.0f);
        CHECK(v3.b() == 3.0f);
        CHECK(v3.a() == 4.0f);
        
        // Factory methods
        auto zero = vec4<float>::zero();
        CHECK(zero.x() == 0.0f);
        CHECK(zero.y() == 0.0f);
        CHECK(zero.z() == 0.0f);
        CHECK(zero.w() == 0.0f);
        
        auto ones = vec4<float>::ones();
        CHECK(ones.x() == 1.0f);
        CHECK(ones.y() == 1.0f);
        CHECK(ones.z() == 1.0f);
        CHECK(ones.w() == 1.0f);
        
        auto unit_x = vec4<float>::unit_x();
        CHECK(unit_x.x() == 1.0f);
        CHECK(unit_x.y() == 0.0f);
        CHECK(unit_x.z() == 0.0f);
        CHECK(unit_x.w() == 0.0f);
        
        auto unit_y = vec4<float>::unit_y();
        CHECK(unit_y.x() == 0.0f);
        CHECK(unit_y.y() == 1.0f);
        CHECK(unit_y.z() == 0.0f);
        CHECK(unit_y.w() == 0.0f);
        
        auto unit_z = vec4<float>::unit_z();
        CHECK(unit_z.x() == 0.0f);
        CHECK(unit_z.y() == 0.0f);
        CHECK(unit_z.z() == 1.0f);
        CHECK(unit_z.w() == 0.0f);
        
        auto unit_w = vec4<float>::unit_w();
        CHECK(unit_w.x() == 0.0f);
        CHECK(unit_w.y() == 0.0f);
        CHECK(unit_w.z() == 0.0f);
        CHECK(unit_w.w() == 1.0f);
        
        // Vector operations
        CHECK(v3.length_squared() == 30.0f);
        CHECK(v3.length() == doctest::Approx(std::sqrt(30.0f)));
        
        // Arithmetic
        vec4<float> sum = v3 + vec4<float>(3.0f, 4.0f, 5.0f, 6.0f);
        CHECK(sum.x() == 4.0f);
        CHECK(sum.y() == 6.0f);
        CHECK(sum.z() == 8.0f);
        CHECK(sum.w() == 10.0f);
        
        vec4<float> diff = v3 - vec4<float>(0.5f, 1.0f, 1.5f, 2.0f);
        CHECK(diff.x() == 0.5f);
        CHECK(diff.y() == 1.0f);
        CHECK(diff.z() == 1.5f);
        CHECK(diff.w() == 2.0f);
        
        vec4<float> scaled = v3 * 2.0f;
        CHECK(scaled.x() == 2.0f);
        CHECK(scaled.y() == 4.0f);
        CHECK(scaled.z() == 6.0f);
        CHECK(scaled.w() == 8.0f);
        
        // Normalization
        vec4<float> normalized = v3.normalized();
        CHECK(normalized.length() == doctest::Approx(1.0f));
    }
}

// Test type traits for all dimensions
TEST_CASE("euler::vector type traits - all dimensions") {
    using namespace euler;
    
    // Test vector dimension trait
    static_assert(vector_dimension_v<vec2<float>> == 2);
    static_assert(vector_dimension_v<vec3<float>> == 3);
    static_assert(vector_dimension_v<vec4<float>> == 4);
    
    static_assert(vector_dimension_v<vec2<double>> == 2);
    static_assert(vector_dimension_v<vec3<double>> == 3);
    static_assert(vector_dimension_v<vec4<double>> == 4);
    
    // Test is_vector trait
    static_assert(is_vector_v<vec2<float>>);
    static_assert(is_vector_v<vec3<float>>);
    static_assert(is_vector_v<vec4<float>>);
    
    static_assert(is_vector_v<vector<float, 2>>);
    static_assert(is_vector_v<vector<float, 3>>);
    static_assert(is_vector_v<vector<float, 4>>);
    
    // Test expression vector size
    static_assert(expression_vector_size_v<vec2<float>> == 2);
    static_assert(expression_vector_size_v<vec3<float>> == 3);
    static_assert(expression_vector_size_v<vec4<float>> == 4);
}

// Test conversions between vector types
TEST_CASE("euler::vector conversions - all dimensions") {
    using namespace euler;
    
    SUBCASE("Row/column vector conversions") {
        // 2D
        vec2<float> v2(1.0f, 2.0f);
        column_vector<float, 2> cv2(v2);
        row_vector<float, 2> rv2(v2);
        
        CHECK(cv2[0] == v2[0]);
        CHECK(cv2[1] == v2[1]);
        CHECK(rv2[0] == v2[0]);
        CHECK(rv2[1] == v2[1]);
        
        // 3D
        vec3<float> v3(1.0f, 2.0f, 3.0f);
        column_vector<float, 3> cv3(v3);
        row_vector<float, 3> rv3(v3);
        
        CHECK(cv3[0] == v3[0]);
        CHECK(cv3[1] == v3[1]);
        CHECK(cv3[2] == v3[2]);
        CHECK(rv3[0] == v3[0]);
        CHECK(rv3[1] == v3[1]);
        CHECK(rv3[2] == v3[2]);
        
        // 4D
        vec4<float> v4(1.0f, 2.0f, 3.0f, 4.0f);
        column_vector<float, 4> cv4(v4);
        row_vector<float, 4> rv4(v4);
        
        CHECK(cv4[0] == v4[0]);
        CHECK(cv4[1] == v4[1]);
        CHECK(cv4[2] == v4[2]);
        CHECK(cv4[3] == v4[3]);
        CHECK(rv4[0] == v4[0]);
        CHECK(rv4[1] == v4[1]);
        CHECK(rv4[2] == v4[2]);
        CHECK(rv4[3] == v4[3]);
    }
    
    SUBCASE("Type conversions") {
        // Float to double conversions
        vec2f v2f(1.0f, 2.0f);
        vec2d v2d(1.0, 2.0);
        
        vec3f v3f(1.0f, 2.0f, 3.0f);
        vec3d v3d(1.0, 2.0, 3.0);
        
        vec4f v4f(1.0f, 2.0f, 3.0f, 4.0f);
        vec4d v4d(1.0, 2.0, 3.0, 4.0);
        
        // Integer vectors
        vec2i v2i(1, 2);
        vec3i v3i(1, 2, 3);
        vec4i v4i(1, 2, 3, 4);
        
        CHECK(v2i[0] == 1);
        CHECK(v2i[1] == 2);
        CHECK(v3i[0] == 1);
        CHECK(v3i[1] == 2);
        CHECK(v3i[2] == 3);
        CHECK(v4i[0] == 1);
        CHECK(v4i[1] == 2);
        CHECK(v4i[2] == 3);
        CHECK(v4i[3] == 4);
    }
}