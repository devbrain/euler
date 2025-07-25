#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>

TEST_CASE("euler::vector expression templates") {
    using namespace euler;
    
    SUBCASE("basic expression template operations") {
        vector3 a(1, 2, 3);
        vector3 b(4, 5, 6);
        vector3 c(7, 8, 9);
        
        // This should use expression templates to avoid temporaries
        vector3 result = a + b * 2.0f - c;
        
        CHECK(result.x() == doctest::Approx(1 + 4*2 - 7));  // 2
        CHECK(result.y() == doctest::Approx(2 + 5*2 - 8));  // 4
        CHECK(result.z() == doctest::Approx(3 + 6*2 - 9));  // 6
    }
    
    SUBCASE("normalized expression") {
        vector3 v(3, 4, 0);
        
        // Using unified normalize - returns expression
        auto norm_expr = normalize(v);
        vector3 normalized = norm_expr;
        
        CHECK(normalized.x() == doctest::Approx(0.6f));
        CHECK(normalized.y() == doctest::Approx(0.8f));
        CHECK(normalized.z() == doctest::Approx(0.0f));
        CHECK(length(normalized) == doctest::Approx(1.0f));
    }
    
    SUBCASE("cross product expression") {
        vector3 a(1, 0, 0);
        vector3 b(0, 1, 0);
        
        // Unified cross - returns expression
        auto cross_expr = cross(a, b);
        vector3 c = cross_expr;
        
        CHECK(c.x() == doctest::Approx(0));
        CHECK(c.y() == doctest::Approx(0));
        CHECK(c.z() == doctest::Approx(1));
    }
    
    SUBCASE("combined expressions") {
        vector3 a(1, 0, 0);
        vector3 b(0, 1, 0);
        vector3 c(1, 1, 1);
        
        // Compute cross product and normalize separately
        vector3 cross_result = cross(a, b);  // (0, 0, 1)
        vector3 normalized = cross_result.normalized();  // Still (0, 0, 1)
        
        // Use expression templates for the arithmetic
        auto expr = normalized + c * 0.5f;
        vector3 result = expr;
        
        CHECK(result.x() == doctest::Approx(0.5f));   // 0 + 0.5
        CHECK(result.y() == doctest::Approx(0.5f));   // 0 + 0.5
        CHECK(result.z() == doctest::Approx(1.5f));   // 1 + 0.5
    }
    
    SUBCASE("reflection expression") {
        vector3 incident(1, -1, 0);
        vector3 normal(0, 1, 0);
        
        auto refl_expr = reflect(incident, normal);
        vector3 reflected = refl_expr;
        
        // Should reflect across y-axis: (1, -1, 0) -> (1, 1, 0)
        CHECK(reflected.x() == doctest::Approx(1));
        CHECK(reflected.y() == doctest::Approx(1));
        CHECK(reflected.z() == doctest::Approx(0));
    }
    
    SUBCASE("lerp expression") {
        vector3 start(0, 0, 0);
        vector3 end(10, 20, 30);
        
        auto lerp_expr = lerp(start, end, 0.5f);
        vector3 mid = lerp_expr;
        
        CHECK(mid.x() == doctest::Approx(5));
        CHECK(mid.y() == doctest::Approx(10));
        CHECK(mid.z() == doctest::Approx(15));
    }
    
    SUBCASE("chained operations") {
        vector3 v1(1, 0, 0);
        vector3 v2(0, 1, 0);
        vector3 v3(0, 0, 1);
        
        // Complex expression with multiple operations
        auto expr = cross(v1, v2) + cross(v2, v3) + cross(v3, v1);
        vector3 result = expr;
        
        // cross(v1,v2) = (0,0,1)
        // cross(v2,v3) = (1,0,0) 
        // cross(v3,v1) = (0,1,0)
        // sum = (1,1,1)
        CHECK(result.x() == doctest::Approx(1));
        CHECK(result.y() == doctest::Approx(1));
        CHECK(result.z() == doctest::Approx(1));
    }
    
    SUBCASE("expression with dot product") {
        vector3 a(1, 2, 3);
        vector3 b(4, 5, 6);
        vector3 c(1, 1, 1);
        
        // dot(a, b) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // result = 32 * c = (32, 32, 32)
        float d = dot(a, b);
        vector3 result = d * c;
        
        CHECK(result.x() == doctest::Approx(32));
        CHECK(result.y() == doctest::Approx(32));
        CHECK(result.z() == doctest::Approx(32));
    }
    

}