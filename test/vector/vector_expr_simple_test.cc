#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>

TEST_CASE("euler::vector expression templates integration") {
    using namespace euler;
    
    SUBCASE("vectors already use expression templates") {
        vector3 a(1, 2, 3);
        vector3 b(4, 5, 6);
        vector3 c(7, 8, 9);
        
        // This already uses expression templates from the base expression class!
        // No temporaries are created until assignment
        auto expr = a + b * 2.0f - c;
        
        // The expression is only evaluated here
        vector3 result = expr;
        
        CHECK(result.x() == doctest::Approx(1 + 4*2 - 7));  // 2
        CHECK(result.y() == doctest::Approx(2 + 5*2 - 8));  // 4
        CHECK(result.z() == doctest::Approx(3 + 6*2 - 9));  // 6
    }
    
    SUBCASE("complex vector expressions") {
        vector3 v1(1, 0, 0);
        vector3 v2(0, 1, 0);
        vector3 v3(0, 0, 1);
        
        // All of these operations create expression templates
        auto expr = v1 * 2.0f + v2 * 3.0f + v3 * 4.0f;
        vector3 result = expr;
        
        CHECK(result.x() == doctest::Approx(2));
        CHECK(result.y() == doctest::Approx(3));
        CHECK(result.z() == doctest::Approx(4));
    }
    
    SUBCASE("mixed scalar and vector operations") {
        vector3 a(10, 20, 30);
        float s = 0.1f;
        
        // Both directions work
        vector3 r1 = s * a;  // scalar * vector
        vector3 r2 = a * s;  // vector * scalar
        
        CHECK(r1.x() == doctest::Approx(1));
        CHECK(r1.y() == doctest::Approx(2));
        CHECK(r1.z() == doctest::Approx(3));
        
        CHECK(r2.x() == doctest::Approx(1));
        CHECK(r2.y() == doctest::Approx(2));
        CHECK(r2.z() == doctest::Approx(3));
    }
    

    
    SUBCASE("chained operations") {
        vector3 a(1, 1, 1);
        
        // Long chain of operations - all lazy evaluated
        auto expr = a + a + a + a + a;  // 5 * a
        vector3 result = expr;
        
        CHECK(result.x() == doctest::Approx(5));
        CHECK(result.y() == doctest::Approx(5));
        CHECK(result.z() == doctest::Approx(5));
    }
    
    SUBCASE("combining with vector operations") {
        vector3 a(3, 4, 0);
        vector3 b(1, 0, 0);
        
        // normalize() returns a new vector, not an expression
        // but the arithmetic operations still use expression templates
        vector3 n = a.normalized();
        auto expr = n + b * 0.4f;
        vector3 result = expr;
        
        CHECK(result.x() == doctest::Approx(0.6f + 0.4f));  // 1.0
        CHECK(result.y() == doctest::Approx(0.8f));         // 0.8
        CHECK(result.z() == doctest::Approx(0.0f));         // 0.0
    }
}