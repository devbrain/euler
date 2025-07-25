#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/vector/scalar_vector_expr.hh>
#include <euler/core/approx_equal.hh>
#include <cmath>

using namespace euler;

TEST_CASE("Scalar-vector operations") {
    SUBCASE("Scalar division by vector") {
        vector<float, 3> v{1.0f, 2.0f, 4.0f};
        
        // 1 / v should give {1/1, 1/2, 1/4} = {1, 0.5, 0.25}
        auto result = 1.0f / v;
        vector<float, 3> expected{1.0f, 0.5f, 0.25f};
        
        CHECK(approx_equal(vector<float, 3>(result), expected));
    }
    
    SUBCASE("Scalar subtraction from vector") {
        vector<float, 4> v{1.0f, 3.0f, 5.0f, 7.0f};
        
        // 10 - v should give {10-1, 10-3, 10-5, 10-7} = {9, 7, 5, 3}
        auto result = 10.0f - v;
        vector<float, 4> expected{9.0f, 7.0f, 5.0f, 3.0f};
        
        CHECK(approx_equal(vector<float, 4>(result), expected));
    }
    
    SUBCASE("Scalar power with vector exponent") {
        vector<float, 3> v{1.0f, 2.0f, 3.0f};
        
        // 2 ^ v should give {2^1, 2^2, 2^3} = {2, 4, 8}
        auto result = 2.0f ^ v;
        vector<float, 3> expected{2.0f, 4.0f, 8.0f};
        
        CHECK(approx_equal(vector<float, 3>(result), expected, 1e-5f));
    }
    
    SUBCASE("Reciprocal function") {
        vector<float, 4> v{2.0f, 4.0f, 5.0f, 10.0f};
        
        auto result = reciprocal(v);
        vector<float, 4> expected{0.5f, 0.25f, 0.2f, 0.1f};
        
        CHECK(approx_equal(vector<float, 4>(result), expected));
    }
    
    SUBCASE("Element inverse function") {
        vector<double, 3> v{0.5, 0.25, 0.125};
        
        auto result = element_inverse(v);
        vector<double, 3> expected{2.0, 4.0, 8.0};
        
        CHECK(approx_equal(vector<double, 3>(result), expected));
    }
    
    SUBCASE("Chained scalar operations") {
        vector<float, 3> v{1.0f, 2.0f, 3.0f};
        
        // (10 - v) / 2
        auto result = (10.0f - v) / 2.0f;
        vector<float, 3> expected{4.5f, 4.0f, 3.5f};
        
        CHECK(approx_equal(vector<float, 3>(result), expected));
    }
    
    SUBCASE("Scalar operations with expressions") {
        vector<float, 3> a{1.0f, 2.0f, 3.0f};
        vector<float, 3> b{2.0f, 3.0f, 4.0f};
        
        // 1 / (a + b) = 1 / {3, 5, 7} = {1/3, 1/5, 1/7}
        auto result = 1.0f / (a + b);
        vector<float, 3> expected{1.0f/3.0f, 1.0f/5.0f, 1.0f/7.0f};
        
        CHECK(approx_equal(vector<float, 3>(result), expected));
    }
    
    SUBCASE("Mixed type scalar operations") {
        vector<float, 3> v{1.0f, 2.0f, 4.0f};
        
        // Double scalar with float vector
        auto result = 8.0 / v;  // double / vector<float>
        vector<double, 3> expected{8.0, 4.0, 2.0};
        
        CHECK(approx_equal(vector<double, 3>(result), expected));
    }
    
    SUBCASE("Zero handling") {
        vector<float, 3> v{2.0f, 0.0f, -2.0f};
        
        auto result = 1.0f / v;
        CHECK(std::isinf(result[1]));  // 1/0 = inf
        CHECK(result[0] == 0.5f);
        CHECK(result[2] == -0.5f);
    }
    
    SUBCASE("Negative values") {
        vector<float, 4> v{-1.0f, -2.0f, -4.0f, -8.0f};
        
        auto result = -16.0f / v;
        vector<float, 4> expected{16.0f, 8.0f, 4.0f, 2.0f};
        
        CHECK(approx_equal(vector<float, 4>(result), expected));
    }
}

TEST_CASE("Scalar-vector operations with different sizes") {
    SUBCASE("Size 2") {
        vector<float, 2> v{3.0f, 6.0f};
        auto result = 12.0f / v;
        CHECK(approx_equal(vector<float, 2>(result), vector<float, 2>{4.0f, 2.0f}));
    }
    
    SUBCASE("Size 5") {
        vector<float, 5> v{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto result = 20.0f - v;
        CHECK(approx_equal(vector<float, 5>(result), 
                          vector<float, 5>{19.0f, 18.0f, 17.0f, 16.0f, 15.0f}));
    }
    
    SUBCASE("Size 8") {
        vector<float, 8> v;
        for (size_t i = 0; i < 8; ++i) {
            v[i] = static_cast<float>(i + 1);
        }
        auto result = reciprocal(v);
        
        vector<float, 8> expected;
        for (size_t i = 0; i < 8; ++i) {
            expected[i] = 1.0f / static_cast<float>(i + 1);
        }
        CHECK(approx_equal(vector<float, 8>(result), expected));
    }
}