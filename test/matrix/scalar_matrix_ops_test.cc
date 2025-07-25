#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/scalar_matrix_expr.hh>
#include <euler/core/approx_equal.hh>
#include <cmath>

using namespace euler;

TEST_CASE("Scalar-matrix operations") {
    SUBCASE("Scalar division by matrix") {
        matrix<float, 2, 2> m{{2.0f, 4.0f}, {8.0f, 16.0f}};
        
        // 1 / m should give {{1/2, 1/4}, {1/8, 1/16}}
        auto result = 1.0f / m;
        matrix<float, 2, 2> expected{{0.5f, 0.25f}, {0.125f, 0.0625f}};
        
        CHECK(approx_equal(matrix<float, 2, 2>(result), expected));
    }
    
    SUBCASE("Scalar subtraction from matrix") {
        matrix<float, 2, 3> m{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        
        // 10 - m
        auto result = 10.0f - m;
        matrix<float, 2, 3> expected{{9.0f, 8.0f, 7.0f}, {6.0f, 5.0f, 4.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 3>(result), expected));
    }
    
    SUBCASE("Scalar power with matrix exponent") {
        matrix<float, 2, 2> m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        // 2 ^ m
        auto result = 2.0f ^ m;
        matrix<float, 2, 2> expected{{2.0f, 4.0f}, {8.0f, 16.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 2>(result), expected, 1e-5f));
    }
    
    SUBCASE("Reciprocal function for matrices") {
        matrix<float, 3, 3> m{{1.0f, 2.0f, 4.0f}, 
                             {5.0f, 10.0f, 20.0f},
                             {25.0f, 50.0f, 100.0f}};
        
        auto result = reciprocal(m);
        matrix<float, 3, 3> expected{{1.0f, 0.5f, 0.25f}, 
                                    {0.2f, 0.1f, 0.05f},
                                    {0.04f, 0.02f, 0.01f}};
        
        CHECK(approx_equal(matrix<float, 3, 3>(result), expected));
    }
    
    SUBCASE("Element inverse for matrices") {
        matrix<double, 2, 2> m{{0.5, 0.25}, {0.125, 0.0625}};
        
        auto result = element_inverse(m);
        matrix<double, 2, 2> expected{{2.0, 4.0}, {8.0, 16.0}};
        
        CHECK(approx_equal(matrix<double, 2, 2>(result), expected));
    }
    
    SUBCASE("Chained scalar operations") {
        matrix<float, 2, 2> m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        // (20 - m) / 2
        auto result = (20.0f - m) / 2.0f;
        matrix<float, 2, 2> expected{{9.5f, 9.0f}, {8.5f, 8.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 2>(result), expected));
    }
    
    SUBCASE("Scalar operations with matrix expressions") {
        matrix<float, 2, 2> A{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> B{{2.0f, 3.0f}, {4.0f, 5.0f}};
        
        // 1 / (A + B)
        auto result = 1.0f / (A + B);
        matrix<float, 2, 2> expected{{1.0f/3.0f, 1.0f/5.0f}, 
                                    {1.0f/7.0f, 1.0f/9.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 2>(result), expected));
    }
    
    SUBCASE("Non-square matrices") {
        matrix<float, 3, 2> m{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        
        auto result = 12.0f - m;
        matrix<float, 3, 2> expected{{11.0f, 10.0f}, {9.0f, 8.0f}, {7.0f, 6.0f}};
        
        CHECK(approx_equal(matrix<float, 3, 2>(result), expected));
    }
    
    SUBCASE("Identity matrix operations") {
        auto I = matrix<float, 3, 3>::identity();
        
        auto result = 2.0f / I;
        // For identity: diagonal = 2/1 = 2, off-diagonal = 2/0 = inf
        
        CHECK(result(0, 0) == 2.0f);
        CHECK(result(1, 1) == 2.0f);
        CHECK(result(2, 2) == 2.0f);
        CHECK(std::isinf(result(0, 1)));
        CHECK(std::isinf(result(0, 2)));
    }
    
    SUBCASE("Large matrices") {
        matrix<float, 5, 5> m = matrix<float, 5, 5>::identity() * 2.0f;
        
        auto result = 10.0f / m;
        
        // Diagonal should be 10/2 = 5, off-diagonal 10/0 = inf
        for (size_t i = 0; i < 5; ++i) {
            CHECK(result(i, i) == 5.0f);
            for (size_t j = 0; j < 5; ++j) {
                if (i != j) {
                    CHECK(std::isinf(result(i, j)));
                }
            }
        }
    }
}

TEST_CASE("Scalar-matrix operations with different layouts") {
    SUBCASE("Row-major scalar operations") {
        matrix<float, 2, 3, true> m{{1.0f, 2.0f, 3.0f}, 
                                    {4.0f, 5.0f, 6.0f}};
        
        auto result = 7.0f - m;
        matrix<float, 2, 3> expected{{6.0f, 5.0f, 4.0f}, 
                                    {3.0f, 2.0f, 1.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 3>(result), expected));
    }
    
    SUBCASE("Column-major scalar operations") {
        matrix<float, 2, 3, false> m{{1.0f, 2.0f, 3.0f}, 
                                     {4.0f, 5.0f, 6.0f}};
        
        auto result = reciprocal(m);
        matrix<float, 2, 3> expected{{1.0f, 0.5f, 1.0f/3.0f}, 
                                    {0.25f, 0.2f, 1.0f/6.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 3>(result), expected));
    }
}

TEST_CASE("Scalar operations edge cases") {
    SUBCASE("Zero matrix") {
        auto zero = matrix<float, 2, 2>::zero();
        
        auto result = 1.0f / zero;
        // All elements should be inf
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(std::isinf(result(i, j)));
            }
        }
    }
    
    SUBCASE("Negative values") {
        matrix<float, 2, 2> m{{-1.0f, -2.0f}, {-3.0f, -4.0f}};
        
        auto result = -12.0f / m;
        matrix<float, 2, 2> expected{{12.0f, 6.0f}, {4.0f, 3.0f}};
        
        CHECK(approx_equal(matrix<float, 2, 2>(result), expected));
    }
}