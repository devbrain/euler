#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
#include <euler/core/types.hh>

TEST_CASE("euler::matrix construction") {
    using namespace euler;
    
    SUBCASE("default construction") {
        matrix<float, 3, 3> m;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(m(i, j) == 0.0f);
            }
        }
    }
    
    SUBCASE("fill construction") {
        matrix<float, 2, 3> m(5.0f);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(m(i, j) == 5.0f);
            }
        }
    }
    
    SUBCASE("nested initializer list construction") {
        matrix<float, 2, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        
        // Check row-major input is correctly stored
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 4.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(1, 2) == 6.0f);
    }
    
    SUBCASE("from_rows factory method") {
        auto m = matrix<float, 2, 3>::from_rows({
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        });
        
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 4.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(1, 2) == 6.0f);
    }
    
    SUBCASE("from_cols factory method") {
        auto m = matrix<float, 2, 3>::from_cols({
            {1.0f, 4.0f},      // col 0
            {2.0f, 5.0f},      // col 1
            {3.0f, 6.0f}       // col 2
        });
        
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 4.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(1, 2) == 6.0f);
    }
    
    SUBCASE("from_row_major factory method") {
        auto m = matrix<float, 2, 3>::from_row_major({
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
        });
        
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 4.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(1, 2) == 6.0f);
    }
    
    SUBCASE("from_col_major factory method") {
        auto m = matrix<float, 2, 3>::from_col_major({
            1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f
        });
        
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 4.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(1, 2) == 6.0f);
    }
    
    SUBCASE("identity matrix") {
        auto m = matrix<float, 3, 3>::identity();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (i == j) {
                    CHECK(m(i, j) == 1.0f);
                } else {
                    CHECK(m(i, j) == 0.0f);
                }
            }
        }
    }
    
    SUBCASE("zero matrix") {
        auto m = matrix<float, 2, 4>::zero();
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                CHECK(m(i, j) == 0.0f);
            }
        }
    }
}

TEST_CASE("euler::matrix element access") {
    using namespace euler;
    
    SUBCASE("2D access") {
        matrix<float, 3, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };
        
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(1, 1) == 5.0f);
        CHECK(m(2, 2) == 9.0f);
        
        // Modify elements
        m(1, 1) = 10.0f;
        CHECK(m(1, 1) == 10.0f);
    }
    
    SUBCASE("linear access") {
        matrix<float, 2, 2> m{
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        };
        
        // Column-major storage order
        CHECK(m[0] == 1.0f);  // (0,0)
        CHECK(m[1] == 3.0f);  // (1,0)
        CHECK(m[2] == 2.0f);  // (0,1)
        CHECK(m[3] == 4.0f);  // (1,1)
    }
    
    SUBCASE("column data access") {
        matrix<float, 3, 2> m{
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        };
        
        auto col0 = m.col_data(0);
        CHECK(col0[0] == 1.0f);
        CHECK(col0[1] == 3.0f);
        CHECK(col0[2] == 5.0f);
        
        auto col1 = m.col_data(1);
        CHECK(col1[0] == 2.0f);
        CHECK(col1[1] == 4.0f);
        CHECK(col1[2] == 6.0f);
    }
}

TEST_CASE("euler::matrix expression templates") {
    using namespace euler;
    
    SUBCASE("assignment from expression") {
        matrix<float, 2, 2> a{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> b{{5.0f, 6.0f}, {7.0f, 8.0f}};
        
        matrix<float, 2, 2> c = a + b;
        CHECK(c(0, 0) == 6.0f);
        CHECK(c(0, 1) == 8.0f);
        CHECK(c(1, 0) == 10.0f);
        CHECK(c(1, 1) == 12.0f);
    }
    
    SUBCASE("scalar operations") {
        matrix<float, 2, 2> m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        matrix<float, 2, 2> result = m + 10.0f;
        CHECK(result(0, 0) == 11.0f);
        CHECK(result(0, 1) == 12.0f);
        CHECK(result(1, 0) == 13.0f);
        CHECK(result(1, 1) == 14.0f);
    }
    
    SUBCASE("complex expressions") {
        matrix<float, 2, 2> a{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> b{{2.0f, 3.0f}, {4.0f, 5.0f}};
        matrix<float, 2, 2> c{{1.0f, 1.0f}, {1.0f, 1.0f}};
        
        matrix<float, 2, 2> result = (a + b) * c - 2.0f;
        CHECK(result(0, 0) == 1.0f);   // (1+2)*1 - 2 = 1
        CHECK(result(0, 1) == 3.0f);   // (2+3)*1 - 2 = 3
        CHECK(result(1, 0) == 5.0f);   // (3+4)*1 - 2 = 5
        CHECK(result(1, 1) == 7.0f);   // (4+5)*1 - 2 = 7
    }
}

TEST_CASE("euler::matrix comparison") {
    using namespace euler;
    
    SUBCASE("equality") {
        matrix<float, 2, 2> a{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> b{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> c{{1.0f, 2.0f}, {3.0f, 5.0f}};
        
        CHECK(a == b);
        CHECK(a != c);
    }
}

TEST_CASE("euler::matrix type aliases") {
    using namespace euler;
    
    SUBCASE("square matrices") {
        CHECK(std::is_same_v<mat2, matrix<scalar, 2, 2>>);
        CHECK(std::is_same_v<mat3, matrix<scalar, 3, 3>>);
        CHECK(std::is_same_v<mat4, matrix<scalar, 4, 4>>);
    }
    
    SUBCASE("rectangular matrices") {
        CHECK(std::is_same_v<mat2x3, matrix<scalar, 2, 3>>);
        CHECK(std::is_same_v<mat3x4, matrix<scalar, 3, 4>>);
        CHECK(std::is_same_v<mat4x2, matrix<scalar, 4, 2>>);
    }
}

TEST_CASE("euler::matrix traits") {
    using namespace euler;
    
    SUBCASE("matrix traits") {
        using mat_type = matrix<float, 3, 4>;
        using traits = matrix_traits<mat_type>;
        
        CHECK(traits::is_matrix == true);
        CHECK(traits::rows == 3);
        CHECK(traits::cols == 4);
        CHECK(std::is_same_v<traits::value_type, float>);
    }
    
    SUBCASE("is_matrix trait") {
        CHECK(is_matrix_v<matrix<float, 2, 2>>);
        CHECK(is_matrix_v<mat3>);
        CHECK(!is_matrix_v<float>);
        CHECK(!is_matrix_v<int>);
    }
}

TEST_CASE("euler::matrix SIMD alignment") {
    using namespace euler;
    
    SUBCASE("matrix storage alignment") {
        // Check that matrix storage is properly aligned for SIMD
        matrix<float, 4, 4> m_float;
        CHECK(is_aligned(m_float.data()));
        
        matrix<double, 4, 4> m_double;
        CHECK(is_aligned(m_double.data()));
        
        // Check various sizes
        matrix<float, 3, 3> m3x3;
        CHECK(is_aligned(m3x3.data()));
        
        matrix<float, 16, 16> m_large;
        CHECK(is_aligned(m_large.data()));
    }
    
    SUBCASE("matrix view alignment") {
        matrix<float, 4, 4> m;
        matrix_view<float> view(m);
        CHECK(view.is_simd_aligned());
        
        // Subviews may not be aligned if they don't start at the beginning
        auto sub = view.submatrix(1, 1, 2, 2);
        // This might not be aligned depending on the layout
        // Just verify we can create subviews without errors
        CHECK(sub.rows() == 2);
        CHECK(sub.cols() == 2);
    }
}