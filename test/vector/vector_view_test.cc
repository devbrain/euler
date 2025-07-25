#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>

TEST_CASE("euler::matrix_view as vector") {
    using namespace euler;
    
    SUBCASE("column view as vector") {
        matrix<float, 4, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f},
            {10.0f, 11.0f, 12.0f}
        };
        
        // Get column 1 as a vector view
        auto col_view = m.col(1);
        
        CHECK(col_view.is_vector());
        CHECK(col_view.vector_size() == 4);
        
        // Check values
        CHECK(col_view[0] == 2.0f);
        CHECK(col_view[1] == 5.0f);
        CHECK(col_view[2] == 8.0f);
        CHECK(col_view[3] == 11.0f);
        
        // Vector operations on the view
        float len_sq = col_view.length_squared();
        CHECK(len_sq == doctest::Approx(214.0f)); // 2^2 + 5^2 + 8^2 + 11^2 = 4 + 25 + 64 + 121
        
        float len = col_view.length();
        CHECK(len == doctest::Approx(std::sqrt(214.0f)));
    }
    
    SUBCASE("row view as vector") {
        matrix<float, 3, 5> m{
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
            {6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
            {11.0f, 12.0f, 13.0f, 14.0f, 15.0f}
        };
        
        // Get row 1 as a vector view
        auto row_view = m.row(1);
        
        CHECK(row_view.is_vector());
        CHECK(row_view.vector_size() == 5);
        
        // Check values
        CHECK(row_view[0] == 6.0f);
        CHECK(row_view[1] == 7.0f);
        CHECK(row_view[2] == 8.0f);
        CHECK(row_view[3] == 9.0f);
        CHECK(row_view[4] == 10.0f);
    }
    
    SUBCASE("diagonal view as vector") {
        matrix<float, 4, 4> m{
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };
        
        auto diag_view = m.diagonal();
        
        CHECK(diag_view.is_vector());
        CHECK(diag_view.vector_size() == 4);
        
        // Check diagonal values
        CHECK(diag_view[0] == 1.0f);
        CHECK(diag_view[1] == 6.0f);
        CHECK(diag_view[2] == 11.0f);
        CHECK(diag_view[3] == 16.0f);
    }
    
    SUBCASE("dot product with matrix views") {
        matrix<float, 3, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };
        
        auto col0 = m.col(0);  // [1, 4, 7]
        auto col1 = m.col(1);  // [2, 5, 8]
        
        // Dot product between two column views
        float d1 = dot(col0, col1);
        CHECK(d1 == doctest::Approx(78.0f)); // 1*2 + 4*5 + 7*8
        
        // Dot product with a regular vector
        vector3 v(1.0f, 1.0f, 1.0f);
        float d2 = dot(col0, v);
        CHECK(d2 == doctest::Approx(12.0f)); // 1*1 + 4*1 + 7*1
        
        // Row views
        auto row0 = m.row(0);  // [1, 2, 3]
        auto row1 = m.row(1);  // [4, 5, 6]
        
        float d3 = dot(row0, row1);
        CHECK(d3 == doctest::Approx(32.0f)); // 1*4 + 2*5 + 3*6
    }
    
    SUBCASE("mixed operations") {
        matrix<float, 4, 3> m1{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f},
            {10.0f, 11.0f, 12.0f}
        };
        
        matrix<float, 3, 4> m2{
            {1.0f, 4.0f, 7.0f, 10.0f},
            {2.0f, 5.0f, 8.0f, 11.0f},
            {3.0f, 6.0f, 9.0f, 12.0f}
        };
        
        // Column of m1 dot row of m2
        auto col = m1.col(1);   // [2, 5, 8, 11]
        auto row = m2.row(1);   // [2, 5, 8, 11]
        
        float d = dot(col, row);
        CHECK(d == doctest::Approx(214.0f)); // 2^2 + 5^2 + 8^2 + 11^2 = 4 + 25 + 64 + 121
    }
    
    SUBCASE("submatrix single row/column as vector") {
        matrix<float, 5, 5> m{
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f},
            {6.0f,  7.0f,  8.0f,  9.0f,  10.0f},
            {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
            {16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
            {21.0f, 22.0f, 23.0f, 24.0f, 25.0f}
        };
        
        // Get a 1x3 submatrix (row vector)
        auto sub_row = m.submatrix(2, 1, 1, 3);
        CHECK(sub_row.is_vector());
        CHECK(sub_row.vector_size() == 3);
        CHECK(sub_row[0] == 12.0f);
        CHECK(sub_row[1] == 13.0f);
        CHECK(sub_row[2] == 14.0f);
        
        // Get a 3x1 submatrix (column vector)
        auto sub_col = m.submatrix(1, 2, 3, 1);
        CHECK(sub_col.is_vector());
        CHECK(sub_col.vector_size() == 3);
        CHECK(sub_col[0] == 8.0f);
        CHECK(sub_col[1] == 13.0f);
        CHECK(sub_col[2] == 18.0f);
    }
}