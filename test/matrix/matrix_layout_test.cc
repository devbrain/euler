#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>

TEST_CASE("euler::matrix storage layout") {
    using namespace euler;
    
    SUBCASE("default layout adapts to system") {
        // The default matrix type uses the system's default layout
        // We can't test the specific layout without build configuration
        // Instead, we verify both explicit layouts work correctly
        matrix<float, 2, 2> m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        // Element access should work regardless of layout
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 2.0f);
        CHECK(m(1, 0) == 3.0f);
        CHECK(m(1, 1) == 4.0f);
    }
    
    SUBCASE("explicit column-major layout") {
        matrix_col<float, 2, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        
        CHECK(m.column_major == true);
        
        // Check storage order (column-major)
        CHECK(m[0] == 1.0f);  // (0,0)
        CHECK(m[1] == 4.0f);  // (1,0)
        CHECK(m[2] == 2.0f);  // (0,1)
        CHECK(m[3] == 5.0f);  // (1,1)
        CHECK(m[4] == 3.0f);  // (0,2)
        CHECK(m[5] == 6.0f);  // (1,2)
        
        // col_data should work
        auto col0 = m.col_data(0);
        CHECK(col0[0] == 1.0f);
        CHECK(col0[1] == 4.0f);
    }
    
    SUBCASE("explicit row-major layout") {
        matrix_row<float, 2, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        
        CHECK(m.column_major == false);
        
        // Check storage order (row-major)
        CHECK(m[0] == 1.0f);  // (0,0)
        CHECK(m[1] == 2.0f);  // (0,1)
        CHECK(m[2] == 3.0f);  // (0,2)
        CHECK(m[3] == 4.0f);  // (1,0)
        CHECK(m[4] == 5.0f);  // (1,1)
        CHECK(m[5] == 6.0f);  // (1,2)
        
        // row_data should work
        auto row0 = m.row_data(0);
        CHECK(row0[0] == 1.0f);
        CHECK(row0[1] == 2.0f);
        CHECK(row0[2] == 3.0f);
    }
    
    SUBCASE("layout conversion") {
        matrix_col<float, 3, 2> col_major{
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        };
        
        // Convert to row-major
        auto row_major = col_major.to_row_major();
        CHECK(row_major.column_major == false);
        
        // Values should be the same when accessed by (i,j)
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(col_major(i, j) == row_major(i, j));
            }
        }
        
        // But storage order is different
        CHECK(col_major[0] == 1.0f);  // col-major: (0,0)
        CHECK(row_major[0] == 1.0f);  // row-major: (0,0)
        CHECK(col_major[1] == 3.0f);  // col-major: (1,0)
        CHECK(row_major[1] == 2.0f);  // row-major: (0,1)
        
        // Convert back
        auto col_major2 = row_major.to_column_major();
        CHECK(col_major2.column_major == true);
        
        // Should be identical to original
        for (size_t i = 0; i < 6; ++i) {
            CHECK(col_major[i] == col_major2[i]);
        }
    }
    
    SUBCASE("from_row_major with different layouts") {
        auto col_layout = matrix_col<float, 2, 2>::from_row_major({1, 2, 3, 4});
        auto row_layout = matrix_row<float, 2, 2>::from_row_major({1, 2, 3, 4});
        
        // Same values when accessed by (i,j)
        CHECK(col_layout(0, 0) == 1.0f);
        CHECK(col_layout(0, 1) == 2.0f);
        CHECK(col_layout(1, 0) == 3.0f);
        CHECK(col_layout(1, 1) == 4.0f);
        
        CHECK(row_layout(0, 0) == 1.0f);
        CHECK(row_layout(0, 1) == 2.0f);
        CHECK(row_layout(1, 0) == 3.0f);
        CHECK(row_layout(1, 1) == 4.0f);
        
        // Different storage
        CHECK(col_layout[0] == 1.0f);  // (0,0)
        CHECK(col_layout[1] == 3.0f);  // (1,0) - column major
        CHECK(row_layout[0] == 1.0f);  // (0,0)
        CHECK(row_layout[1] == 2.0f);  // (0,1) - row major
    }
    
    SUBCASE("from_col_major with different layouts") {
        auto col_layout = matrix_col<float, 2, 2>::from_col_major({1, 3, 2, 4});
        auto row_layout = matrix_row<float, 2, 2>::from_col_major({1, 3, 2, 4});
        
        // Same values when accessed by (i,j)
        CHECK(col_layout(0, 0) == 1.0f);
        CHECK(col_layout(0, 1) == 2.0f);
        CHECK(col_layout(1, 0) == 3.0f);
        CHECK(col_layout(1, 1) == 4.0f);
        
        CHECK(row_layout(0, 0) == 1.0f);
        CHECK(row_layout(0, 1) == 2.0f);
        CHECK(row_layout(1, 0) == 3.0f);
        CHECK(row_layout(1, 1) == 4.0f);
    }
    
    SUBCASE("type aliases") {
        // Check that explicit layout aliases work
        matrix2_col<float> m_col{{1, 2}, {3, 4}};
        matrix2_row<float> m_row{{1, 2}, {3, 4}};
        
        CHECK(m_col.column_major == true);
        CHECK(m_row.column_major == false);
    }
}