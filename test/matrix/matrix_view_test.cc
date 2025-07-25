#include <doctest/doctest.h>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
#include <vector>

TEST_CASE("euler::matrix_view basic functionality") {
    using namespace euler;
    
    SUBCASE("view from matrix") {
        matrix<float, 3, 4> m{
            {1.0f,  2.0f,  3.0f,  4.0f},
            {5.0f,  6.0f,  7.0f,  8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f}
        };
        
        matrix_view<float> view(m);
        
        CHECK(view.rows() == 3);
        CHECK(view.cols() == 4);
        CHECK(view.size() == 12);
        CHECK(view.is_contiguous());
        CHECK(view.is_column_major());
        
        // Check values
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                CHECK(view(i, j) == m(i, j));
            }
        }
        
        // Modify through view
        view(1, 2) = 99.0f;
        CHECK(m(1, 2) == 99.0f);
    }
    
    SUBCASE("view from raw data") {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        matrix_view<float> view(data.data(), 2, 3);
        
        CHECK(view.rows() == 2);
        CHECK(view.cols() == 3);
        CHECK(view.is_column_major());
        
        // Column-major interpretation
        CHECK(view(0, 0) == 1.0f);
        CHECK(view(1, 0) == 2.0f);  // Next in column
        CHECK(view(0, 1) == 3.0f);
        CHECK(view(1, 1) == 4.0f);
        CHECK(view(0, 2) == 5.0f);
        CHECK(view(1, 2) == 6.0f);
    }
    
    SUBCASE("const view") {
        const matrix<float, 2, 2> m{{1, 2}, {3, 4}};
        const_matrix_view<float> view(m);
        
        CHECK(view.rows() == 2);
        CHECK(view.cols() == 2);
        CHECK(view(0, 0) == 1.0f);
        CHECK(view(1, 1) == 4.0f);
        
        // Should not be able to modify
        // view(0, 0) = 5.0f; // This should not compile
    }
}

TEST_CASE("euler::matrix_view subviews") {
    using namespace euler;
    
    matrix<float, 4, 5> m{
        {1.0f,  2.0f,  3.0f,  4.0f,  5.0f},
        {6.0f,  7.0f,  8.0f,  9.0f, 10.0f},
        {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
        {16.0f, 17.0f, 18.0f, 19.0f, 20.0f}
    };
    
    matrix_view<float> full_view(m);
    
    SUBCASE("submatrix view") {
        auto sub = full_view.submatrix(1, 1, 2, 3);
        
        CHECK(sub.rows() == 2);
        CHECK(sub.cols() == 3);
        
        // Check values
        CHECK(sub(0, 0) == 7.0f);
        CHECK(sub(0, 1) == 8.0f);
        CHECK(sub(0, 2) == 9.0f);
        CHECK(sub(1, 0) == 12.0f);
        CHECK(sub(1, 1) == 13.0f);
        CHECK(sub(1, 2) == 14.0f);
        
        // Modify through subview
        sub(0, 1) = 99.0f;
        CHECK(m(1, 2) == 99.0f);
    }
    
    SUBCASE("row view") {
        auto row1 = full_view.row(1);
        
        CHECK(row1.rows() == 1);
        CHECK(row1.cols() == 5);
        
        for (size_t j = 0; j < 5; ++j) {
            CHECK(row1(0, j) == m(1, j));
        }
        
        // Modify through row view
        row1(0, 3) = 88.0f;
        CHECK(m(1, 3) == 88.0f);
    }
    
    SUBCASE("column view") {
        auto col2 = full_view.col(2);
        
        CHECK(col2.rows() == 4);
        CHECK(col2.cols() == 1);
        
        for (size_t i = 0; i < 4; ++i) {
            CHECK(col2(i, 0) == m(i, 2));
        }
        
        // Modify through column view
        col2(2, 0) = 77.0f;
        CHECK(m(2, 2) == 77.0f);
    }
    
    SUBCASE("diagonal view") {
        matrix<float, 4, 4> sq{
            {1.0f,  2.0f,  3.0f,  4.0f},
            {5.0f,  6.0f,  7.0f,  8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };
        
        matrix_view<float> sq_view(sq);
        
        // Main diagonal
        auto diag = sq_view.diagonal();
        CHECK(diag.rows() == 4);
        CHECK(diag.cols() == 1);
        CHECK(diag(0, 0) == 1.0f);
        CHECK(diag(1, 0) == 6.0f);
        CHECK(diag(2, 0) == 11.0f);
        CHECK(diag(3, 0) == 16.0f);
        
        // Super-diagonal (offset = 1)
        auto super_diag = sq_view.diagonal(1);
        CHECK(super_diag.rows() == 3);
        CHECK(super_diag.cols() == 1);
        CHECK(super_diag(0, 0) == 2.0f);
        CHECK(super_diag(1, 0) == 7.0f);
        CHECK(super_diag(2, 0) == 12.0f);
        
        // Sub-diagonal (offset = -1)
        auto sub_diag = sq_view.diagonal(-1);
        CHECK(sub_diag.rows() == 3);
        CHECK(sub_diag.cols() == 1);
        CHECK(sub_diag(0, 0) == 5.0f);
        CHECK(sub_diag(1, 0) == 10.0f);
        CHECK(sub_diag(2, 0) == 15.0f);
    }
}

TEST_CASE("euler::matrix_view strides") {
    using namespace euler;
    
    SUBCASE("custom strides") {
        float data[] = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12
        };
        
        // View every other element in rows, every other row
        // row_stride=6 to skip a whole row (6 elements), col_stride=2 to skip every other element
        matrix_view<float> strided(data, 2, 3, 6, 2);
        
        CHECK(strided(0, 0) == 1.0f);
        CHECK(strided(0, 1) == 3.0f);
        CHECK(strided(0, 2) == 5.0f);
        CHECK(strided(1, 0) == 7.0f);
        CHECK(strided(1, 1) == 9.0f);
        CHECK(strided(1, 2) == 11.0f);
        
        CHECK(!strided.is_contiguous());
    }
    
    SUBCASE("transpose via strides") {
        matrix<float, 2, 3> m{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        
        // Create transposed view by swapping strides
        // For column-major 2x3 matrix, data layout is [1,4,2,5,3,6]
        // To get 3x2 transpose: row_stride=2, col_stride=1
        matrix_view<float> transposed(m.data(), 3, 2, 2, 1);
        
        CHECK(transposed.rows() == 3);
        CHECK(transposed.cols() == 2);
        
        // Check transposed values
        CHECK(transposed(0, 0) == m(0, 0));  // 1
        CHECK(transposed(0, 1) == m(1, 0));  // 4
        CHECK(transposed(1, 0) == m(0, 1));  // 2
        CHECK(transposed(1, 1) == m(1, 1));  // 5
        CHECK(transposed(2, 0) == m(0, 2));  // 3
        CHECK(transposed(2, 1) == m(1, 2));  // 6
    }
}

TEST_CASE("euler::matrix_view assignment") {
    using namespace euler;
    
    SUBCASE("assign from expression") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::zero();
        matrix_view<float> view(m);
        
        matrix<float, 3, 3> a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        matrix<float, 3, 3> b{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        
        // Assign expression result to view
        view = a + b;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(m(i, j) == 10.0f);
            }
        }
    }
    
    SUBCASE("assign scalar") {
        matrix<float, 2, 4> m = matrix<float, 2, 4>::zero();
        matrix_view<float> view(m);
        
        view = 5.0f;
        
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                CHECK(m(i, j) == 5.0f);
            }
        }
    }
    
    SUBCASE("partial assignment through subview") {
        matrix<float, 4, 4> m = matrix<float, 4, 4>::identity();
        matrix_view<float> view(m);
        
        // Set a 2x2 block to all 9s
        auto block = view.submatrix(1, 1, 2, 2);
        block = 9.0f;
        
        // Check the block
        CHECK(m(1, 1) == 9.0f);
        CHECK(m(1, 2) == 9.0f);
        CHECK(m(2, 1) == 9.0f);
        CHECK(m(2, 2) == 9.0f);
        
        // Check that other elements are unchanged
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(3, 3) == 1.0f);
        CHECK(m(0, 1) == 0.0f);
    }
}

TEST_CASE("euler::matrix_view helper functions") {
    using namespace euler;
    
    SUBCASE("make_view") {
        matrix<float, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
        
        auto view = make_view(m);
        CHECK(view.rows() == 2);
        CHECK(view.cols() == 3);
        
        const auto& const_m = m;
        auto const_view = make_view(const_m);
        CHECK(const_view.rows() == 2);
        CHECK(const_view.cols() == 3);
    }
}

TEST_CASE("euler::matrix_view edge cases") {
    using namespace euler;
    
    SUBCASE("single element view") {
        matrix<float, 3, 3> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        matrix_view<float> view(m);
        
        auto single = view.submatrix(1, 1, 1, 1);
        CHECK(single.rows() == 1);
        CHECK(single.cols() == 1);
        CHECK(single(0, 0) == 5.0f);
        
        single(0, 0) = 99.0f;
        CHECK(m(1, 1) == 99.0f);
    }
    
    SUBCASE("empty matrix handling") {
        // Views require non-zero dimensions due to our safety checks
        // Test the smallest valid views instead
        float dummy = 1.0f;
        matrix_view<float> single(&dummy, 1, 1);
        CHECK(single.rows() == 1);
        CHECK(single.cols() == 1);
        CHECK(single.size() == 1);
        CHECK(single(0, 0) == 1.0f);
    }
    
    SUBCASE("view properties") {
        matrix<float, 3, 4> col_major_mat;
        matrix_view<float> col_view(col_major_mat);
        CHECK(col_view.is_column_major());
        CHECK(!col_view.is_row_major());
        
        // Create a row-major view using strides
        float data[12];
        matrix_view<float> row_view(data, 3, 4, 4, 1);
        CHECK(!row_view.is_column_major());
        CHECK(row_view.is_row_major());
    }
    
    SUBCASE("alignment checks") {
        // Aligned matrix should create aligned view
        matrix<float, 4, 4> aligned_mat;
        matrix_view<float> aligned_view(aligned_mat);
        CHECK(aligned_view.is_simd_aligned());
        
        // Const view should also report alignment
        const matrix<float, 4, 4> const_mat;
        const_matrix_view<float> const_view(const_mat);
        CHECK(const_view.is_simd_aligned());
        
        // Stack array might not be aligned
        float unaligned_data[16];
        // Force misalignment by using an odd address
        float* misaligned_ptr = unaligned_data + (is_aligned(unaligned_data) ? 1 : 0);
        matrix_view<float> unaligned_view(misaligned_ptr, 3, 3, 1, 0, false);  // expect_aligned = false
        // This view might not be aligned
        
        // Aligned allocation
        alignas(16) float aligned_data[16];  // 16-byte alignment for float SIMD
        matrix_view<float> manual_aligned_view(aligned_data, 4, 4);
        CHECK(manual_aligned_view.is_simd_aligned());
        
        // Check subview alignment
        matrix<float, 8, 8> large_mat;
        matrix_view<float> large_view(large_mat);
        
        // The main view should be aligned
        CHECK(large_view.is_simd_aligned());
        
        // Check if various subviews would be aligned
        // This depends on the layout and element size
        bool sub00_aligned = large_view.is_subview_aligned(0, 0);
        bool sub11_aligned = large_view.is_subview_aligned(1, 1);
        
        // At least the (0,0) subview should be aligned
        CHECK(sub00_aligned);
        
        // The (1,1) subview alignment depends on the layout and size
        // Just verify we can query it without errors
        (void)sub11_aligned;  // Suppress unused variable warning
        
        // Check alignment offset
        size_t offset00 = large_view.alignment_offset(0, 0);
        CHECK(offset00 == 0);  // First element should be aligned
    }
}