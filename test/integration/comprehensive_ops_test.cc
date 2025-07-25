#include <euler/core/approx_equal.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_view.hh>
#include <doctest/doctest.h>
#include <tuple>
#include <cmath>

using namespace euler;

TEST_CASE("Comprehensive integration test - Vector operations") {
    SUBCASE("2D vector operations") {
        vector<float, 2> v1{3, 4};
        vector<float, 2> v2{1, 0};
        
        // Basic arithmetic
        auto sum = v1 + v2;
        auto diff = v1 - v2;
        auto scaled = 2.0f * v1;
        CHECK(approx_equal(vector<float, 2>(sum), vector<float, 2>{4, 4}));
        CHECK(approx_equal(vector<float, 2>(diff), vector<float, 2>{2, 4}));
        CHECK(approx_equal(vector<float, 2>(scaled), vector<float, 2>{6, 8}));
        
        // Dot product
        CHECK(dot(v1, v2) == doctest::Approx(3));
        
        // Length
        CHECK(length(v1) == doctest::Approx(5));
        CHECK(length_squared(v1) == doctest::Approx(25));
        
        // Normalization
        auto normalized = normalize(v1);
        CHECK(length(vector<float, 2>(normalized)) == doctest::Approx(1));
        
        // Component-wise operations
        auto min_v = min(v1, v2);
        auto max_v = max(v1, v2);
        CHECK(approx_equal(vector<float, 2>(min_v), vector<float, 2>{1, 0}));
        CHECK(approx_equal(vector<float, 2>(max_v), vector<float, 2>{3, 4}));
        
        // Orthonormalization
        orthonormalize(v1, v2);
        CHECK(length(v1) == doctest::Approx(1));
        CHECK(length(v2) == doctest::Approx(1));
        CHECK(dot(v1, v2) == doctest::Approx(0));
    }
    
    SUBCASE("3D vector operations") {
        vector<float, 3> v1{1, 2, 3};
        vector<float, 3> v2{4, 5, 6};
        vector<float, 3> v3{7, 8, 10};  // Changed to ensure linear independence
        
        // Cross product
        auto cross_v = cross(v1, v2);
        vector<float, 3> cross_result = cross_v;
        CHECK(dot(cross_result, v1) == doctest::Approx(0));
        CHECK(dot(cross_result, v2) == doctest::Approx(0));
        
        // Reflection
        vector<float, 3> incident{1, -1, 0};
        vector<float, 3> normal{0, 1, 0};
        auto reflected = reflect(incident, normal);
        CHECK(length(vector<float, 3>(reflected)) == doctest::Approx(length(incident)));
        
        // Projection
        auto proj = project(v1, v2);
        auto rej = reject(v1, v2);
        vector<float, 3> proj_vec = proj;
        vector<float, 3> rej_vec = rej;
        CHECK(approx_equal(v1, proj_vec + rej_vec));
        
        // Gram-Schmidt
        orthonormalize(v1, v2, v3);
        CHECK(length(v1) == doctest::Approx(1));
        CHECK(length(v2) == doctest::Approx(1));
        CHECK(length(v3) == doctest::Approx(1));
        CHECK(dot(v1, v2) == doctest::Approx(0));
        CHECK(dot(v1, v3) == doctest::Approx(0));
        CHECK(dot(v2, v3) == doctest::Approx(0));
    }
    
    SUBCASE("4D vector operations") {
        vector<float, 4> v1{1, 0, 0, 0};
        vector<float, 4> v2{1, 1, 0, 0};
        vector<float, 4> v3{1, 1, 1, 0};
        vector<float, 4> v4{1, 1, 1, 1};
        
        // Gram-Schmidt for 4D
        orthonormalize(v1, v2, v3, v4);
        CHECK(length(v1) == doctest::Approx(1));
        CHECK(length(v2) == doctest::Approx(1));
        CHECK(length(v3) == doctest::Approx(1));
        CHECK(length(v4) == doctest::Approx(1));
        
        // Build orthonormal basis
        vector<float, 4> v{1, 2, 3, 4};
        auto basis = build_orthonormal_basis(v);
        // Verify that basis was created successfully - check first vector in tuple
        CHECK(length(std::get<0>(basis)) == doctest::Approx(1));
        auto v_norm = normalize(v);
        CHECK(length(v_norm) == doctest::Approx(1));
    }
}

TEST_CASE("Comprehensive integration test - Matrix operations") {
    SUBCASE("2x2 matrix operations") {
        matrix<float, 2, 2> A{{1, 2}, {3, 4}};
        matrix<float, 2, 2> B{{5, 6}, {7, 8}};
        
        // Basic arithmetic
        auto sum = A + B;
        auto diff = A - B;
        auto scaled = 2.0f * A;
        
        CHECK(approx_equal(matrix<float, 2, 2>(sum), 
                          matrix<float, 2, 2>{{6, 8}, {10, 12}}));
        CHECK(approx_equal(matrix<float, 2, 2>(diff), 
                          matrix<float, 2, 2>{{-4, -4}, {-4, -4}}));
        CHECK(approx_equal(matrix<float, 2, 2>(scaled), 
                          matrix<float, 2, 2>{{2, 4}, {6, 8}}));
        
        // Matrix multiplication
        auto mult = A * B;
        CHECK(approx_equal(matrix<float, 2, 2>(mult), 
                          matrix<float, 2, 2>{{19, 22}, {43, 50}}));
        
        // Hadamard operations
        auto had = hadamard(A, B);
        CHECK(approx_equal(matrix<float, 2, 2>(had), 
                          matrix<float, 2, 2>{{5, 12}, {21, 32}}));
        
        // Transpose
        auto trans = transpose(A);
        CHECK(approx_equal(matrix<float, 2, 2>(trans), 
                          matrix<float, 2, 2>{{1, 3}, {2, 4}}));
        
        // Determinant
        CHECK(determinant(A) == doctest::Approx(-2));
        
        // Inverse
        auto inv = inverse(A);
        auto identity = A * inv;  // Keep as expression instead of forcing evaluation
        CHECK(approx_equal(identity, matrix<float, 2, 2>::identity()));
        
        // Trace
        CHECK(trace(A) == doctest::Approx(5));
        
        // Power
        auto A2 = pow(A, 2);
        auto A_squared = A * A;
        CHECK(approx_equal(matrix<float, 2, 2>(A2), A_squared));
    }
    
    SUBCASE("3x3 matrix operations") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        
        // Determinant
        CHECK(determinant(A) == doctest::Approx(-3));
        
        // Inverse
        auto inv = inverse(A);
        auto identity = A * inv;  // Keep as expression
        CHECK(approx_equal(identity, matrix<float, 3, 3>::identity(), 1e-4f));
        
        // Trace
        CHECK(trace(A) == doctest::Approx(16));
    }
    
    SUBCASE("4x4 matrix operations") {
        auto A = matrix<float, 4, 4>::identity();
        A(0, 0) = 2; A(1, 1) = 3; A(2, 2) = 4; A(3, 3) = 5;
        
        // Determinant
        CHECK(determinant(A) == doctest::Approx(120));
        
        // Inverse
        auto inv = inverse(A);
        auto identity = A * inv;  // Keep as expression
        CHECK(approx_equal(identity, matrix<float, 4, 4>::identity()));
        
        // Trace
        CHECK(trace(A) == doctest::Approx(14));
    }
    
    SUBCASE("Non-square matrices") {
        matrix<float, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 2> B{{1, 2}, {3, 4}, {5, 6}};
        
        // Matrix multiplication produces 2x2
        auto C = A * B;
        CHECK(approx_equal(matrix<float, 2, 2>(C), 
                          matrix<float, 2, 2>{{22, 28}, {49, 64}}));
        
        // Transpose
        auto AT = transpose(A);
        CHECK(approx_equal(matrix<float, 3, 2>(AT), 
                          matrix<float, 3, 2>{{1, 4}, {2, 5}, {3, 6}}));
    }
}

TEST_CASE("Comprehensive integration test - Matrix-Vector operations") {
    SUBCASE("2D matrix-vector multiplication") {
        matrix<float, 2, 2> A{{1, 2}, {3, 4}};
        vector<float, 2> v{1, 2};
        row_vector<float, 2> rv{1, 2};
        
        // Matrix * column vector
        auto result1 = A * v;
        CHECK(approx_equal(vector<float, 2>(result1), vector<float, 2>{5, 11}));
        
        // Row vector * matrix
        auto result2 = rv * A;
        CHECK(approx_equal(row_vector<float, 2>(result2), row_vector<float, 2>{7, 10}));
    }
    
    SUBCASE("3D matrix-vector multiplication") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v{1, 0, 1};
        
        auto result = A * v;
        CHECK(approx_equal(vector<float, 3>(result), vector<float, 3>{4, 10, 16}));
        
        // Using matrix as vector
        matrix<float, 3, 1> col_mat{{1}, {0}, {1}};
        auto result2 = A * col_mat;
        CHECK(approx_equal(matrix<float, 3, 1>(result2), matrix<float, 3, 1>{{4}, {10}, {16}}));
    }
    
    SUBCASE("Outer product") {
        vector<float, 3> u{1, 2, 3};
        vector<float, 2> v{4, 5};
        
        auto outer = outer_product(u, v);
        CHECK(outer(0, 0) == doctest::Approx(4));
        CHECK(outer(0, 1) == doctest::Approx(5));
        CHECK(outer(1, 0) == doctest::Approx(8));
        CHECK(outer(1, 1) == doctest::Approx(10));
        CHECK(outer(2, 0) == doctest::Approx(12));
        CHECK(outer(2, 1) == doctest::Approx(15));
    }
}

TEST_CASE("Comprehensive integration test - View operations") {
    SUBCASE("Matrix views") {
        matrix<float, 4, 4> M;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                M(i, j) = static_cast<float>(i * 4 + j);
            }
        }
        
        // Row view
        auto row = M.row(0);
        CHECK(row.size() == 4);
        CHECK(row[0] == 0.0f);
        CHECK(row[3] == 3.0f);
        
        // Column view
        auto col = M.col(0);
        CHECK(col.size() == 4);
        CHECK(col[0] == 0.0f);
        CHECK(col[3] == 12.0f);
        
        // Submatrix view
        auto sub = M.view();
        CHECK(sub.rows() == 4);
        CHECK(sub.cols() == 4);
        
        // Operations with views
        vector<float, 4> v{1, 1, 1, 1};
        CHECK(dot(row, v) == doctest::Approx(6));
    }
    
    SUBCASE("Views in expressions") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v{1, 1, 1};
        
        // Use row view in dot product
        auto row0 = A.row(0);
        CHECK(dot(row0, v) == doctest::Approx(6));
        
        // Use column view
        auto col0 = A.col(0);
        CHECK(dot(col0, vector<float, 3>{1, 1, 1}) == doctest::Approx(12));
    }
}

TEST_CASE("Comprehensive integration test - Expression templates") {
    SUBCASE("Lazy evaluation verification") {
        matrix<float, 3, 3> A{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        matrix<float, 3, 3> B{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}};
        vector<float, 3> v{1, 2, 3};
        
        // Create complex expression
        auto expr = 2.0f * A * v + B * v;
        
        // Verify it's an expression
        CHECK(is_matrix_expression_v<decltype(expr)>);
        
        // Force evaluation
        vector<float, 3> result = expr;
        CHECK(approx_equal(result, vector<float, 3>{4, 8, 12}));
    }
    
    SUBCASE("Chained operations") {
        matrix<float, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 4> B{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
        matrix<float, 4, 2> C{{1, 0}, {0, 1}, {1, 1}, {0, 0}};
        
        // Chain multiplication
        auto result = A * B * C;
        CHECK(expression_traits<decltype(result)>::rows == 2);
        CHECK(expression_traits<decltype(result)>::cols == 2);
        
        // Force evaluation
        matrix<float, 2, 2> result_mat = result;
        
        // Verify dimensions
        CHECK(result_mat.rows == 2);
        CHECK(result_mat.cols == 2);
    }
}

TEST_CASE("Comprehensive integration test - Mixed dimensions") {
    SUBCASE("5D and 6D vectors") {
        vector<float, 5> v1;
        vector<float, 5> v2;
        for (size_t i = 0; i < 5; ++i) {
            v1[i] = static_cast<float>(i + 1);
            v2[i] = static_cast<float>(5 - i);
        }
        
        // Basic operations still work
        auto sum = v1 + v2;
        CHECK(approx_equal(vector<float, 5>(sum), vector<float, 5>{6, 6, 6, 6, 6}));
        CHECK(dot(v1, v2) == doctest::Approx(35));
        CHECK(length_squared(v1) == doctest::Approx(55));
    }
    
    SUBCASE("Large matrices") {
        matrix<float, 5, 5> A = matrix<float, 5, 5>::identity();
        matrix<float, 5, 5> B = 2.0f * A;
        
        auto C = A + B;
        auto expected = 3.0f * matrix<float, 5, 5>::identity();
        
        // Evaluate both expressions to matrices before comparison
        matrix<float, 5, 5> C_mat(C);
        matrix<float, 5, 5> expected_mat(expected);
        CHECK(approx_equal(C_mat, expected_mat));  // Compare evaluated matrices
        
        // Note: determinant/inverse not available for 5x5
        CHECK(frobenius_norm(A) == doctest::Approx(std::sqrt(5.0f)));
    }
}

TEST_CASE("Comprehensive integration test - Edge cases") {
    SUBCASE("Zero vectors and matrices") {
        auto zero_vec = vector<float, 3>::zero();
        auto zero_mat = matrix<float, 3, 3>::zero();
        
        CHECK(length(zero_vec) == doctest::Approx(0));
        CHECK(frobenius_norm(zero_mat) == doctest::Approx(0));
        
        vector<float, 3> v{1, 2, 3};
        auto result = zero_mat * v;
        CHECK(approx_equal(vector<float, 3>(result), zero_vec));
    }
    
    SUBCASE("Identity operations") {
        auto I = matrix<float, 3, 3>::identity();
        vector<float, 3> v{1, 2, 3};
        
        auto result = I * v;
        CHECK(approx_equal(vector<float, 3>(result), v));
        
        CHECK(determinant(I) == doctest::Approx(1));
        CHECK(trace(I) == doctest::Approx(3));
        
        auto I_inv = inverse(I);
        CHECK(approx_equal(matrix<float, 3, 3>(I_inv), I));
    }
    
    SUBCASE("Small matrices") {
        matrix<float, 1, 2> A{{3, 4}};
        matrix<float, 2, 1> B{{1}, {2}};
        
        // 1x2 * 2x1 = 1x1
        auto C = A * B;
        CHECK(C(0, 0) == doctest::Approx(11));
        
        // 2x1 * 1x2 = 2x2
        auto D = B * A;
        CHECK(D(0, 0) == doctest::Approx(3));
        CHECK(D(0, 1) == doctest::Approx(4));
        CHECK(D(1, 0) == doctest::Approx(6));
        CHECK(D(1, 1) == doctest::Approx(8));
    }
}

TEST_CASE("Comprehensive integration test - Layout independence") {
    // Test both row-major and column-major layouts
    SUBCASE("Row-major operations") {
        matrix<float, 3, 3, true> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v{1, 0, 1};
        
        auto result = A * v;
        CHECK(approx_equal(vector<float, 3>(result), vector<float, 3>{4, 10, 16}));
    }
    
    SUBCASE("Column-major operations") {
        matrix<float, 3, 3, false> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v{1, 0, 1};
        
        auto result = A * v;
        CHECK(approx_equal(vector<float, 3>(result), vector<float, 3>{4, 10, 16}));
    }
    
    SUBCASE("Mixed layout operations") {
        matrix<float, 2, 2, true> A_row{{1, 2}, {3, 4}};
        matrix<float, 2, 2, false> A_col{{1, 2}, {3, 4}};
        
        auto sum = A_row + A_col;
        CHECK(approx_equal(matrix<float, 2, 2>(sum), 
                          matrix<float, 2, 2>{{2, 4}, {6, 8}}));
    }
}