#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/vector/vector.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>

using namespace euler;

TEST_CASE("Vector-Matrix operations with lazy evaluation") {
    SUBCASE("Matrix * Vector (column vector result)") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v{1, 2, 3};
        
        // Test lazy evaluation
        auto expr = A * v;
        
        // Verify it's an expression, not evaluated yet
        CHECK(is_matrix_expression_v<decltype(expr)>);
        CHECK(!is_matrix_v<decltype(expr)>);
        
        // Force evaluation
        vector<float, 3> result = expr;
        
        // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3] = [14, 32, 50]
        CHECK(result[0] == doctest::Approx(14));
        CHECK(result[1] == doctest::Approx(32));
        CHECK(result[2] == doctest::Approx(50));
        
        // Test with matrix expression
        auto expr2 = (2.0f * A) * v;
        vector<float, 3> result2 = expr2;
        CHECK(result2[0] == doctest::Approx(28));
        CHECK(result2[1] == doctest::Approx(64));
        CHECK(result2[2] == doctest::Approx(100));
    }
    
    SUBCASE("Row Vector * Matrix (row vector result)") {
        row_vector<float, 3> v{1, 2, 3};
        matrix<float, 3, 3> A{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        
        // Test lazy evaluation
        auto expr = v * A;
        
        // Verify it's an expression
        CHECK(is_matrix_expression_v<decltype(expr)>);
        CHECK(!is_matrix_v<decltype(expr)>);
        
        // Force evaluation
        row_vector<float, 3> result = expr;
        
        // Expected: [1*1+2*2+3*3, 1*4+2*5+3*6, 1*7+2*8+3*9] = [14, 32, 50]
        CHECK(result[0] == doctest::Approx(14));
        CHECK(result[1] == doctest::Approx(32));
        CHECK(result[2] == doctest::Approx(50));
    }
    
    SUBCASE("Chained operations") {
        matrix<float, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 2> B{{1, 2}, {3, 4}, {5, 6}};
        vector<float, 2> v{1, 2};
        
        // Chain: A * B * v
        auto expr = A * B * v;
        
        // Verify it's still an expression
        CHECK(is_matrix_expression_v<decltype(expr)>);
        
        // Force evaluation
        vector<float, 2> result = expr;
        
        // Compute expected value step by step
        // A * B = [[22, 28], [49, 64]]
        // [[22, 28], [49, 64]] * [1, 2] = [22+56, 49+128] = [78, 177]
        CHECK(result[0] == doctest::Approx(78));
        CHECK(result[1] == doctest::Approx(177));
    }
    
    SUBCASE("Mixed matrix-vector expressions") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v1{1, 0, 0};
        vector<float, 3> v2{0, 1, 0};
        
        // (A * v1) + (A * v2)
        auto expr = A * v1 + A * v2;
        
        // Force evaluation
        vector<float, 3> result = expr;
        
        // A * v1 = [1, 4, 7], A * v2 = [2, 5, 8]
        // Sum = [3, 9, 15]
        CHECK(result[0] == doctest::Approx(3));
        CHECK(result[1] == doctest::Approx(9));
        CHECK(result[2] == doctest::Approx(15));
    }
    
    SUBCASE("Vector expression as input") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<float, 3> v1{1, 2, 3};
        vector<float, 3> v2{4, 5, 6};
        
        // A * (v1 + v2)
        auto vec_expr = v1 + v2;  // Vector expression
        auto result_expr = A * vec_expr;
        
        // Force evaluation
        vector<float, 3> result = result_expr;
        
        // v1 + v2 = [5, 7, 9]
        // A * [5, 7, 9] = [1*5+2*7+3*9, 4*5+5*7+6*9, 7*5+8*7+9*9] = [46, 109, 172]
        CHECK(result[0] == doctest::Approx(46));
        CHECK(result[1] == doctest::Approx(109));
        CHECK(result[2] == doctest::Approx(172));
    }
    
    SUBCASE("Matrix expression as input") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        matrix<float, 3, 3> B{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}};
        vector<float, 3> v{1, 2, 3};
        
        // (A + B) * v
        auto mat_expr = A + B;  // Matrix expression
        auto result_expr = mat_expr * v;
        
        // Force evaluation
        vector<float, 3> result = result_expr;
        
        // A + B = [[3, 2, 3], [4, 7, 6], [7, 8, 11]]
        // Result = [3*1+2*2+3*3, 4*1+7*2+6*3, 7*1+8*2+11*3] = [16, 36, 56]
        CHECK(result[0] == doctest::Approx(16));
        CHECK(result[1] == doctest::Approx(36));
        CHECK(result[2] == doctest::Approx(56));
    }
    
    SUBCASE("Using 1xN and Nx1 matrices as vectors") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        matrix<float, 3, 1> col_vec{{1}, {2}, {3}};
        matrix<float, 1, 3> row_vec{{1, 2, 3}};
        
        // Matrix * column matrix
        auto expr1 = A * col_vec;
        matrix<float, 3, 1> result1 = expr1;
        CHECK(result1(0, 0) == doctest::Approx(14));
        CHECK(result1(1, 0) == doctest::Approx(32));
        CHECK(result1(2, 0) == doctest::Approx(50));
        
        // Row matrix * matrix
        auto expr2 = row_vec * A;
        matrix<float, 1, 3> result2 = expr2;
        CHECK(result2(0, 0) == doctest::Approx(30));
        CHECK(result2(0, 1) == doctest::Approx(36));
        CHECK(result2(0, 2) == doctest::Approx(42));
    }
}