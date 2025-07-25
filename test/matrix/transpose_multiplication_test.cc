#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>

using namespace euler;

TEST_CASE("Transpose multiplication property investigation") {
    SUBCASE("Simple 2x2 case") {
        matrix<float, 2, 2> A{{1, 2}, {3, 4}};
        matrix<float, 2, 2> B{{5, 6}, {7, 8}};
        
        // Compute (A*B)^T
        auto AB = A * B;
        auto AB_T = transpose(AB);
        matrix<float, 2, 2> ab_t_concrete = AB_T;
        
        // Compute B^T * A^T
        auto BT = transpose(B);
        auto AT = transpose(A);
        auto BT_AT = BT * AT;
        matrix<float, 2, 2> bt_at_concrete = BT_AT;
        
        
        // Property: (A*B)^T = B^T * A^T
        CHECK(approx_equal(ab_t_concrete, bt_at_concrete));
    }
    
    SUBCASE("Original failing case") {
        matrix<float, 2, 3> M1{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 4> M2{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        
        // Test transpose property: (M1*M2)^T = M2^T * M1^T
        auto M1M2 = M1 * M2;
        auto M1M2_T = transpose(M1M2);
        auto M2T_M1T = transpose(M2) * transpose(M1);
        
        // Force evaluation to concrete matrices
        matrix<float, 4, 2> m1m2_t_concrete = M1M2_T;
        matrix<float, 4, 2> m2t_m1t_concrete = M2T_M1T;
        
        CHECK(approx_equal(m1m2_t_concrete, m2t_m1t_concrete));
    }
    
    SUBCASE("Test with expression chain") {
        matrix<float, 2, 3> M1{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 4> M2{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        
        // Test if the issue is with expression evaluation
        auto M1M2_expr = M1 * M2;  // This is an expression
        auto M1M2_T_expr = transpose(M1M2_expr);  // Transpose of expression
        
        auto M2T_expr = transpose(M2);
        auto M1T_expr = transpose(M1);
        auto M2T_M1T_expr = M2T_expr * M1T_expr;
        
        // Force evaluation
        matrix<float, 4, 2> result1 = M1M2_T_expr;
        matrix<float, 4, 2> result2 = M2T_M1T_expr;
        
        CHECK(approx_equal(result1, result2));
    }
}