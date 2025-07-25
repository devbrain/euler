#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/core/approx_equal.hh>
#include <doctest/doctest.h>

using namespace euler;

TEST_CASE("Matrix multiplication chains with compatible sizes") {
    SUBCASE("2x3 * 3x4 * 4x2 chain") {
        matrix<float, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 4> B{{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
        matrix<float, 4, 2> C{{19, 20}, {21, 22}, {23, 24}, {25, 26}};
        
        // Test left-to-right evaluation: (A*B)*C
        auto AB = A * B;
        auto ABC_lr = AB * C;
        
        // Test right-to-left evaluation: A*(B*C)
        auto BC = B * C;
        auto ABC_rl = A * BC;
        
        // Both should give the same result
        CHECK(approx_equal(ABC_lr, ABC_rl));
        
        // Test lazy evaluation - the entire chain as one expression
        auto ABC_lazy = A * B * C;
        CHECK(approx_equal(ABC_lazy, ABC_lr));
        
        // Verify the result dimensions
        CHECK(expression_traits<decltype(ABC_lazy)>::rows == 2);
        CHECK(expression_traits<decltype(ABC_lazy)>::cols == 2);
        
        // Verify the lazy evaluation result
        matrix<float, 2, 2> lazy_result = ABC_lazy;
        
        // For debugging, let's compute step by step
        matrix<float, 2, 4> AB_manual = A * B;
        matrix<float, 2, 2> ABC_manual = AB_manual * C;
        
        CHECK(approx_equal(lazy_result, ABC_manual));
    }
    
    SUBCASE("Vector-Matrix multiplication chains") {
        // Row vector * matrix * matrix * column vector
        matrix<float, 1, 3> row_vec{{1, 2, 3}};
        matrix<float, 3, 4> M1{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        matrix<float, 4, 3> M2{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}, {0, 1, 0}};
        matrix<float, 3, 1> col_vec{{1}, {2}, {3}};
        
        // Chain: row_vec * M1 * M2 * col_vec -> scalar (1x1)
        auto result = row_vec * M1 * M2 * col_vec;
        
        CHECK(expression_traits<decltype(result)>::rows == 1);
        CHECK(expression_traits<decltype(result)>::cols == 1);
        
        // Compute expected value step by step for verification
        auto temp1 = row_vec * M1;  // 1x4
        auto temp2 = temp1 * M2;     // 1x3
        auto expected = temp2 * col_vec; // 1x1
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Square matrix chain with different sizes") {
        matrix<float, 2, 2> A{{1, 2}, {3, 4}};
        matrix<float, 2, 3> B{{5, 6, 7}, {8, 9, 10}};
        matrix<float, 3, 3> C{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
        matrix<float, 3, 2> D{{2, 3}, {4, 5}, {6, 7}};
        
        // Chain: A * B * C * D -> 2x2
        auto result = A * B * C * D;
        
        CHECK(expression_traits<decltype(result)>::rows == 2);
        CHECK(expression_traits<decltype(result)>::cols == 2);
        
        // Test associativity
        auto result_lr = ((A * B) * C) * D;
        auto result_rl = A * (B * (C * D));
        auto result_mixed = (A * B) * (C * D);
        
        CHECK(approx_equal(result, result_lr));
        CHECK(approx_equal(result, result_rl));
        CHECK(approx_equal(result, result_mixed));
    }
    
    SUBCASE("Long chain with alternating dimensions") {
        matrix<float, 1, 4> A{{1, 2, 3, 4}};
        matrix<float, 4, 2> B{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        matrix<float, 2, 5> C{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
        matrix<float, 5, 3> D{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
        matrix<float, 3, 1> E{{1}, {2}, {3}};
        
        // Chain: A * B * C * D * E -> 1x1 (scalar result)
        auto result = A * B * C * D * E;
        
        CHECK(expression_traits<decltype(result)>::rows == 1);
        CHECK(expression_traits<decltype(result)>::cols == 1);
        
        // Verify it's truly lazy - check type is an expression
        using result_type = decltype(result);
        CHECK(is_matrix_expression_v<result_type>);
        CHECK(!is_matrix_v<result_type>);
    }
    
    SUBCASE("Chain with transposes") {
        matrix<float, 3, 2> A{{1, 2}, {3, 4}, {5, 6}};
        matrix<float, 3, 4> B{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        matrix<float, 4, 2> C{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        
        // Chain: A^T * B * C -> 2x2
        auto result = transpose(A) * B * C;
        
        CHECK(expression_traits<decltype(result)>::rows == 2);
        CHECK(expression_traits<decltype(result)>::cols == 2);
        
        // Test transpose with chain: (A^T * A) is always symmetric
        auto AT = transpose(A);  // 2x3
        auto ATA = AT * A;       // 2x3 * 3x2 -> 2x2
        
        CHECK(expression_traits<decltype(ATA)>::rows == 2);
        CHECK(expression_traits<decltype(ATA)>::cols == 2);
        
        // Check that A^T * A is symmetric
        matrix<float, 2, 2> ata_concrete = ATA;
        CHECK(approx_equal(ata_concrete(0, 1), ata_concrete(1, 0)));
    }
    
    SUBCASE("Chain with mixed operations") {
        matrix<float, 2, 2> A{{2, 0}, {0, 2}};
        matrix<float, 2, 3> B{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 2> C{{1, 2}, {3, 4}, {5, 6}};
        
        // Complex expression: (2*A) * B * C + A
        auto result = (2.0f * A) * B * C + A;
        
        CHECK(expression_traits<decltype(result)>::rows == 2);
        CHECK(expression_traits<decltype(result)>::cols == 2);
        
        // Manual calculation for verification
        auto temp1 = 2.0f * A * B;  // 2x3
        auto temp2 = temp1 * C;      // 2x2
        auto expected = temp2 + A;    // 2x2
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Chain with identity matrices") {
        matrix<float, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        auto I3 = matrix<float, 3, 3>::identity();
        matrix<float, 3, 2> B{{1, 2}, {3, 4}, {5, 6}};
        auto I2 = matrix<float, 2, 2>::identity();
        
        // I * A * I * B * I = A * B
        auto result = I3 * A * I3 * B * I2;
        auto expected = A * B;
        
        CHECK(approx_equal(result, expected));
    }
    
    SUBCASE("Performance test - verify lazy evaluation") {
        // Create a long chain and verify it's not evaluated until needed
        matrix<float, 10, 8> A;
        matrix<float, 8, 6> B;
        matrix<float, 6, 4> C;
        matrix<float, 4, 2> D;
        
        // Initialize with some values
        for (size_t i = 0; i < 10; ++i)
            for (size_t j = 0; j < 8; ++j)
                A(i, j) = static_cast<float>(i + j);
                
        for (size_t i = 0; i < 8; ++i)
            for (size_t j = 0; j < 6; ++j)
                B(i, j) = static_cast<float>(i * j);
                
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 4; ++j)
                C(i, j) = static_cast<float>(i - j);
                
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 2; ++j)
                D(i, j) = static_cast<float>(i + 2*j);
        
        // Create the chain
        auto chain = A * B * C * D;
        
        // Verify it's an expression, not evaluated yet
        CHECK(is_matrix_expression_v<decltype(chain)>);
        CHECK(!is_matrix_v<decltype(chain)>);
        
        // Force evaluation
        matrix<float, 10, 2> result = chain;
        
        // Verify dimensions
        CHECK(result.rows == 10);
        CHECK(result.cols == 2);
    }
    
    SUBCASE("Chain with dynamic-like behavior using fixed sizes") {
        // Test chains where intermediate results have different sizes
        // simulating what would be dynamic matrix multiplication
        
        // 1x2 * 2x5 * 5x3 * 3x1 = 1x1
        matrix<float, 1, 2> v1{{1, 2}};
        matrix<float, 2, 5> m1{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
        matrix<float, 5, 3> m2{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
        matrix<float, 3, 1> v2{{1}, {2}, {3}};
        
        auto scalar_result = v1 * m1 * m2 * v2;
        
        // The result should be a 1x1 matrix
        CHECK(expression_traits<decltype(scalar_result)>::rows == 1);
        CHECK(expression_traits<decltype(scalar_result)>::cols == 1);
        
        // Test with explicit parentheses to verify associativity
        auto grouped1 = (v1 * m1) * (m2 * v2);
        auto grouped2 = v1 * (m1 * m2) * v2;
        
        CHECK(approx_equal(scalar_result, grouped1));
        CHECK(approx_equal(scalar_result, grouped2));
    }
}