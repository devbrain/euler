#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/core/traits.hh>
#include <doctest/doctest.h>

using namespace euler;

TEST_CASE("Matrix Expression Templates") {
    SUBCASE("Basic arithmetic expressions") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        matrix<float, 2, 2> c{{9, 10}, {11, 12}};
        
        // This should create an expression, not compute immediately
        auto expr = a + b - c;
        
        // Verify it's an expression type, not a matrix
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        // Force evaluation
        matrix<float, 2, 2> result = expr;
        
        CHECK(result(0,0) == doctest::Approx(-3));  // 1+5-9
        CHECK(result(0,1) == doctest::Approx(-2));  // 2+6-10
        CHECK(result(1,0) == doctest::Approx(-1));  // 3+7-11
        CHECK(result(1,1) == doctest::Approx(0));   // 4+8-12
    }
    
    SUBCASE("Scalar operations expressions") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        
        auto expr = 2.0f * a + a / 2.0f;
        
        // Verify it's an expression
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        CHECK(result(0,0) == doctest::Approx(2.5f));  // 2*1 + 1/2
        CHECK(result(0,1) == doctest::Approx(5.0f));  // 2*2 + 2/2
        CHECK(result(1,0) == doctest::Approx(7.5f));  // 2*3 + 3/2
        CHECK(result(1,1) == doctest::Approx(10.0f)); // 2*4 + 4/2
    }
    
    SUBCASE("Matrix multiplication expressions") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        matrix<float, 2, 2> c{{1, 0}, {0, 1}};  // identity
        
        // This should create a multiplication expression
        auto expr = a * b * c;
        
        // Verify it's an expression
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        // a * b = {{19, 22}, {43, 50}}
        // (a * b) * I = {{19, 22}, {43, 50}}
        CHECK(result(0,0) == doctest::Approx(19));
        CHECK(result(0,1) == doctest::Approx(22));
        CHECK(result(1,0) == doctest::Approx(43));
        CHECK(result(1,1) == doctest::Approx(50));
    }
    
    SUBCASE("Complex expressions with multiplication") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        matrix<float, 2, 2> c{{2, 0}, {0, 2}};
        
        // Complex expression: A * B + C * 3
        auto expr = a * b + c * 3.0f;
        
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        // a * b = {{19, 22}, {43, 50}}
        // c * 3 = {{6, 0}, {0, 6}}
        // sum = {{25, 22}, {43, 56}}
        CHECK(result(0,0) == doctest::Approx(25));
        CHECK(result(0,1) == doctest::Approx(22));
        CHECK(result(1,0) == doctest::Approx(43));
        CHECK(result(1,1) == doctest::Approx(56));
    }
    
    SUBCASE("Transpose expressions") {
        matrix<float, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        
        // Transpose of a concrete matrix returns a concrete matrix
        auto direct = transpose(a);
        static_assert(std::is_same_v<decltype(direct), matrix<float, 3, 2>>);
        
        // Transpose of an expression returns an expression
        auto add_expr = a + a;
        static_assert(is_matrix_expression_v<decltype(add_expr)>, "a + a should be a matrix expression");
        static_assert(is_expression_v<decltype(add_expr)>, "a + a should be an expression");
        auto expr = transpose(add_expr);
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 3, 2>>);
        
        matrix<float, 3, 2> result = expr;
        
        CHECK(result(0,0) == doctest::Approx(2));  // 2*1
        CHECK(result(0,1) == doctest::Approx(8));  // 2*4
        CHECK(result(1,0) == doctest::Approx(4));  // 2*2
        CHECK(result(1,1) == doctest::Approx(10)); // 2*5
        CHECK(result(2,0) == doctest::Approx(6));  // 2*3
        CHECK(result(2,1) == doctest::Approx(12)); // 2*6
    }
    
    SUBCASE("Expression with transpose and multiplication") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        
        // A^T * B - transpose returns a matrix, but multiplication creates expression
        auto a_t = transpose(a);  // This is a concrete matrix
        auto expr = a_t * b;      // This creates an expression
        
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        // a^T = {{1, 3}, {2, 4}}
        // a^T * b = {{26, 30}, {38, 44}}
        CHECK(result(0,0) == doctest::Approx(26));  // 1*5 + 3*7
        CHECK(result(0,1) == doctest::Approx(30));  // 1*6 + 3*8
        CHECK(result(1,0) == doctest::Approx(38));  // 2*5 + 4*7
        CHECK(result(1,1) == doctest::Approx(44));  // 2*6 + 4*8
    }
    
    SUBCASE("Mixed concrete and expression multiplication") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        matrix<float, 2, 2> c{{1, 1}, {1, 1}};
        
        // (A + B) * C - should create expression
        auto expr1 = (a + b) * c;
        
        static_assert(!std::is_same_v<decltype(expr1), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result1 = expr1;
        
        // a + b = {{6, 8}, {10, 12}}
        // (a + b) * c = {{14, 14}, {22, 22}}
        CHECK(result1(0,0) == doctest::Approx(14));
        CHECK(result1(0,1) == doctest::Approx(14));
        CHECK(result1(1,0) == doctest::Approx(22));
        CHECK(result1(1,1) == doctest::Approx(22));
        
        // A * (B + C) - should also create expression
        auto expr2 = a * (b + c);
        
        static_assert(!std::is_same_v<decltype(expr2), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result2 = expr2;
        
        // b + c = {{6, 7}, {8, 9}}
        // a * (b + c) = {{22, 25}, {50, 57}}
        CHECK(result2(0,0) == doctest::Approx(22));  // 1*6 + 2*8
        CHECK(result2(0,1) == doctest::Approx(25));  // 1*7 + 2*9
        CHECK(result2(1,0) == doctest::Approx(50));  // 3*6 + 4*8
        CHECK(result2(1,1) == doctest::Approx(57));  // 3*7 + 4*9
    }
    
    SUBCASE("Negation expression") {
        matrix<float, 2, 2> a{{1, -2}, {3, -4}};
        
        auto expr = -a;
        
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        CHECK(result(0,0) == doctest::Approx(-1));
        CHECK(result(0,1) == doctest::Approx(2));
        CHECK(result(1,0) == doctest::Approx(-3));
        CHECK(result(1,1) == doctest::Approx(4));
    }
    
    SUBCASE("Matrix multiplication always creates expressions") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        
        // All operations create expressions for lazy evaluation
        auto expr = a * b;
        
        // This should be an expression, not a concrete matrix
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        // Force evaluation
        matrix<float, 2, 2> result = expr;
        
        CHECK(result(0,0) == doctest::Approx(19));
        CHECK(result(0,1) == doctest::Approx(22));
        CHECK(result(1,0) == doctest::Approx(43));
        CHECK(result(1,1) == doctest::Approx(50));
    }
    
    SUBCASE("Hadamard operations create expressions") {
        matrix<float, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 2, 3> b{{2, 4, 6}, {8, 10, 12}};
        
        // Hadamard multiplication creates expression
        auto expr_mul = hadamard(a, b);
        static_assert(!std::is_same_v<decltype(expr_mul), matrix<float, 2, 3>>);
        
        matrix<float, 2, 3> result_mul = expr_mul;
        CHECK(result_mul(0,0) == doctest::Approx(2));   // 1*2
        CHECK(result_mul(0,1) == doctest::Approx(8));   // 2*4
        CHECK(result_mul(0,2) == doctest::Approx(18));  // 3*6
        CHECK(result_mul(1,0) == doctest::Approx(32));  // 4*8
        CHECK(result_mul(1,1) == doctest::Approx(50));  // 5*10
        CHECK(result_mul(1,2) == doctest::Approx(72));  // 6*12
        
        // Hadamard division creates expression
        auto expr_div = hadamard_div(b, a);
        static_assert(!std::is_same_v<decltype(expr_div), matrix<float, 2, 3>>);
        
        matrix<float, 2, 3> result_div = expr_div;
        CHECK(result_div(0,0) == doctest::Approx(2));   // 2/1
        CHECK(result_div(0,1) == doctest::Approx(2));   // 4/2
        CHECK(result_div(0,2) == doctest::Approx(2));   // 6/3
        CHECK(result_div(1,0) == doctest::Approx(2));   // 8/4
        CHECK(result_div(1,1) == doctest::Approx(2));   // 10/5
        CHECK(result_div(1,2) == doctest::Approx(2));   // 12/6
    }
    
    SUBCASE("Complex Hadamard expressions") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{2, 3}, {4, 5}};
        matrix<float, 2, 2> c{{1, 1}, {1, 1}};
        
        // Complex expression: hadamard(a + b, c) / 2
        auto expr = hadamard(a + b, c) / 2.0f;
        
        static_assert(!std::is_same_v<decltype(expr), matrix<float, 2, 2>>);
        
        matrix<float, 2, 2> result = expr;
        
        // (a + b) = {{3, 5}, {7, 9}}
        // hadamard with c = {{3, 5}, {7, 9}}
        // divided by 2 = {{1.5, 2.5}, {3.5, 4.5}}
        CHECK(result(0,0) == doctest::Approx(1.5f));
        CHECK(result(0,1) == doctest::Approx(2.5f));
        CHECK(result(1,0) == doctest::Approx(3.5f));
        CHECK(result(1,1) == doctest::Approx(4.5f));
    }
    
    SUBCASE("Trace works with expressions") {
        matrix<float, 3, 3> a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        matrix<float, 3, 3> b{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        
        // Trace of concrete matrix
        auto tr_a = trace(a);
        CHECK(tr_a == doctest::Approx(15));  // 1 + 5 + 9
        
        // Trace of expression
        auto tr_expr = trace(a + b);
        CHECK(tr_expr == doctest::Approx(21));  // (1+1) + (5+2) + (9+3) = 2 + 7 + 12
        
        // Trace of complex expression
        auto tr_complex = trace(hadamard(a, b) * 2.0f);
        CHECK(tr_complex == doctest::Approx(76));  // (1*1 + 5*2 + 9*3) * 2 = (1 + 10 + 27) * 2 = 38 * 2 = 76
    }
    
    SUBCASE("Power works with expressions") {
        matrix<float, 2, 2> a{{2, 1}, {0, 2}};
        matrix<float, 2, 2> identity{{1, 0}, {0, 1}};
        
        // Power of concrete matrix
        auto a2 = pow(a, 2);
        CHECK(a2(0,0) == doctest::Approx(4));   // 2*2 + 1*0
        CHECK(a2(0,1) == doctest::Approx(4));   // 2*1 + 1*2
        CHECK(a2(1,0) == doctest::Approx(0));   // 0*2 + 2*0
        CHECK(a2(1,1) == doctest::Approx(4));   // 0*1 + 2*2
        
        // Power of expression (a - identity)
        // (a - I) = {{1, 1}, {0, 1}}
        auto expr_pow = pow(a - identity, 3);
        // (a - I)^2 = {{1, 2}, {0, 1}}
        // (a - I)^3 = {{1, 3}, {0, 1}}
        CHECK(expr_pow(0,0) == doctest::Approx(1));
        CHECK(expr_pow(0,1) == doctest::Approx(3));
        CHECK(expr_pow(1,0) == doctest::Approx(0));
        CHECK(expr_pow(1,1) == doctest::Approx(1));
        
        // Power of 0 returns identity
        auto expr_pow0 = pow(a + identity, 0);
        CHECK(expr_pow0(0,0) == doctest::Approx(1));
        CHECK(expr_pow0(0,1) == doctest::Approx(0));
        CHECK(expr_pow0(1,0) == doctest::Approx(0));
        CHECK(expr_pow0(1,1) == doctest::Approx(1));
    }
    
    SUBCASE("Inverse creates lazy expression") {
        matrix<float, 2, 2> a{{4, 3}, {2, 1}};
        
        // Inverse of concrete matrix creates expression
        auto inv_expr = inverse(a);
        static_assert(!std::is_same_v<decltype(inv_expr), matrix<float, 2, 2>>);
        
        // Check values (det = 4*1 - 3*2 = -2)
        CHECK(inv_expr(0,0) == doctest::Approx(-0.5f));   // 1/-2
        CHECK(inv_expr(0,1) == doctest::Approx(1.5f));    // -3/-2
        CHECK(inv_expr(1,0) == doctest::Approx(1.0f));    // -2/-2
        CHECK(inv_expr(1,1) == doctest::Approx(-2.0f));   // 4/-2
        
        // Inverse of expression
        auto expr_inv = inverse(a * 2.0f);
        static_assert(!std::is_same_v<decltype(expr_inv), matrix<float, 2, 2>>);
        
        // Check values (det = 8*2 - 6*4 = -8)
        CHECK(expr_inv(0,0) == doctest::Approx(-0.25f));  // 2/-8
        CHECK(expr_inv(0,1) == doctest::Approx(0.75f));   // -6/-8
        CHECK(expr_inv(1,0) == doctest::Approx(0.5f));    // -4/-8
        CHECK(expr_inv(1,1) == doctest::Approx(-1.0f));   // 8/-8
        
        // Complex expression with inverse
        auto identity = matrix<float, 2, 2>::identity();
        auto complex_expr = inverse(a + identity) * a;
        matrix<float, 2, 2> result = complex_expr;
        
        // (a + I) = {{5, 3}, {2, 2}}, det = 10 - 6 = 4
        // inv(a + I) = {{0.5, -0.75}, {-0.5, 1.25}}
        // inv(a + I) * a = {{0.5, 0.75}, {0.5, -0.25}}
        CHECK(result(0,0) == doctest::Approx(0.5f));
        CHECK(result(0,1) == doctest::Approx(0.75f));
        CHECK(result(1,0) == doctest::Approx(0.5f));
        CHECK(result(1,1) == doctest::Approx(-0.25f));
    }
}