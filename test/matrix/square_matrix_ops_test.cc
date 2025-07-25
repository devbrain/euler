#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;

TEST_CASE("Comprehensive Square Matrix Operations") {
    SUBCASE("2x2 Matrix Operations") {
        matrix<float, 2, 2> a{{4, 3}, {2, 1}};
        matrix<float, 2, 2> b{{2, 1}, {3, 4}};
        matrix<float, 2, 2> identity = matrix<float, 2, 2>::identity();
        
        SUBCASE("Basic Operations") {
            // Addition
            auto sum = a + b;
            CHECK(sum(0,0) == doctest::Approx(6));
            CHECK(sum(0,1) == doctest::Approx(4));
            CHECK(sum(1,0) == doctest::Approx(5));
            CHECK(sum(1,1) == doctest::Approx(5));
            
            // Subtraction
            auto diff = a - b;
            CHECK(diff(0,0) == doctest::Approx(2));
            CHECK(diff(0,1) == doctest::Approx(2));
            CHECK(diff(1,0) == doctest::Approx(-1));
            CHECK(diff(1,1) == doctest::Approx(-3));
            
            // Scalar multiplication
            auto scaled = a * 2.0f;
            CHECK(scaled(0,0) == doctest::Approx(8));
            CHECK(scaled(0,1) == doctest::Approx(6));
            CHECK(scaled(1,0) == doctest::Approx(4));
            CHECK(scaled(1,1) == doctest::Approx(2));
            
            // Scalar division
            auto divided = a / 2.0f;
            CHECK(divided(0,0) == doctest::Approx(2));
            CHECK(divided(0,1) == doctest::Approx(1.5));
            CHECK(divided(1,0) == doctest::Approx(1));
            CHECK(divided(1,1) == doctest::Approx(0.5));
            
            // Negation
            auto neg = -a;
            CHECK(neg(0,0) == doctest::Approx(-4));
            CHECK(neg(0,1) == doctest::Approx(-3));
            CHECK(neg(1,0) == doctest::Approx(-2));
            CHECK(neg(1,1) == doctest::Approx(-1));
        }
        
        SUBCASE("Matrix Multiplication") {
            auto prod = a * b;
            CHECK(prod(0,0) == doctest::Approx(17));  // 4*2 + 3*3
            CHECK(prod(0,1) == doctest::Approx(16));  // 4*1 + 3*4
            CHECK(prod(1,0) == doctest::Approx(7));   // 2*2 + 1*3
            CHECK(prod(1,1) == doctest::Approx(6));   // 2*1 + 1*4
            
            // Identity property
            auto a_times_i = a * identity;
            CHECK(approx_equal(a_times_i, a));
            
            auto i_times_a = identity * a;
            CHECK(approx_equal(i_times_a, a));
        }
        
        SUBCASE("Transpose") {
            auto at = transpose(a);
            CHECK(at(0,0) == doctest::Approx(4));
            CHECK(at(0,1) == doctest::Approx(2));
            CHECK(at(1,0) == doctest::Approx(3));
            CHECK(at(1,1) == doctest::Approx(1));
            
            // Transpose of transpose is original
            auto att = transpose(at);
            CHECK(approx_equal(att, a));
            
            // Transpose of expression
            auto expr_t = transpose(a + b);
            CHECK(expr_t(0,0) == doctest::Approx(6));   // (a+b)(0,0)
            CHECK(expr_t(0,1) == doctest::Approx(5));   // (a+b)(1,0)
            CHECK(expr_t(1,0) == doctest::Approx(4));   // (a+b)(0,1)
            CHECK(expr_t(1,1) == doctest::Approx(5));   // (a+b)(1,1)
        }
        
        SUBCASE("Determinant") {
            auto det_a = determinant(a);
            CHECK(det_a == doctest::Approx(-2));  // 4*1 - 3*2
            
            auto det_b = determinant(b);
            CHECK(det_b == doctest::Approx(5));   // 2*4 - 1*3
            
            auto det_identity = determinant(identity);
            CHECK(det_identity == doctest::Approx(1));
            
            // Determinant of expression
            auto det_expr = determinant(a * 2.0f);
            CHECK(det_expr == doctest::Approx(-8));  // det(2A) = 2^2 * det(A)
        }
        
        SUBCASE("Inverse") {
            auto inv_a = inverse(a);
            CHECK(inv_a(0,0) == doctest::Approx(-0.5));   // 1/-2
            CHECK(inv_a(0,1) == doctest::Approx(1.5));    // -3/-2
            CHECK(inv_a(1,0) == doctest::Approx(1));      // -2/-2
            CHECK(inv_a(1,1) == doctest::Approx(-2));     // 4/-2
            
            // A * A^(-1) = I
            auto a_inv_a = a * inv_a;
            CHECK(approx_equal(a_inv_a, identity, 1e-4f));
            
            // A^(-1) * A = I
            auto inv_a_a = inv_a * a;
            CHECK(approx_equal(inv_a_a, identity, 1e-4f));
            
            // Inverse of expression
            auto inv_expr = inverse(a + identity);
            // (A + I)^(-1) where A + I = {{5, 3}, {2, 2}}
            // det = 10 - 6 = 4
            CHECK(inv_expr(0,0) == doctest::Approx(0.5));    // 2/4
            CHECK(inv_expr(0,1) == doctest::Approx(-0.75));  // -3/4
            CHECK(inv_expr(1,0) == doctest::Approx(-0.5));   // -2/4
            CHECK(inv_expr(1,1) == doctest::Approx(1.25));   // 5/4
        }
        
        SUBCASE("Trace") {
            auto tr_a = trace(a);
            CHECK(tr_a == doctest::Approx(5));  // 4 + 1
            
            auto tr_b = trace(b);
            CHECK(tr_b == doctest::Approx(6));  // 2 + 4
            
            auto tr_identity = trace(identity);
            CHECK(tr_identity == doctest::Approx(2));
            
            // Trace of expression
            auto tr_expr = trace(a + b);
            CHECK(tr_expr == doctest::Approx(11));  // trace(A) + trace(B)
        }
        
        SUBCASE("Power") {
            auto a2 = pow(a, 2);
            CHECK(a2(0,0) == doctest::Approx(22));  // 4*4 + 3*2
            CHECK(a2(0,1) == doctest::Approx(15));  // 4*3 + 3*1
            CHECK(a2(1,0) == doctest::Approx(10));  // 2*4 + 1*2
            CHECK(a2(1,1) == doctest::Approx(7));   // 2*3 + 1*1
            
            auto a0 = pow(a, 0);
            CHECK(approx_equal(a0, identity));
            
            auto a1 = pow(a, 1);
            CHECK(approx_equal(a1, a));
            
            // Power of expression
            auto expr_pow = pow(identity * 2.0f, 3);
            CHECK(expr_pow(0,0) == doctest::Approx(8));  // 2^3
            CHECK(expr_pow(0,1) == doctest::Approx(0));
            CHECK(expr_pow(1,0) == doctest::Approx(0));
            CHECK(expr_pow(1,1) == doctest::Approx(8));
        }
        
        SUBCASE("Hadamard Operations") {
            auto had_mul = hadamard(a, b);
            CHECK(had_mul(0,0) == doctest::Approx(8));   // 4*2
            CHECK(had_mul(0,1) == doctest::Approx(3));   // 3*1
            CHECK(had_mul(1,0) == doctest::Approx(6));   // 2*3
            CHECK(had_mul(1,1) == doctest::Approx(4));   // 1*4
            
            auto had_div = hadamard_div(a, b);
            CHECK(had_div(0,0) == doctest::Approx(2));      // 4/2
            CHECK(had_div(0,1) == doctest::Approx(3));      // 3/1
            CHECK(had_div(1,0) == doctest::Approx(2.0/3.0)); // 2/3
            CHECK(had_div(1,1) == doctest::Approx(0.25));   // 1/4
            
            // Hadamard with expressions
            auto had_expr = hadamard(a + identity, b);
            CHECK(had_expr(0,0) == doctest::Approx(10));  // 5*2
            CHECK(had_expr(0,1) == doctest::Approx(3));   // 3*1
            CHECK(had_expr(1,0) == doctest::Approx(6));   // 2*3
            CHECK(had_expr(1,1) == doctest::Approx(8));   // 2*4
        }
        
        SUBCASE("Complex Expression Chains") {
            // (A * B + C) * D^T - E / 2
            matrix<float, 2, 2> c{{1, 0}, {0, 1}};  // identity
            matrix<float, 2, 2> d{{1, 2}, {3, 4}};
            matrix<float, 2, 2> e{{2, 2}, {2, 2}};
            
            auto expr = (a * b + c) * transpose(d) - e / 2.0f;
            matrix<float, 2, 2> result = expr;
            
            // Compute expected result step by step
            auto ab = a * b;
            auto ab_plus_c = ab + c;
            auto d_t = transpose(d);
            auto product = ab_plus_c * d_t;
            auto e_div_2 = e / 2.0f;
            auto expected = product - e_div_2;
            
            CHECK(approx_equal(result, expected));
        }
    }
    
    SUBCASE("3x3 Matrix Operations") {
        matrix<float, 3, 3> a{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        matrix<float, 3, 3> b{{2, 0, 1}, {1, 2, 1}, {0, 1, 2}};
        matrix<float, 3, 3> identity = matrix<float, 3, 3>::identity();
        
        SUBCASE("Basic Operations") {
            // Addition
            auto sum = a + b;
            CHECK(sum(0,0) == doctest::Approx(3));
            CHECK(sum(1,1) == doctest::Approx(7));
            CHECK(sum(2,2) == doctest::Approx(12));
            
            // Scalar operations
            auto scaled = a * 3.0f;
            CHECK(scaled(0,0) == doctest::Approx(3));
            CHECK(scaled(1,1) == doctest::Approx(15));
            CHECK(scaled(2,2) == doctest::Approx(30));
            
            // Negation
            auto neg = -a;
            CHECK(neg(0,0) == doctest::Approx(-1));
            CHECK(neg(1,1) == doctest::Approx(-5));
            CHECK(neg(2,2) == doctest::Approx(-10));
        }
        
        SUBCASE("Matrix Multiplication") {
            auto prod = a * b;
            CHECK(prod(0,0) == doctest::Approx(4));   // 1*2 + 2*1 + 3*0
            CHECK(prod(0,1) == doctest::Approx(7));   // 1*0 + 2*2 + 3*1
            CHECK(prod(0,2) == doctest::Approx(9));   // 1*1 + 2*1 + 3*2
            CHECK(prod(1,0) == doctest::Approx(13));  // 4*2 + 5*1 + 6*0
            CHECK(prod(1,1) == doctest::Approx(16));  // 4*0 + 5*2 + 6*1
            CHECK(prod(1,2) == doctest::Approx(21));  // 4*1 + 5*1 + 6*2
            CHECK(prod(2,0) == doctest::Approx(22));  // 7*2 + 8*1 + 10*0
            CHECK(prod(2,1) == doctest::Approx(26));  // 7*0 + 8*2 + 10*1
            CHECK(prod(2,2) == doctest::Approx(35));  // 7*1 + 8*1 + 10*2
            
            // Identity property
            auto a_times_i = a * identity;
            CHECK(approx_equal(a_times_i, a));
        }
        
        SUBCASE("Transpose") {
            auto at = transpose(a);
            CHECK(at(0,0) == doctest::Approx(1));
            CHECK(at(0,1) == doctest::Approx(4));
            CHECK(at(0,2) == doctest::Approx(7));
            CHECK(at(1,0) == doctest::Approx(2));
            CHECK(at(1,1) == doctest::Approx(5));
            CHECK(at(1,2) == doctest::Approx(8));
            CHECK(at(2,0) == doctest::Approx(3));
            CHECK(at(2,1) == doctest::Approx(6));
            CHECK(at(2,2) == doctest::Approx(10));
            
            // Transpose of expression
            auto expr_t = transpose(a * 2.0f);
            CHECK(expr_t(0,1) == doctest::Approx(8));   // 2 * a(1,0)
            CHECK(expr_t(1,2) == doctest::Approx(16));  // 2 * a(2,1)
        }
        
        SUBCASE("Determinant") {
            auto det_a = determinant(a);
            CHECK(det_a == doctest::Approx(-3));  // Computed using rule of Sarrus
            
            auto det_b = determinant(b);
            CHECK(det_b == doctest::Approx(7));
            
            auto det_identity = determinant(identity);
            CHECK(det_identity == doctest::Approx(1));
            
            // Determinant of scalar multiplication
            auto det_scaled = determinant(a * 2.0f);
            CHECK(det_scaled == doctest::Approx(-24));  // 2^3 * det(A) = 8 * -3
        }
        
        SUBCASE("Inverse") {
            auto inv_a = inverse(a);
            
            // Verify A * A^(-1) = I
            auto a_inv_a = a * inv_a;
            CHECK(approx_equal(a_inv_a, identity, 1e-4f));
            
            // Verify A^(-1) * A = I
            auto inv_a_a = inv_a * a;
            CHECK(approx_equal(inv_a_a, identity, 1e-4f));
            
            // Test specific values
            CHECK(inv_a(0,0) == doctest::Approx(-2.0/3.0));
            CHECK(inv_a(0,1) == doctest::Approx(-4.0/3.0));
            CHECK(inv_a(0,2) == doctest::Approx(1));
            
            // Inverse of expression
            auto inv_expr = inverse(b * 2.0f);
            auto check_expr = (b * 2.0f) * inv_expr;
            CHECK(approx_equal(check_expr, identity, 1e-4f));
        }
        
        SUBCASE("Trace") {
            auto tr_a = trace(a);
            CHECK(tr_a == doctest::Approx(16));  // 1 + 5 + 10
            
            auto tr_b = trace(b);
            CHECK(tr_b == doctest::Approx(6));   // 2 + 2 + 2
            
            // Trace is linear
            auto tr_sum = trace(a + b);
            CHECK(tr_sum == doctest::Approx(22));  // trace(A) + trace(B)
            
            // Trace of product (generally trace(AB) != trace(A)*trace(B))
            auto tr_prod = trace(a * b);
            CHECK(tr_prod == doctest::Approx(55));  // 4 + 16 + 35
        }
        
        SUBCASE("Power") {
            matrix<float, 3, 3> simple{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};  // diagonal
            
            auto simple2 = pow(simple, 2);
            CHECK(simple2(0,0) == doctest::Approx(4));   // 2^2
            CHECK(simple2(1,1) == doctest::Approx(9));   // 3^2
            CHECK(simple2(2,2) == doctest::Approx(16));  // 4^2
            
            auto simple3 = pow(simple, 3);
            CHECK(simple3(0,0) == doctest::Approx(8));   // 2^3
            CHECK(simple3(1,1) == doctest::Approx(27));  // 3^3
            CHECK(simple3(2,2) == doctest::Approx(64));  // 4^3
            
            // Power 0 is identity
            auto a0 = pow(a, 0);
            CHECK(approx_equal(a0, identity));
            
            // Power of expression
            auto expr_pow = pow(simple / 2.0f, 2);
            CHECK(expr_pow(0,0) == doctest::Approx(1));     // (2/2)^2
            CHECK(expr_pow(1,1) == doctest::Approx(2.25));  // (3/2)^2
            CHECK(expr_pow(2,2) == doctest::Approx(4));     // (4/2)^2
        }
        
        SUBCASE("Hadamard Operations") {
            auto had_mul = hadamard(a, b);
            CHECK(had_mul(0,0) == doctest::Approx(2));   // 1*2
            CHECK(had_mul(1,1) == doctest::Approx(10));  // 5*2
            CHECK(had_mul(2,2) == doctest::Approx(20));  // 10*2
            
            matrix<float, 3, 3> c{{2, 4, 6}, {8, 10, 12}, {14, 16, 20}};
            auto had_div = hadamard_div(c, a);
            CHECK(had_div(0,0) == doctest::Approx(2));   // 2/1
            CHECK(had_div(1,1) == doctest::Approx(2));   // 10/5
            CHECK(had_div(2,2) == doctest::Approx(2));   // 20/10
            
            // Hadamard with expressions
            auto had_expr = hadamard(a + identity, b - identity);
            matrix<float, 3, 3> expected_lhs = a + identity;
            matrix<float, 3, 3> expected_rhs = b - identity;
            matrix<float, 3, 3> expected = hadamard(expected_lhs, expected_rhs);
            matrix<float, 3, 3> result = had_expr;
            CHECK(approx_equal(result, expected));
        }
    }
    
    SUBCASE("4x4 Matrix Operations") {
        matrix<float, 4, 4> a{
            {4, 1, 0, 0},
            {1, 4, 1, 0},
            {0, 1, 4, 1},
            {0, 0, 1, 4}
        };
        matrix<float, 4, 4> b{
            {1, 0, 0, 1},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {1, 0, 0, 1}
        };
        matrix<float, 4, 4> identity = matrix<float, 4, 4>::identity();
        
        SUBCASE("Basic Operations") {
            // Addition
            auto sum = a + b;
            CHECK(sum(0,0) == doctest::Approx(5));   // 4+1
            CHECK(sum(1,1) == doctest::Approx(5));   // 4+1
            CHECK(sum(2,2) == doctest::Approx(5));   // 4+1
            CHECK(sum(3,3) == doctest::Approx(5));   // 4+1
            CHECK(sum(0,3) == doctest::Approx(1));   // 0+1
            CHECK(sum(3,0) == doctest::Approx(1));   // 0+1
            
            // Scalar multiplication
            auto scaled = a * 0.5f;
            CHECK(scaled(0,0) == doctest::Approx(2));   // 4*0.5
            CHECK(scaled(1,1) == doctest::Approx(2));   // 4*0.5
            CHECK(scaled(2,2) == doctest::Approx(2));   // 4*0.5
            CHECK(scaled(3,3) == doctest::Approx(2));   // 4*0.5
            
            // Negation
            auto neg = -b;
            CHECK(neg(0,0) == doctest::Approx(-1));
            CHECK(neg(0,3) == doctest::Approx(-1));
            CHECK(neg(3,0) == doctest::Approx(-1));
            CHECK(neg(3,3) == doctest::Approx(-1));
        }
        
        SUBCASE("Matrix Multiplication") {
            auto prod = a * b;
            CHECK(prod(0,0) == doctest::Approx(4));   // 4*1 + 1*0 + 0*0 + 0*1
            CHECK(prod(0,1) == doctest::Approx(1));   // 4*0 + 1*1 + 0*0 + 0*0
            CHECK(prod(0,2) == doctest::Approx(0));   // 4*0 + 1*0 + 0*1 + 0*0
            CHECK(prod(0,3) == doctest::Approx(4));   // 4*1 + 1*0 + 0*0 + 0*1
            
            // Identity property
            auto a_times_i = a * identity;
            CHECK(approx_equal(a_times_i, a));
            
            // Associativity with expression
            matrix<float, 4, 4> c = identity * 2.0f;
            auto expr1 = (a * b) * c;
            auto expr2 = a * (b * c);
            matrix<float, 4, 4> result1 = expr1;
            matrix<float, 4, 4> result2 = expr2;
            CHECK(approx_equal(result1, result2, 1e-4f));
        }
        
        SUBCASE("Transpose") {
            auto at = transpose(a);
            // Check symmetry along diagonal
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    CHECK(at(i,j) == doctest::Approx(a(j,i)));
                }
            }
            
            // a is symmetric, so a^T = a
            CHECK(approx_equal(at, a));
            
            // Transpose of expression
            auto expr_t = transpose(a - identity);
            matrix<float, 4, 4> expected = transpose(matrix<float, 4, 4>(a - identity));
            matrix<float, 4, 4> result = expr_t;
            CHECK(approx_equal(result, expected));
        }
        
        SUBCASE("Determinant") {
            // For this tridiagonal matrix (4 on diagonal, 1 on off-diagonals)
            // The determinant can be computed recursively
            auto det_a = determinant(a);
            CHECK(det_a == doctest::Approx(209));  // Computed for this specific tridiagonal matrix
            
            auto det_b = determinant(b);
            CHECK(det_b == doctest::Approx(0));  // b is singular (rows 0 and 3 are identical)
            
            auto det_identity = determinant(identity);
            CHECK(det_identity == doctest::Approx(1));
            
            // Determinant of scalar multiplication
            auto det_scaled = determinant(a * 3.0f);
            CHECK(det_scaled == doctest::Approx(16929));  // 3^4 * det(A) = 81 * 209
        }
        
        SUBCASE("Inverse") {
            auto inv_a = inverse(a);
            
            // Verify A * A^(-1) = I
            auto a_inv_a = a * inv_a;
            CHECK(approx_equal(a_inv_a, identity, 1e-4f));
            
            // Verify A^(-1) * A = I
            auto inv_a_a = inv_a * a;
            CHECK(approx_equal(inv_a_a, identity, 1e-4f));
            
            // Double inverse is original
            auto inv_inv_a = inverse(inv_a);
            CHECK(approx_equal(inv_inv_a, a, 1e-4f));
            
            // Inverse of expression
            matrix<float, 4, 4> c{{3, 1, 0, 0}, {1, 3, 1, 0}, {0, 1, 3, 1}, {0, 0, 1, 3}};
            auto inv_expr = inverse(c + identity);
            auto check = (c + identity) * inv_expr;
            CHECK(approx_equal(check, identity, 1e-4f));
        }
        
        SUBCASE("Trace") {
            auto tr_a = trace(a);
            CHECK(tr_a == doctest::Approx(16));  // 4 + 4 + 4 + 4
            
            auto tr_b = trace(b);
            CHECK(tr_b == doctest::Approx(4));  // 1 + 1 + 1 + 1
            
            // Trace of product
            auto tr_ab = trace(a * b);
            CHECK(tr_ab == doctest::Approx(16));  // 4 + 4 + 4 + 4
            
            // Cyclic property: trace(ABC) = trace(CAB) = trace(BCA)
            matrix<float, 4, 4> c{{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}};
            auto tr_abc = trace(a * b * c);
            auto tr_cab = trace(c * a * b);
            auto tr_bca = trace(b * c * a);
            CHECK(tr_abc == doctest::Approx(tr_cab));
            CHECK(tr_cab == doctest::Approx(tr_bca));
        }
        
        SUBCASE("Power") {
            // Use a simpler matrix for power tests
            matrix<float, 4, 4> simple{
                {1, 1, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 2, 0},
                {0, 0, 0, 3}
            };
            
            auto simple2 = pow(simple, 2);
            CHECK(simple2(0,0) == doctest::Approx(1));   // Upper triangular structure
            CHECK(simple2(0,1) == doctest::Approx(2));   // 1*1 + 1*1
            CHECK(simple2(1,1) == doctest::Approx(1));   // 1^2
            CHECK(simple2(2,2) == doctest::Approx(4));   // 2^2
            CHECK(simple2(3,3) == doctest::Approx(9));   // 3^2
            
            auto simple0 = pow(simple, 0);
            CHECK(approx_equal(simple0, identity));
            
            // Power of expression
            auto expr_pow = pow(simple - identity, 2);
            matrix<float, 4, 4> base = simple - identity;
            matrix<float, 4, 4> expected = base * base;
            matrix<float, 4, 4> result = expr_pow;
            CHECK(approx_equal(result, expected, 1e-4f));
        }
        
        SUBCASE("Hadamard Operations") {
            auto had_mul = hadamard(a, b);
            CHECK(had_mul(0,0) == doctest::Approx(4));   // 4*1
            CHECK(had_mul(0,1) == doctest::Approx(0));   // 1*0
            CHECK(had_mul(1,1) == doctest::Approx(4));   // 4*1
            CHECK(had_mul(3,3) == doctest::Approx(4));   // 4*1
            
            // Create a matrix with no zeros for division
            matrix<float, 4, 4> c{
                {4, 2, 2, 4},
                {2, 4, 2, 2},
                {2, 2, 4, 2},
                {4, 2, 2, 4}
            };
            auto had_div = hadamard_div(c, a);
            CHECK(had_div(0,0) == doctest::Approx(1));     // 4/4
            CHECK(had_div(0,1) == doctest::Approx(2));     // 2/1
            CHECK(had_div(1,1) == doctest::Approx(1));     // 4/4
            CHECK(had_div(2,2) == doctest::Approx(1));     // 4/4
            CHECK(had_div(3,3) == doctest::Approx(1));     // 4/4
            
            // Hadamard with expressions
            auto had_expr = hadamard(a * 2.0f, c / 2.0f);
            matrix<float, 4, 4> expected = hadamard(matrix<float, 4, 4>(a * 2.0f), 
                                                   matrix<float, 4, 4>(c / 2.0f));
            matrix<float, 4, 4> result = had_expr;
            CHECK(approx_equal(result, expected));
        }
        
        SUBCASE("Complex Expression Chains") {
            // Test a complex expression: (A^T * B + I) * C^(-1) - D / 3
            matrix<float, 4, 4> c{
                {4, 1, 0, 0},
                {1, 4, 1, 0},
                {0, 1, 4, 1},
                {0, 0, 1, 4}
            };  // Positive definite, invertible
            
            matrix<float, 4, 4> d{
                {3, 3, 3, 3},
                {3, 3, 3, 3},
                {3, 3, 3, 3},
                {3, 3, 3, 3}
            };
            
            auto expr = (transpose(a) * b + identity) * inverse(c) - d / 3.0f;
            matrix<float, 4, 4> result = expr;
            
            // Verify the computation step by step
            auto at = transpose(a);
            auto at_b = at * b;
            auto at_b_plus_i = at_b + identity;
            auto c_inv = inverse(c);
            auto product = at_b_plus_i * c_inv;
            auto d_div_3 = d / 3.0f;
            auto expected = product - d_div_3;
            
            CHECK(approx_equal(result, expected, 1e-4f));
        }
    }
    
    SUBCASE("Expression Template Verification") {
        // Verify that operations create expressions, not immediate evaluations
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        
        // All these should be expression types, not matrix types
        auto add_expr = a + b;
        static_assert(!std::is_same_v<decltype(add_expr), matrix<float, 2, 2>>);
        
        auto mult_expr = a * b;
        static_assert(!std::is_same_v<decltype(mult_expr), matrix<float, 2, 2>>);
        
        auto scalar_expr = a * 2.0f;
        static_assert(!std::is_same_v<decltype(scalar_expr), matrix<float, 2, 2>>);
        
        auto transpose_expr = transpose(a + b);
        static_assert(!std::is_same_v<decltype(transpose_expr), matrix<float, 2, 2>>);
        
        auto inverse_expr = inverse(a);
        static_assert(!std::is_same_v<decltype(inverse_expr), matrix<float, 2, 2>>);
        
        auto hadamard_expr = hadamard(a, b);
        static_assert(!std::is_same_v<decltype(hadamard_expr), matrix<float, 2, 2>>);
        
        // Complex expression
        auto complex_expr = inverse(transpose(a) * b + matrix<float, 2, 2>::identity());
        static_assert(!std::is_same_v<decltype(complex_expr), matrix<float, 2, 2>>);
        
        // Force evaluation
        matrix<float, 2, 2> result = complex_expr;
        
        // Verify it computed correctly by recomputing step by step
        matrix<float, 2, 2> at = transpose(a);
        matrix<float, 2, 2> at_b = at * b;
        matrix<float, 2, 2> at_b_plus_i = at_b + matrix<float, 2, 2>::identity();
        matrix<float, 2, 2> expected = inverse(at_b_plus_i);
        CHECK(approx_equal(result, expected, 1e-4f));
    }
}