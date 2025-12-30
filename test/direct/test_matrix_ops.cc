/**
 * @file test_matrix_ops.cc
 * @brief Unit tests for direct SIMD matrix operations
 */

#include <doctest/doctest.h>
#include <euler/direct/matrix_ops.hh>
#include <euler/matrix/matrix.hh>
#include <euler/vector/vector.hh>
#include <euler/core/approx_equal.hh>
#include <random>

using namespace euler;
using namespace euler::direct;

// Test configuration
constexpr float FLOAT_TOL = 1e-5f;

// Helper to generate random matrices
template<typename T>
class RandomMatrixGenerator {
public:
    RandomMatrixGenerator(T min_val = -10, T max_val = 10) 
        : gen(std::random_device{}()), dist(min_val, max_val) {}
    
    template<size_t Rows, size_t Cols, bool ColumnMajor = true>
    matrix<T, Rows, Cols, ColumnMajor> generate() {
        matrix<T, Rows, Cols, ColumnMajor> m;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                m(i, j) = dist(gen);
            }
        }
        return m;
    }
    
private:
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
};

// =============================================================================
// Binary Operations Tests
// =============================================================================

TEST_CASE("Direct matrix addition") {
    RandomMatrixGenerator<float> rng_f;
    RandomMatrixGenerator<double> rng_d;
    
    SUBCASE("Basic addition - matrix3<float>") {
        matrix3<float> a = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> b = {{9.0f, 8.0f, 7.0f},
                          {6.0f, 5.0f, 4.0f},
                          {3.0f, 2.0f, 1.0f}};
        matrix3<float> result;
        
        add(a, b, result);
        
        matrix3<float> expected = {{10.0f, 10.0f, 10.0f},
                                 {10.0f, 10.0f, 10.0f},
                                 {10.0f, 10.0f, 10.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("Addition with aliasing - result = a + a") {
        matrix3<float> a = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> expected = {{2.0f, 4.0f, 6.0f},
                                 {8.0f, 10.0f, 12.0f},
                                 {14.0f, 16.0f, 18.0f}};
        
        add(a, a, a);  // a = a + a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Addition with aliasing - result = a + b where result is a") {
        matrix2<float> a = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> b = {{5.0f, 6.0f},
                          {7.0f, 8.0f}};
        matrix2<float> expected = {{6.0f, 8.0f},
                                 {10.0f, 12.0f}};
        
        add(a, b, a);  // a = a + b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Random addition tests - various sizes") {
        for (int test = 0; test < 10; ++test) {
            // 2x2
            {
                auto a = rng_f.generate<2, 2>();
                auto b = rng_f.generate<2, 2>();
                matrix2<float> result;
                matrix2<float> expected;
                
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        expected(i, j) = a(i, j) + b(i, j);
                    }
                }
                
                add(a, b, result);
                CHECK(approx_equal(result, expected, FLOAT_TOL));
            }
            
            // 4x4
            {
                auto a = rng_f.generate<4, 4>();
                auto b = rng_f.generate<4, 4>();
                matrix4<float> result;
                matrix4<float> expected;
                
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        expected(i, j) = a(i, j) + b(i, j);
                    }
                }
                
                add(a, b, result);
                CHECK(approx_equal(result, expected, FLOAT_TOL));
            }
        }
    }
    
    SUBCASE("Addition - double precision") {
        matrix2<double> a = {{1.0, 2.0},
                           {3.0, 4.0}};
        matrix2<double> b = {{5.0, 6.0},
                           {7.0, 8.0}};
        matrix2<double> result;
        
        add(a, b, result);
        
        CHECK(result(0, 0) == doctest::Approx(6.0));
        CHECK(result(0, 1) == doctest::Approx(8.0));
        CHECK(result(1, 0) == doctest::Approx(10.0));
        CHECK(result(1, 1) == doctest::Approx(12.0));
    }
}

TEST_CASE("Direct matrix subtraction") {
    SUBCASE("Basic subtraction") {
        matrix2<float> a = {{5.0f, 7.0f},
                          {9.0f, 11.0f}};
        matrix2<float> b = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> result;
        
        sub(a, b, result);
        
        matrix2<float> expected = {{4.0f, 5.0f},
                                 {6.0f, 7.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("Subtraction with aliasing - result = a - a") {
        matrix3<float> a = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> expected(0.0f);  // Zero matrix
        
        sub(a, a, a);  // a = a - a
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
}

// =============================================================================
// Scalar Operations Tests
// =============================================================================

TEST_CASE("Scalar matrix operations") {
    SUBCASE("Scale matrix") {
        matrix2<float> m = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> result;
        
        scale(m, 2.0f, result);
        
        matrix2<float> expected = {{2.0f, 4.0f},
                                 {6.0f, 8.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("Scale with aliasing") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> expected = {{0.5f, 1.0f, 1.5f},
                                 {2.0f, 2.5f, 3.0f},
                                 {3.5f, 4.0f, 4.5f}};
        
        scale(m, 0.5f, m);  // m = 0.5 * m
        
        CHECK(approx_equal(m, expected, FLOAT_TOL));
    }
    
    SUBCASE("Multiplication aliases") {
        matrix2<float> m = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> result1, result2;
        
        // Test both aliases
        mul(3.0f, m, result1);
        mul(m, 3.0f, result2);
        
        CHECK(approx_equal(result1, result2, FLOAT_TOL));
    }
}

// =============================================================================
// Matrix Multiplication Tests
// =============================================================================

TEST_CASE("Matrix multiplication") {
    SUBCASE("Basic 2x2 multiplication") {
        matrix2<float> a = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> b = {{5.0f, 6.0f},
                          {7.0f, 8.0f}};
        matrix2<float> result;
        
        mul(a, b, result);
        
        matrix2<float> expected = {{19.0f, 22.0f},   // 1*5+2*7, 1*6+2*8
                                 {43.0f, 50.0f}};   // 3*5+4*7, 3*6+4*8
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("Optimized 2x2 multiplication") {
        matrix2<float> a = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> b = {{5.0f, 6.0f},
                          {7.0f, 8.0f}};
        matrix2<float> result;
        
        mul(a, b, result);
        
        matrix2<float> expected = {{19.0f, 22.0f},
                                 {43.0f, 50.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("3x3 multiplication") {
        matrix3<float> a = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> b = {{9.0f, 8.0f, 7.0f},
                          {6.0f, 5.0f, 4.0f},
                          {3.0f, 2.0f, 1.0f}};
        matrix3<float> result;
        
        mul(a, b, result);
        
        // Verify a few elements
        CHECK(result(0, 0) == doctest::Approx(30.0f));  // 1*9+2*6+3*3 = 9+12+9 = 30
        CHECK(result(1, 1) == doctest::Approx(69.0f));  // 4*8+5*5+6*2 = 32+25+12 = 69
        CHECK(result(2, 2) == doctest::Approx(90.0f));  // 7*7+8*4+9*1 = 49+32+9 = 90
    }
    
    SUBCASE("4x4 multiplication") {
        matrix4<float> a = matrix4<float>::identity();
        matrix4<float> b = {{1.0f, 2.0f, 3.0f, 4.0f},
                          {5.0f, 6.0f, 7.0f, 8.0f},
                          {9.0f, 10.0f, 11.0f, 12.0f},
                          {13.0f, 14.0f, 15.0f, 16.0f}};
        matrix4<float> result;
        
        mul(a, b, result);
        
        // Identity * B = B
        CHECK(approx_equal(result, b, FLOAT_TOL));
    }
    
    SUBCASE("Matrix multiplication with aliasing") {
        matrix2<float> a = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        matrix2<float> b = {{5.0f, 6.0f},
                          {7.0f, 8.0f}};
        matrix2<float> expected = {{19.0f, 22.0f},
                                 {43.0f, 50.0f}};
        
        mul(a, b, a);  // a = a * b
        
        CHECK(approx_equal(a, expected, FLOAT_TOL));
    }
    
    SUBCASE("Non-square matrix multiplication") {
        matrix<float, 2, 3> a = {{1.0f, 2.0f, 3.0f},
                                {4.0f, 5.0f, 6.0f}};
        matrix<float, 3, 2> b = {{7.0f, 8.0f},
                                {9.0f, 10.0f},
                                {11.0f, 12.0f}};
        matrix<float, 2, 2> result;
        
        mul(a, b, result);
        
        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12]
        //           [4*7+5*9+6*11, 4*8+5*10+6*12]
        matrix<float, 2, 2> expected = {{58.0f, 64.0f},
                                       {139.0f, 154.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
}

// =============================================================================
// Matrix Transpose Tests
// =============================================================================

TEST_CASE("Matrix transpose") {
    SUBCASE("Basic transpose") {
        matrix<float, 2, 3> m = {{1.0f, 2.0f, 3.0f},
                                {4.0f, 5.0f, 6.0f}};
        matrix<float, 3, 2> result;
        
        transpose(m, result);
        
        matrix<float, 3, 2> expected = {{1.0f, 4.0f},
                                       {2.0f, 5.0f},
                                       {3.0f, 6.0f}};
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("In-place transpose for square matrix") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> expected = {{1.0f, 4.0f, 7.0f},
                                 {2.0f, 5.0f, 8.0f},
                                 {3.0f, 6.0f, 9.0f}};
        
        transpose(m, m);  // In-place transpose
        
        CHECK(approx_equal(m, expected, FLOAT_TOL));
    }
}

// =============================================================================
// Scalar-returning Operations Tests
// =============================================================================

TEST_CASE("Matrix trace") {
    SUBCASE("Trace of identity matrix") {
        matrix4<float> m = matrix4<float>::identity();
        
        float tr = trace(m);
        
        CHECK(tr == doctest::Approx(4.0f));
    }
    
    SUBCASE("Trace of general matrix") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        
        float tr = trace(m);
        
        CHECK(tr == doctest::Approx(15.0f));  // 1 + 5 + 9
    }
}

TEST_CASE("Matrix determinant") {
    SUBCASE("2x2 determinant") {
        matrix2<float> m = {{1.0f, 2.0f},
                          {3.0f, 4.0f}};
        
        float det = determinant(m);
        
        CHECK(det == doctest::Approx(-2.0f));  // 1*4 - 2*3
    }
    
    SUBCASE("3x3 determinant") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        
        float det = determinant(m);
        
        CHECK(std::abs(det) < FLOAT_TOL);  // This matrix is singular
    }
    
    SUBCASE("4x4 determinant") {
        matrix4<float> m = matrix4<float>::identity();
        
        float det = determinant(m);
        
        CHECK(det == doctest::Approx(1.0f));
    }
}

// =============================================================================
// Matrix-Vector Operations Tests
// =============================================================================

TEST_CASE("Matrix-vector operations") {
    SUBCASE("Matrix-vector multiplication") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> result;
        
        matvec(m, v, result);
        
        vec3<float> expected(14.0f, 32.0f, 50.0f);  // [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3]
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
    
    SUBCASE("Matrix-vector multiplication with aliasing") {
        matrix3<float> m = {{2.0f, 0.0f, 0.0f},
                          {0.0f, 2.0f, 0.0f},
                          {0.0f, 0.0f, 2.0f}};
        vec3<float> v(1.0f, 2.0f, 3.0f);
        vec3<float> expected(2.0f, 4.0f, 6.0f);
        
        matvec(m, v, v);  // v = m * v
        
        CHECK(approx_equal(v, expected, FLOAT_TOL));
    }
    
    SUBCASE("Vector-matrix multiplication") {
        vec3<float> v(1.0f, 2.0f, 3.0f);
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        vec3<float> result;
        
        vecmat(v, m, result);
        
        vec3<float> expected(30.0f, 36.0f, 42.0f);  // [1*1+2*4+3*7, 1*2+2*5+3*8, 1*3+2*6+3*9]
        
        CHECK(approx_equal(result, expected, FLOAT_TOL));
    }
}

// =============================================================================
// Edge Cases and Special Values
// =============================================================================

TEST_CASE("Edge cases and special values") {
    SUBCASE("Operations with zero matrices") {
        matrix3<float> zero(0.0f);
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> result;
        
        add(zero, m, result);
        CHECK(approx_equal(result, m, FLOAT_TOL));
        
        mul(zero, m, result);
        CHECK(approx_equal(result, zero, FLOAT_TOL));
    }
    
    SUBCASE("Identity matrix operations") {
        matrix3<float> id = matrix3<float>::identity();
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                          {4.0f, 5.0f, 6.0f},
                          {7.0f, 8.0f, 9.0f}};
        matrix3<float> result;
        
        mul(id, m, result);
        CHECK(approx_equal(result, m, FLOAT_TOL));
        
        mul(m, id, result);
        CHECK(approx_equal(result, m, FLOAT_TOL));
    }
}

// =============================================================================
// Matrix Inverse and Related Operations Tests
// =============================================================================

TEST_CASE("Matrix inverse operations") {
    SUBCASE("2x2 matrix inverse") {
        matrix2<float> m = {{4.0f, 7.0f},
                           {2.0f, 6.0f}};
        matrix2<float> result;
        
        inverse(m, result);
        
        // Check that m * m^(-1) = I
        matrix2<float> identity_check;
        mul(m, result, identity_check);
        
        CHECK(identity_check(0, 0) == doctest::Approx(1.0f).epsilon(FLOAT_TOL));
        CHECK(identity_check(0, 1) == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        CHECK(identity_check(1, 0) == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        CHECK(identity_check(1, 1) == doctest::Approx(1.0f).epsilon(FLOAT_TOL));
    }
    
    SUBCASE("3x3 matrix inverse") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                           {0.0f, 1.0f, 4.0f},
                           {5.0f, 6.0f, 0.0f}};
        matrix3<float> result;
        
        inverse(m, result);
        
        // Check that m * m^(-1) = I
        matrix3<float> identity_check;
        mul(m, result, identity_check);
        
        auto identity = matrix3<float>::identity();
        CHECK(approx_equal(identity_check, identity, FLOAT_TOL));
    }
    
    SUBCASE("4x4 matrix inverse") {
        matrix4<float> m = {{2.0f, 0.0f, 0.0f, 1.0f},
                           {0.0f, 1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f, 1.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f, 1.0f}};
        matrix4<float> result;
        
        inverse(m, result);
        
        // Check that m * m^(-1) = I
        matrix4<float> identity_check;
        mul(m, result, identity_check);
        
        auto identity = matrix4<float>::identity();
        CHECK(approx_equal(identity_check, identity, FLOAT_TOL));
    }
    
#ifdef EULER_DEBUG
    SUBCASE("Singular matrix throws") {
        matrix2<float> singular = {{1.0f, 2.0f},
                                  {2.0f, 4.0f}};  // Rows are linearly dependent
        matrix2<float> result;

        CHECK_THROWS_AS(inverse(singular, result), std::runtime_error);
    }
#endif
}

TEST_CASE("Adjugate and cofactor matrices") {
    SUBCASE("2x2 adjugate") {
        matrix2<float> m = {{3.0f, 7.0f},
                           {2.0f, 5.0f}};
        matrix2<float> adj;
        
        adjugate(m, adj);
        
        CHECK(adj(0, 0) == doctest::Approx(5.0f));
        CHECK(adj(0, 1) == doctest::Approx(-7.0f));
        CHECK(adj(1, 0) == doctest::Approx(-2.0f));
        CHECK(adj(1, 1) == doctest::Approx(3.0f));
        
        // Check that m * adj(m) = det(m) * I
        matrix2<float> check;
        mul(m, adj, check);
        float det = determinant(m);
        
        CHECK(check(0, 0) == doctest::Approx(det));
        CHECK(check(0, 1) == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        CHECK(check(1, 0) == doctest::Approx(0.0f).epsilon(FLOAT_TOL));
        CHECK(check(1, 1) == doctest::Approx(det));
    }
    
    SUBCASE("3x3 adjugate") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                           {4.0f, 5.0f, 6.0f},
                           {7.0f, 8.0f, 10.0f}};
        matrix3<float> adj;
        
        adjugate(m, adj);
        
        // Check that m * adj(m) = det(m) * I
        matrix3<float> check;
        mul(m, adj, check);
        float det = determinant(m);
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                float expected = (i == j) ? det : 0.0f;
                CHECK(check(i, j) == doctest::Approx(expected).epsilon(FLOAT_TOL));
            }
        }
    }
    
    SUBCASE("2x2 cofactor") {
        matrix2<float> m = {{3.0f, 7.0f},
                           {2.0f, 5.0f}};
        matrix2<float> cof;
        
        cofactor(m, cof);
        
        CHECK(cof(0, 0) == doctest::Approx(5.0f));
        CHECK(cof(0, 1) == doctest::Approx(-2.0f));
        CHECK(cof(1, 0) == doctest::Approx(-7.0f));
        CHECK(cof(1, 1) == doctest::Approx(3.0f));
    }
    
    SUBCASE("3x3 cofactor") {
        matrix3<float> m = {{1.0f, 2.0f, 3.0f},
                           {4.0f, 5.0f, 6.0f},
                           {7.0f, 8.0f, 10.0f}};
        matrix3<float> cof;
        
        cofactor(m, cof);
        
        // The adjugate is the transpose of the cofactor
        matrix3<float> adj;
        adjugate(m, adj);
        
        // Check that adj = transpose(cof)
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(adj(i, j) == doctest::Approx(cof(j, i)));
            }
        }
    }
}
