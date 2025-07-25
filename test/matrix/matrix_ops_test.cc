#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/vector/vector.hh>
#include <doctest/doctest.h>
#include <cmath>

using namespace euler;

TEST_CASE("Matrix-Matrix Multiplication") {
    SUBCASE("2x2 multiplication") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        
        auto c = a * b;
        
        CHECK(c(0,0) == doctest::Approx(19));  // 1*5 + 2*7
        CHECK(c(0,1) == doctest::Approx(22));  // 1*6 + 2*8
        CHECK(c(1,0) == doctest::Approx(43));  // 3*5 + 4*7
        CHECK(c(1,1) == doctest::Approx(50));  // 3*6 + 4*8
    }
    
    SUBCASE("3x3 multiplication") {
        matrix<float, 3, 3> a{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        matrix<float, 3, 3> b{
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1}
        };
        
        auto c = a * b;
        
        CHECK(c(0,0) == doctest::Approx(30));   // 1*9 + 2*6 + 3*3
        CHECK(c(1,1) == doctest::Approx(69));   // 4*8 + 5*5 + 6*2
        CHECK(c(2,2) == doctest::Approx(90));   // 7*7 + 8*4 + 9*1
    }
    
    SUBCASE("4x4 multiplication") {
        matrix<float, 4, 4> a = matrix<float, 4, 4>::identity();
        matrix<float, 4, 4> b{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };
        
        auto c = a * b;
        
        // Identity * B = B
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                CHECK(c(i,j) == doctest::Approx(b(i,j)));
            }
        }
    }
    
    SUBCASE("Non-square multiplication") {
        matrix<float, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        matrix<float, 3, 2> b{{7, 8}, {9, 10}, {11, 12}};
        
        auto c = a * b;  // Result is 2x2
        
        CHECK(c(0,0) == doctest::Approx(58));   // 1*7 + 2*9 + 3*11
        CHECK(c(0,1) == doctest::Approx(64));   // 1*8 + 2*10 + 3*12
        CHECK(c(1,0) == doctest::Approx(139));  // 4*7 + 5*9 + 6*11
        CHECK(c(1,1) == doctest::Approx(154));  // 4*8 + 5*10 + 6*12
    }
}

TEST_CASE("Matrix-Vector Multiplication") {
    SUBCASE("Matrix * column vector") {
        matrix<float, 3, 3> m{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        vec3f v(1, 2, 3);
        
        auto result = m * v;
        
        CHECK(result[0] == doctest::Approx(14));  // 1*1 + 2*2 + 3*3
        CHECK(result[1] == doctest::Approx(32));  // 4*1 + 5*2 + 6*3
        CHECK(result[2] == doctest::Approx(50));  // 7*1 + 8*2 + 9*3
    }
    
    SUBCASE("Row vector * matrix") {
        row_vector<float, 3> v(1, 2, 3);
        matrix<float, 3, 3> m{
            {1, 4, 7},
            {2, 5, 8},
            {3, 6, 9}
        };
        
        auto result = v * m;
        
        CHECK(result[0] == doctest::Approx(14));  // 1*1 + 2*2 + 3*3
        CHECK(result[1] == doctest::Approx(32));  // 1*4 + 2*5 + 3*6
        CHECK(result[2] == doctest::Approx(50));  // 1*7 + 2*8 + 3*9
    }
}

TEST_CASE("Matrix Transpose") {
    SUBCASE("2x2 transpose") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        auto b = transpose(a);
        
        CHECK(b(0,0) == doctest::Approx(1));
        CHECK(b(0,1) == doctest::Approx(3));
        CHECK(b(1,0) == doctest::Approx(2));
        CHECK(b(1,1) == doctest::Approx(4));
    }
    
    SUBCASE("Non-square transpose") {
        matrix<float, 2, 3> a{{1, 2, 3}, {4, 5, 6}};
        auto b = transpose(a);  // Result is 3x2
        
        CHECK(b(0,0) == doctest::Approx(1));
        CHECK(b(0,1) == doctest::Approx(4));
        CHECK(b(1,0) == doctest::Approx(2));
        CHECK(b(1,1) == doctest::Approx(5));
        CHECK(b(2,0) == doctest::Approx(3));
        CHECK(b(2,1) == doctest::Approx(6));
    }
}

TEST_CASE("Matrix Determinant") {
    SUBCASE("2x2 determinant") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        float det = determinant(a);
        CHECK(det == doctest::Approx(-2));  // 1*4 - 2*3
    }
    
    SUBCASE("3x3 determinant") {
        matrix<float, 3, 3> a{
            {1, 2, 3},
            {0, 1, 4},
            {5, 6, 0}
        };
        float det = determinant(a);
        CHECK(det == doctest::Approx(1));  // Calculated by hand
    }
    
    SUBCASE("4x4 determinant") {
        matrix<float, 4, 4> identity = matrix<float, 4, 4>::identity();
        float det = determinant(identity);
        CHECK(det == doctest::Approx(1));
    }
}

TEST_CASE("Matrix Inverse") {
    SUBCASE("2x2 inverse") {
        matrix<float, 2, 2> a{{4, 7}, {2, 6}};
        auto inv = inverse(a);
        auto identity = a * inv;
        
        CHECK(identity(0,0) == doctest::Approx(1).epsilon(0.0001));
        CHECK(identity(0,1) == doctest::Approx(0).epsilon(0.0001));
        CHECK(identity(1,0) == doctest::Approx(0).epsilon(0.0001));
        CHECK(identity(1,1) == doctest::Approx(1).epsilon(0.0001));
    }
    
    SUBCASE("3x3 inverse") {
        matrix<float, 3, 3> a{
            {1, 2, 3},
            {0, 1, 4},
            {5, 6, 0}
        };
        auto inv = inverse(a);
        auto identity = a * inv;
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                float expected = (i == j) ? 1.0f : 0.0f;
                CHECK(identity(i,j) == doctest::Approx(expected).epsilon(0.0001));
            }
        }
    }
}

TEST_CASE("Matrix Additional Operations") {
    SUBCASE("Trace") {
        matrix<float, 3, 3> a{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        float tr = trace(a);
        CHECK(tr == doctest::Approx(15));  // 1 + 5 + 9
    }
    
    SUBCASE("Hadamard product") {
        matrix<float, 2, 2> a{{1, 2}, {3, 4}};
        matrix<float, 2, 2> b{{5, 6}, {7, 8}};
        auto c = hadamard(a, b);
        
        CHECK(c(0,0) == doctest::Approx(5));   // 1*5
        CHECK(c(0,1) == doctest::Approx(12));  // 2*6
        CHECK(c(1,0) == doctest::Approx(21));  // 3*7
        CHECK(c(1,1) == doctest::Approx(32));  // 4*8
    }
    
    SUBCASE("Frobenius norm") {
        matrix<float, 2, 2> a{{3, 0}, {0, 4}};
        float norm = frobenius_norm(a);
        CHECK(norm == doctest::Approx(5));  // sqrt(9 + 0 + 0 + 16)
    }
    
    SUBCASE("Outer product") {
        vec3f u(1, 2, 3);
        vec3f v(4, 5, 6);
        auto m = outer_product(u, v);
        
        CHECK(m(0,0) == doctest::Approx(4));   // 1*4
        CHECK(m(0,1) == doctest::Approx(5));   // 1*5
        CHECK(m(1,2) == doctest::Approx(12));  // 2*6
        CHECK(m(2,2) == doctest::Approx(18));  // 3*6
    }
    
    SUBCASE("Matrix power") {
        matrix<float, 2, 2> a{{1, 1}, {0, 1}};
        auto a2 = pow(a, 2);
        auto a3 = pow(a, 3);
        
        // a^2
        CHECK(a2(0,0) == doctest::Approx(1));
        CHECK(a2(0,1) == doctest::Approx(2));
        CHECK(a2(1,0) == doctest::Approx(0));
        CHECK(a2(1,1) == doctest::Approx(1));
        
        // a^3
        CHECK(a3(0,0) == doctest::Approx(1));
        CHECK(a3(0,1) == doctest::Approx(3));
        CHECK(a3(1,0) == doctest::Approx(0));
        CHECK(a3(1,1) == doctest::Approx(1));
    }
}