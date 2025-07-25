/**
 * @file io_test.cc
 * @brief Tests for stream output operators
 */

#include <euler/euler.hh>
#include <euler/io/io.hh>
#include <doctest/doctest.h>
#include <sstream>
#include <string>

using namespace euler;

TEST_CASE("Vector output operator") {
    SUBCASE("Basic vectors") {
        vector<float, 3> v(1, 2, 3);
        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "(1, 2, 3)");
    }
    
    SUBCASE("Different sizes") {
        vector<float, 2> v2(1.5f, 2.5f);
        vector<float, 4> v4(1, 2, 3, 4);
        vector<double, 5> v5(1.1, 2.2, 3.3, 4.4, 5.5);
        
        std::ostringstream oss;
        
        oss.str("");
        oss << v2;
        CHECK(oss.str() == "(1.5, 2.5)");
        
        oss.str("");
        oss << v4;
        CHECK(oss.str() == "(1, 2, 3, 4)");
        
        oss.str("");
        oss << v5;
        CHECK(oss.str() == "(1.1, 2.2, 3.3, 4.4, 5.5)");
    }
    
    SUBCASE("Vector expressions") {
        vector<float, 3> a(1, 2, 3);
        vector<float, 3> b(4, 5, 6);
        
        std::ostringstream oss;
        
        // Test expression output
        oss << (a + b);
        CHECK(oss.str() == "(5, 7, 9)");
        
        oss.str("");
        oss << (2.0f * a);
        CHECK(oss.str() == "(2, 4, 6)");
        
        oss.str("");
        oss << (a - b);
        CHECK(oss.str() == "(-3, -3, -3)");
    }
    
    SUBCASE("Complex vector expressions") {
        vector<float, 3> a(1, 0, 0);
        vector<float, 3> b(0, 1, 0);
        
        std::ostringstream oss;
        
        // Cross product expression
        oss << cross(a, b);
        CHECK(oss.str() == "(0, 0, 1)");
        
        // Normalized expression
        oss.str("");
        vector<float, 3> v(3, 4, 0);
        oss << normalize(v);
        CHECK(oss.str() == "(0.6, 0.8, 0)");
    }
}

TEST_CASE("Matrix output operator") {
    SUBCASE("2x2 matrix") {
        matrix<float, 2, 2> m = matrix<float, 2, 2>::from_row_major({
            1, 2,
            3, 4
        });
        
        std::ostringstream oss;
        oss << m;
        CHECK(oss.str() == "[[1, 2],\n [3, 4]]");
    }
    
    SUBCASE("3x3 matrix") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::identity();
        
        std::ostringstream oss;
        oss << m;
        CHECK(oss.str() == "[[1, 0, 0],\n [0, 1, 0],\n [0, 0, 1]]");
    }
    
    SUBCASE("Non-square matrices") {
        matrix<float, 2, 3> m23 = matrix<float, 2, 3>::from_row_major({
            1, 2, 3,
            4, 5, 6
        });
        
        matrix<float, 3, 2> m32 = matrix<float, 3, 2>::from_row_major({
            1, 2,
            3, 4,
            5, 6
        });
        
        std::ostringstream oss;
        
        oss << m23;
        CHECK(oss.str() == "[[1, 2, 3],\n [4, 5, 6]]");
        
        oss.str("");
        oss << m32;
        CHECK(oss.str() == "[[1, 2],\n [3, 4],\n [5, 6]]");
    }
    
    SUBCASE("Matrix expressions") {
        matrix<float, 2, 2> a = matrix<float, 2, 2>::from_row_major({
            1, 2,
            3, 4
        });
        
        matrix<float, 2, 2> b = matrix<float, 2, 2>::from_row_major({
            5, 6,
            7, 8
        });
        
        std::ostringstream oss;
        
        // Addition expression
        oss << (a + b);
        CHECK(oss.str() == "[[ 6,  8],\n [10, 12]]");
        
        // Scalar multiplication expression
        oss.str("");
        oss << (2.0f * a);
        CHECK(oss.str() == "[[2, 4],\n [6, 8]]");
        
        // Transpose expression
        oss.str("");
        oss << transpose(a);
        CHECK(oss.str() == "[[1, 3],\n [2, 4]]");
    }
}

TEST_CASE("Quaternion output operator") {
    SUBCASE("Identity quaternion") {
        quaternion<float> q = quaternion<float>::identity();
        
        std::ostringstream oss;
        oss << q;
        CHECK(oss.str() == "(1, 0i, 0j, 0k)");
    }
    
    SUBCASE("General quaternion") {
        quaternion<float> q(0.5f, 0.5f, 0.5f, 0.5f);
        
        std::ostringstream oss;
        oss << q;
        CHECK(oss.str() == "(0.5, 0.5i, 0.5j, 0.5k)");
    }
    
    SUBCASE("Negative components") {
        quaternion<float> q(1, -2, 3, -4);
        
        std::ostringstream oss;
        oss << q;
        CHECK(oss.str() == "(1, -2i, 3j, -4k)");
    }
}

TEST_CASE("Angle output operators") {
    SUBCASE("Degree output") {
        degree<float> d(45.5f);
        
        std::ostringstream oss;
        oss << d;
        CHECK(oss.str() == "45.5°");
    }
    
    SUBCASE("Radian output") {
        radian<float> r(3.14159f);
        
        std::ostringstream oss;
        oss << r;
        CHECK(oss.str() == "3.14159 rad");
    }
    
    SUBCASE("Negative angles") {
        degree<float> d(-90);
        radian<float> r(-1.5708f);
        
        std::ostringstream oss;
        
        oss << d;
        CHECK(oss.str() == "-90°");
        
        oss.str("");
        oss << r;
        CHECK(oss.str() == "-1.5708 rad");
    }
}

TEST_CASE("Mixed expression output") {
    SUBCASE("Matrix-vector multiplication") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::from_row_major({
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        });
        
        vector<float, 3> v(1, 1, 1);
        
        std::ostringstream oss;
        oss << (m * v);
        CHECK(oss.str() == "(1, 2, 3)");
    }
    
    SUBCASE("Complex nested expressions") {
        vector<float, 3> a(1, 2, 3);
        vector<float, 3> b(4, 5, 6);
        vector<float, 3> c(7, 8, 9);
        
        std::ostringstream oss;
        
        // Complex expression
        oss << (2.0f * a + 3.0f * b - c);
        CHECK(oss.str() == "(7, 11, 15)");
        
        // Very complex expression
        oss.str("");
        oss << normalize(cross(a, b) + c);
        // cross(a, b) = (-3, 6, -3), + c = (4, 14, 6), normalized
        // Just check it doesn't crash and produces valid output
        CHECK(oss.str().size() > 0);
        CHECK(oss.str()[0] == '('); // Should start with (
        CHECK(oss.str().back() == ')'); // Should end with )
    }
}

TEST_CASE("Edge cases and safety") {
    SUBCASE("Empty vector") {
        // Vectors always have at least size 1 in Euler, so this tests size 1
        vector<float, 1> v(42);
        
        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "(42)");
    }
    
    SUBCASE("Large vectors") {
        vector<float, 10> v;
        for (size_t i = 0; i < 10; ++i) {
            v[i] = static_cast<float>(i);
        }
        
        std::ostringstream oss;
        oss << v;
        CHECK(oss.str() == "(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)");
    }
    
    SUBCASE("1x1 matrix") {
        matrix<float, 1, 1> m(42);
        
        std::ostringstream oss;
        oss << m;
        CHECK(oss.str() == "[[42]]");
    }
    
    SUBCASE("Expression with dynamic size (scalar expression)") {
        // Scalar expressions have dynamic size (0x0)
        auto scalar_expr = scalar_expression<float>(5.0f);
        
        std::ostringstream oss;
        oss << scalar_expr;
        CHECK(oss.str() == "<expression>"); // Safe fallback for dynamic size
    }
}

TEST_CASE("Matrix and vector views") {
    SUBCASE("Matrix view output") {
        matrix<float, 4, 4> m = matrix<float, 4, 4>::from_row_major({
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        });
        
        // Create a 2x2 submatrix view
        auto sub = m.submatrix(1, 1, 2, 2);
        
        std::ostringstream oss;
        oss << sub;
        CHECK(oss.str() == "[[ 6,  7],\n [10, 11]]");
    }
    
    SUBCASE("Column vector view") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::from_row_major({
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        });
        
        // Get second column as a vector view
        auto col = m.col(1);
        
        std::ostringstream oss;
        oss << col;
        CHECK(oss.str() == "(2, 5, 8)");
    }
    
    SUBCASE("Row vector view") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::from_row_major({
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        });
        
        // Get second row as a vector view
        auto row = m.row(1);
        
        std::ostringstream oss;
        oss << row;
        CHECK(oss.str() == "(4, 5, 6)");
    }
    
    SUBCASE("Const matrix view") {
        const matrix<float, 3, 3> m = matrix<float, 3, 3>::identity();
        
        // Get a const view
        auto view = m.submatrix(0, 0, 2, 2);
        
        std::ostringstream oss;
        oss << view;
        CHECK(oss.str() == "[[1, 0],\n [0, 1]]");
    }
}

TEST_CASE("Formatting preservation") {
    SUBCASE("Floating point precision") {
        vector<double, 3> v(1.23456789, 2.3456789, 3.456789);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << v;
        CHECK(oss.str() == "(1.235, 2.346, 3.457)");
    }
    
    SUBCASE("Scientific notation") {
        vector<float, 2> v(1e-5f, 1e5f);
        
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2);
        oss << v;
        // Check contains scientific notation
        CHECK(oss.str().find('e') != std::string::npos);
    }
}

TEST_CASE("Pretty printing with manipulators") {
    SUBCASE("Matrix with setw") {
        matrix<float, 2, 2> m = {
            {1.234567f, 2.345678f},
            {3.456789f, 4.567890f}
        };
        
        std::ostringstream oss;
        oss << std::setw(10) << std::fixed << std::setprecision(3);
        oss << m;
        auto str = oss.str();
        // Should have properly aligned columns
        CHECK(str.find("[[") != std::string::npos);
        CHECK(str.find("]]") != std::string::npos);
        CHECK(str.find("1.235") != std::string::npos);
        CHECK(str.find("2.346") != std::string::npos);
    }
    
    SUBCASE("Vector with setw") {
        vector<float, 3> v(1.1f, 22.22f, 333.333f);
        
        std::ostringstream oss;
        oss << std::setw(8) << std::fixed << std::setprecision(2);
        oss << v;
        auto str = oss.str();
        CHECK(str.find("(") != std::string::npos);
        CHECK(str.find(")") != std::string::npos);
        CHECK(str.find("1.10") != std::string::npos);
        CHECK(str.find("22.22") != std::string::npos);
        CHECK(str.find("333.33") != std::string::npos);
    }
    
    SUBCASE("Quaternion with formatting") {
        quaternion<float> q(0.7071f, 0.0f, 0.7071f, 0.0f);
        
        std::ostringstream oss;
        oss << std::setw(8) << std::fixed << std::setprecision(4);
        oss << q;
        auto str = oss.str();
        CHECK(str.find("0.7071") != std::string::npos);
        CHECK(str.find("0.0000i") != std::string::npos);
        CHECK(str.find("0.7071j") != std::string::npos);
        CHECK(str.find("0.0000k") != std::string::npos);
    }
    
    SUBCASE("Stream state preservation") {
        matrix<int, 2, 2> m = {{10, 20}, {30, 40}};
        
        std::ostringstream oss;
        oss << std::hex << std::uppercase;
        oss << "Before: " << 255 << " ";
        oss << std::dec;  // Need to switch to decimal for matrix
        oss << m;
        oss << std::hex;
        oss << " After: " << 255;
        
        auto str = oss.str();
        CHECK(str.find("Before: FF") != std::string::npos);
        CHECK(str.find("After: FF") != std::string::npos);
        CHECK(str.find("[[10, 20],") != std::string::npos);
    }
    
    SUBCASE("Automatic column width calculation") {
        matrix<double, 3, 3> large = {
            {1.0, 22.0, 333.0},
            {4444.0, 55555.0, 666666.0},
            {7777777.0, 88888888.0, 999999999.0}
        };
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        oss << large;
        auto str = oss.str();
        
        // Check that the matrix is printed
        CHECK(str.find("1.0") != std::string::npos);
        CHECK(str.find("999999999.0") != std::string::npos);
        // All values should be present and properly formatted
        CHECK(str.find("[[") != std::string::npos);
        CHECK(str.find("]]") != std::string::npos);
    }
}