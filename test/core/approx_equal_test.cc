#include <euler/core/approx_equal.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_expr.hh>
#include <doctest/doctest.h>

using namespace euler;

TEST_CASE("Generic approx_equal function") {
    SUBCASE("Scalar comparison") {
        CHECK(approx_equal(1.0, 1.0));
        CHECK(approx_equal(1.0f, 1.0f));
        CHECK(approx_equal(42, 42));
        CHECK(approx_equal(1.0, 1.0 + 1e-10, 1e-9));
        CHECK_FALSE(approx_equal(1.0, 2.0));
        CHECK_FALSE(approx_equal(1.0, 1.1));
        
        // Mixed types
        CHECK(approx_equal(1.0, 1.0f));
        CHECK(approx_equal(1, 1.0));
    }
    
    SUBCASE("Vector comparison") {
        vector<float, 3> v1{1.0f, 2.0f, 3.0f};
        vector<float, 3> v2{1.0f, 2.0f, 3.0f};
        vector<float, 3> v3{1.0f, 2.0f, 3.1f};
        
        CHECK(approx_equal(v1, v2));
        CHECK_FALSE(approx_equal(v1, v3));
        
        // With expressions
        auto expr = v1 + v2;
        vector<float, 3> expected{2.0f, 4.0f, 6.0f};
        CHECK(approx_equal(expr, expected));
    }
    
    SUBCASE("Matrix comparison") {
        matrix<float, 2, 2> m1{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> m2{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> m3{{1.0f, 2.0f}, {3.0f, 4.1f}};
        
        CHECK(approx_equal(m1, m2));
        CHECK_FALSE(approx_equal(m1, m3));
        
        // With expressions
        auto expr = m1 + m2;
        matrix<float, 2, 2> expected{{2.0f, 4.0f}, {6.0f, 8.0f}};
        CHECK(approx_equal(expr, expected));
    }
    
    SUBCASE("Vector-Matrix comparison with row vectors") {
        row_vector<float, 3> v;
        v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;
        matrix<float, 1, 3> m_same{{1.0f, 2.0f, 3.0f}};
        matrix<float, 1, 3> m_diff{{1.0f, 2.0f, 3.1f}};
        
        CHECK(approx_equal(v, m_same));
        CHECK(approx_equal(m_same, v));
        CHECK_FALSE(approx_equal(v, m_diff));
        
        // Test with column matrix
        matrix<float, 3, 1> m_col{{1.0f}, {2.0f}, {3.0f}};
        CHECK(approx_equal(v, m_col));
        CHECK(approx_equal(m_col, v));
    }
    
    SUBCASE("Vector-Matrix comparison with column vectors") {
        column_vector<float, 3> v;
        v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;
        matrix<float, 3, 1> m_same{{1.0f}, {2.0f}, {3.0f}};
        matrix<float, 3, 1> m_diff{{1.0f}, {2.0f}, {3.1f}};
        
        CHECK(approx_equal(v, m_same));
        CHECK(approx_equal(m_same, v));
        CHECK_FALSE(approx_equal(v, m_diff));
        
        // Test with row matrix
        matrix<float, 1, 3> m_row{{1.0f, 2.0f, 3.0f}};
        CHECK(approx_equal(v, m_row));
        CHECK(approx_equal(m_row, v));
    }
    
    SUBCASE("Row and column matrix comparison") {
        matrix<float, 1, 3> m_row{{1.0f, 2.0f, 3.0f}};
        matrix<float, 3, 1> m_col{{1.0f}, {2.0f}, {3.0f}};
        CHECK(approx_equal(m_row, m_col));
        CHECK(approx_equal(m_col, m_row));
    }
    
    SUBCASE("Mixed expressions") {
        vector<float, 3> v1{1.0f, 2.0f, 3.0f};
        vector<float, 3> v2{0.5f, 1.0f, 1.5f};
        vector<float, 3> expected{1.5f, 3.0f, 4.5f};
        
        // v1 + v2 should equal expected
        CHECK(approx_equal(v1 + v2, expected));
        
        // Test with matrix expressions
        matrix<float, 2, 2> m1{{1.0f, 2.0f}, {3.0f, 4.0f}};
        matrix<float, 2, 2> m2{{0.5f, 1.0f}, {1.5f, 2.0f}};
        matrix<float, 2, 2> m_expected{{1.5f, 3.0f}, {4.5f, 6.0f}};
        CHECK(approx_equal(m1 + m2, m_expected));
    }
    
    SUBCASE("Custom tolerance") {
        CHECK(approx_equal(1.0, 1.01, 0.1));
        CHECK_FALSE(approx_equal(1.0, 1.01, 0.001));
        
        vector<float, 2> v1{1.0f, 2.0f};
        vector<float, 2> v2{1.01f, 2.01f};
        CHECK(approx_equal(v1, v2, 0.1f));
        CHECK_FALSE(approx_equal(v1, v2, 0.001f));
    }
}