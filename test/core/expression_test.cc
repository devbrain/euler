#include <doctest/doctest.h>
#include <euler/core/expression.hh>
#include <vector>
#include <array>
#include <iostream>
#include <typeinfo>

// Simple test class that acts as an expression
template<typename T>
class test_vector : public euler::expression<test_vector<T>, T> {
public:
    using value_type = T;
    static constexpr euler::size_t static_size = 4;

    test_vector() { data.fill(T(0)); }
    test_vector(T val) { data.fill(val); }
    test_vector(T a, T b, T c, T d) : data{a, b, c, d} {}

    T eval_scalar(euler::size_t idx) const { return data[idx]; }
    T& operator[](euler::size_t idx) { return data[idx]; }
    const T& operator[](euler::size_t idx) const { return data[idx]; }

    template<typename Expr>
    test_vector& operator=(const euler::expression<Expr, T>& expr) {
        for (euler::size_t i = 0; i < static_size; ++i) {
            data[i] = expr[i];
        }
        return *this;
    }
    
private:
    std::array<T, 4> data;
};

TEST_CASE("euler::expression basic functionality") {
    using namespace euler;
    
    SUBCASE("scalar expression") {
        scalar_expression<float> s(5.0f);
        CHECK(s.eval_scalar(0) == 5.0f);
        CHECK(s.eval_scalar(100) == 5.0f);
        CHECK(s.eval_scalar(0, 0) == 5.0f);
    }
    
    SUBCASE("binary expressions") {
        test_vector<float> v1(1, 2, 3, 4);
        test_vector<float> v2(5, 6, 7, 8);
        
        auto sum = v1 + v2;
        CHECK(sum[0] == 6.0f);
        CHECK(sum[1] == 8.0f);
        CHECK(sum[2] == 10.0f);
        CHECK(sum[3] == 12.0f);
        
        auto diff = v2 - v1;
        CHECK(diff[0] == 4.0f);
        CHECK(diff[1] == 4.0f);
        CHECK(diff[2] == 4.0f);
        CHECK(diff[3] == 4.0f);
        
        auto prod = v1 * v2;
        CHECK(prod[0] == 5.0f);
        CHECK(prod[1] == 12.0f);
        CHECK(prod[2] == 21.0f);
        CHECK(prod[3] == 32.0f);
        
        auto quot = v2 / v1;
        CHECK(quot[0] == 5.0f);
        CHECK(quot[1] == 3.0f);
        CHECK(quot[2] == doctest::Approx(7.0f/3.0f));
        CHECK(quot[3] == 2.0f);
    }
    
    SUBCASE("scalar operations") {
        test_vector<float> v(2, 4, 6, 8);
        
        auto sum = v + 3.0f;
        CHECK(sum[0] == 5.0f);
        CHECK(sum[1] == 7.0f);
        CHECK(sum[2] == 9.0f);
        CHECK(sum[3] == 11.0f);
        
        auto sum2 = 3.0f + v;
        CHECK(sum2[0] == 5.0f);
        CHECK(sum2[1] == 7.0f);
        
        auto prod = v * 2.0f;
        CHECK(prod[0] == 4.0f);
        CHECK(prod[1] == 8.0f);
        CHECK(prod[2] == 12.0f);
        CHECK(prod[3] == 16.0f);
        
        auto quot = v / 2.0f;
        CHECK(quot[0] == 1.0f);
        CHECK(quot[1] == 2.0f);
        CHECK(quot[2] == 3.0f);
        CHECK(quot[3] == 4.0f);
    }
    
    SUBCASE("unary expressions") {
        test_vector<float> v(1, -2, 3, -4);
        
        auto neg = -v;
        CHECK(neg[0] == -1.0f);
        CHECK(neg[1] == 2.0f);
        CHECK(neg[2] == -3.0f);
        CHECK(neg[3] == 4.0f);
    }
    
    SUBCASE("complex expressions") {
        test_vector<float> a(1, 2, 3, 4);
        test_vector<float> b(5, 6, 7, 8);
        test_vector<float> c(9, 10, 11, 12);
        
        // (a + b) * c - 2
        auto expr = (a + b) * c - 2.0f;
        CHECK(expr[0] == (1.0f + 5.0f) * 9.0f - 2.0f);
        CHECK(expr[1] == (2.0f + 6.0f) * 10.0f - 2.0f);
        CHECK(expr[2] == (3.0f + 7.0f) * 11.0f - 2.0f);
        CHECK(expr[3] == (4.0f + 8.0f) * 12.0f - 2.0f);
        
        // Assignment from expression
        test_vector<float> result;
        result = expr;
        CHECK(result[0] == 52.0f);
        CHECK(result[1] == 78.0f);
        CHECK(result[2] == 108.0f);
        CHECK(result[3] == 142.0f);
    }
    
    SUBCASE("expression evaluation order") {
        test_vector<float> v1(10, 20, 30, 40);
        test_vector<float> v2(1, 2, 3, 4);
        
        // Should evaluate as (v1 / v2) + 5, not v1 / (v2 + 5)
        auto expr = v1 / v2 + 5.0f;
        CHECK(expr[0] == 15.0f);  // 10/1 + 5
        CHECK(expr[1] == 15.0f);  // 20/2 + 5
        CHECK(expr[2] == 15.0f);  // 30/3 + 5
        CHECK(expr[3] == 15.0f);  // 40/4 + 5
    }
}