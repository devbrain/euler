#pragma once

#include <euler/core/expression.hh>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_traits.hh>
#include <cmath>

namespace euler {

// Base class for vector expressions that provides matrix-like interface
template<typename Derived, typename T>
class vector_expression_base : public expression<Derived, T> {
public:
    using value_type = T;
    
    // Matrix-like interface for vector expressions (treated as column vectors)
    value_type operator()(size_t row, size_t col) const {
        if (col == 0) {
            return static_cast<const Derived*>(this)->eval_scalar(row);
        }
        return value_type(0);
    }
    
    value_type operator[](size_t idx) const {
        return static_cast<const Derived*>(this)->eval_scalar(idx);
    }
};

// Helper to create unary expressions with custom operations
template<typename Expr, typename Op>
class custom_unary_expr : public expression<custom_unary_expr<Expr, Op>, typename Expr::value_type> {
public:
    using value_type = typename Expr::value_type;
    using expr_storage = typename expression_storage<Expr>::type;
    
    custom_unary_expr(const Expr& expr, Op op) : expr_(expr), op_(op) {}
    
    value_type eval_scalar(size_t idx) const {
        return op_(expr_.eval_scalar(idx));
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return op_(expr_.eval_scalar(row, col));
    }
    
    // Matrix-like interface
    value_type operator()(size_t row, size_t col) const {
        return eval_scalar(row, col);
    }
    
    value_type operator[](size_t idx) const {
        return eval_scalar(idx);
    }
    
private:
    expr_storage expr_;
    Op op_;
};

// Helper to create binary expressions with custom operations
template<typename Expr1, typename Expr2, typename Op>
class custom_binary_expr : public expression<custom_binary_expr<Expr1, Expr2, Op>, typename Expr1::value_type> {
public:
    using value_type = typename Expr1::value_type;
    using expr1_storage = typename expression_storage<Expr1>::type;
    using expr2_storage = typename expression_storage<Expr2>::type;
    
    custom_binary_expr(const Expr1& e1, const Expr2& e2, Op op) 
        : expr1_(e1), expr2_(e2), op_(op) {}
    
    value_type eval_scalar(size_t idx) const {
        return op_(expr1_.eval_scalar(idx), expr2_.eval_scalar(idx));
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return op_(expr1_.eval_scalar(row, col), expr2_.eval_scalar(row, col));
    }
    
    // Matrix-like interface
    value_type operator()(size_t row, size_t col) const {
        return eval_scalar(row, col);
    }
    
    value_type operator[](size_t idx) const {
        return eval_scalar(idx);
    }
    
private:
    expr1_storage expr1_;
    expr2_storage expr2_;
    Op op_;
};

// Factory functions for custom expressions
template<typename Expr, typename Op>
auto make_unary_expression(const expression<Expr, typename Expr::value_type>& expr, Op op) {
    return custom_unary_expr<Expr, Op>(static_cast<const Expr&>(expr), op);
}

template<typename Expr1, typename Expr2, typename Op>
auto make_binary_expression(const expression<Expr1, typename Expr1::value_type>& e1,
                           const expression<Expr2, typename Expr2::value_type>& e2, Op op) {
    return custom_binary_expr<Expr1, Expr2, Op>(static_cast<const Expr1&>(e1), 
                                                static_cast<const Expr2&>(e2), op);
}

// Expression template for vector normalization
template<typename Expr, size_t Size>
class normalized_expr : public vector_expression_base<normalized_expr<Expr, Size>, typename Expr::value_type> {
public:
    using value_type = typename Expr::value_type;
    using expr_storage = typename expression_storage<Expr>::type;
    static constexpr size_t size = Size;
    
    explicit normalized_expr(const Expr& expr) : expr_(expr) {
        // Compute length once
        // Note: This evaluates the expression multiple times during construction
        value_type sum = 0;
        for (size_t i = 0; i < size; ++i) {
            value_type val = expr_.eval_scalar(i);
            sum += val * val;
        }
        inv_length_ = value_type(1) / std::sqrt(sum);
    }
    
    value_type eval_scalar(size_t idx) const {
        return expr_.eval_scalar(idx) * inv_length_;
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return expr_.eval_scalar(row, col) * inv_length_;
    }
    
private:
    expr_storage expr_;
    value_type inv_length_;
};

// Expression template for scalar * vector
template<typename Expr, typename T>
class scalar_vector_multiply_expr : public expression<scalar_vector_multiply_expr<Expr, T>, T> {
public:
    using value_type = T;
    
    scalar_vector_multiply_expr(T scalar_value, const Expr& expr) 
        : scalar_(scalar_value), expr_(expr) {}
    
    T eval_scalar(size_t idx) const {
        return scalar_ * expr_[idx];
    }
    
    T eval_scalar(size_t row, size_t col) const {
        return scalar_ * expr_(row, col);
    }
    
private:
    T scalar_;
    const Expr& expr_;
};

// Expression template for dot product result * vector
template<typename Vec1, typename Vec2, typename Vec3>
class dot_multiply_expr : public expression<dot_multiply_expr<Vec1, Vec2, Vec3>, typename Vec1::value_type> {
public:
    using value_type = typename Vec1::value_type;
    
    dot_multiply_expr(const Vec1& v1, const Vec2& v2, const Vec3& v3)
        : dot_result_(dot(v1, v2)), v3_(v3) {}
    
    value_type eval_scalar(size_t idx) const {
        return dot_result_ * v3_[idx];
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return dot_result_ * v3_(row, col);
    }
    
private:
    value_type dot_result_;
    const Vec3& v3_;
};

// Expression template for cross product
template<typename Vec1, typename Vec2>
class cross_product_expr : public vector_expression_base<cross_product_expr<Vec1, Vec2>, typename Vec1::value_type> {
public:
    using value_type = typename Vec1::value_type;
    using vec1_storage = typename expression_storage<Vec1>::type;
    using vec2_storage = typename expression_storage<Vec2>::type;
    
    cross_product_expr(const Vec1& a, const Vec2& b)
        : a_(a), b_(b) {}
    
    value_type eval_scalar(size_t idx) const {
        switch(idx) {
            case 0: return a_[1] * b_[2] - a_[2] * b_[1];
            case 1: return a_[2] * b_[0] - a_[0] * b_[2];
            case 2: return a_[0] * b_[1] - a_[1] * b_[0];
            default: return value_type(0);
        }
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        if (col == 0) {
            return eval_scalar(row);
        } else {
            return eval_scalar(col);
        }
    }
    
private:
    vec1_storage a_;
    vec2_storage b_;
};


// Reflection expression
template<typename Vec, typename Normal>
class reflect_expr : public vector_expression_base<reflect_expr<Vec, Normal>, typename Vec::value_type> {
public:
    using value_type = typename Vec::value_type;
    using vec_storage = typename expression_storage<Vec>::type;
    using normal_storage = typename expression_storage<Normal>::type;
    
    reflect_expr(const Vec& incident, const Normal& normal)
        : incident_(incident), normal_(normal) {
        // Compute 2 * dot(incident, normal) once
        two_dot_ = value_type(2) * dot(incident, normal);
    }
    
    value_type eval_scalar(size_t idx) const {
        return incident_[idx] - two_dot_ * normal_[idx];
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return incident_(row, col) - two_dot_ * normal_(row, col);
    }
    
private:
    vec_storage incident_;
    normal_storage normal_;
    value_type two_dot_;
};

// Linear interpolation expression
template<typename Vec1, typename Vec2, typename T>
class lerp_expr : public vector_expression_base<lerp_expr<Vec1, Vec2, T>, T> {
public:
    using value_type = T;
    using vec1_storage = typename expression_storage<Vec1>::type;
    using vec2_storage = typename expression_storage<Vec2>::type;
    
    lerp_expr(const Vec1& a, const Vec2& b, T t)
        : a_(a), b_(b), t_(t), one_minus_t_(T(1) - t) {}
    
    T eval_scalar(size_t idx) const {
        return one_minus_t_ * a_[idx] + t_ * b_[idx];
    }
    
    T eval_scalar(size_t row, size_t col) const {
        return one_minus_t_ * a_(row, col) + t_ * b_(row, col);
    }
    
private:
    vec1_storage a_;
    vec2_storage b_;
    T t_;
    T one_minus_t_;
};

// Component-wise min expression
template<typename Vec1, typename Vec2>
class min_expr : public vector_expression_base<min_expr<Vec1, Vec2>, typename Vec1::value_type> {
public:
    using value_type = typename Vec1::value_type;
    using vec1_storage = typename expression_storage<Vec1>::type;
    using vec2_storage = typename expression_storage<Vec2>::type;
    
    min_expr(const Vec1& a, const Vec2& b) : a_(a), b_(b) {}
    
    value_type eval_scalar(size_t idx) const {
        return std::min(a_[idx], b_[idx]);
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return std::min(a_(row, col), b_(row, col));
    }
    
private:
    vec1_storage a_;
    vec2_storage b_;
};

// Component-wise max expression
template<typename Vec1, typename Vec2>
class max_expr : public vector_expression_base<max_expr<Vec1, Vec2>, typename Vec1::value_type> {
public:
    using value_type = typename Vec1::value_type;
    using vec1_storage = typename expression_storage<Vec1>::type;
    using vec2_storage = typename expression_storage<Vec2>::type;
    
    max_expr(const Vec1& a, const Vec2& b) : a_(a), b_(b) {}
    
    value_type eval_scalar(size_t idx) const {
        return std::max(a_[idx], b_[idx]);
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return std::max(a_(row, col), b_(row, col));
    }
    
private:
    vec1_storage a_;
    vec2_storage b_;
};

// Component-wise abs expression
template<typename Vec>
class abs_expr : public vector_expression_base<abs_expr<Vec>, typename Vec::value_type> {
public:
    using value_type = typename Vec::value_type;
    using vec_storage = typename expression_storage<Vec>::type;
    
    abs_expr(const Vec& v) : v_(v) {}
    
    value_type eval_scalar(size_t idx) const {
        return std::abs(v_[idx]);
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return std::abs(v_(row, col));
    }
    
private:
    vec_storage v_;
};

// Clamp expression
template<typename Vec, typename Min, typename Max>
class clamp_expr : public vector_expression_base<clamp_expr<Vec, Min, Max>, typename Vec::value_type> {
public:
    using value_type = typename Vec::value_type;
    using vec_storage = typename expression_storage<Vec>::type;
    using min_storage = typename std::conditional<std::is_arithmetic_v<Min>, Min, const Min&>::type;
    using max_storage = typename std::conditional<std::is_arithmetic_v<Max>, Max, const Max&>::type;
    
    clamp_expr(const Vec& v, const Min& min_val, const Max& max_val)
        : v_(v), min_val_(min_val), max_val_(max_val) {}
    
    value_type eval_scalar(size_t idx) const {
        value_type val = v_[idx];
        value_type min_v = get_component(min_val_, idx);
        value_type max_v = get_component(max_val_, idx);
        return std::min(std::max(val, min_v), max_v);
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        value_type val = v_(row, col);
        value_type min_v = get_component(min_val_, row, col);
        value_type max_v = get_component(max_val_, row, col);
        return std::min(std::max(val, min_v), max_v);
    }
    
private:
    vec_storage v_;
    min_storage min_val_;
    max_storage max_val_;
    
    // Helper to get component from either scalar or vector
    template<typename T>
    value_type get_component(const T& val, size_t idx) const {
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<value_type>(val);
        } else {
            return val[idx];
        }
    }
    
    template<typename T>
    value_type get_component(const T& val, size_t row, size_t col) const {
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<value_type>(val);
        } else {
            return val(row, col);
        }
    }
};

// Project expression
template<typename Vec1, typename Vec2>
class project_expr : public vector_expression_base<project_expr<Vec1, Vec2>, typename Vec1::value_type> {
public:
    using value_type = typename Vec1::value_type;
    using vec1_storage = typename expression_storage<Vec1>::type;
    using vec2_storage = typename expression_storage<Vec2>::type;
    
    project_expr(const Vec1& a, const Vec2& b) : a_(a), b_(b) {
        // Precompute dot(a,b) / length_squared(b)
        value_type b_len_sq = dot(b, b);
        scale_ = dot(a, b) / b_len_sq;
    }
    
    value_type eval_scalar(size_t idx) const {
        return scale_ * b_[idx];
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return scale_ * b_(row, col);
    }
    
private:
    vec1_storage a_;
    vec2_storage b_;
    value_type scale_;
};

// Faceforward expression
template<typename Vec, typename Incident, typename Normal>
class faceforward_expr : public vector_expression_base<faceforward_expr<Vec, Incident, Normal>, typename Vec::value_type> {
public:
    using value_type = typename Vec::value_type;
    using vec_storage = typename expression_storage<Vec>::type;
    
    faceforward_expr(const Vec& n, const Incident& i, const Normal& nref) 
        : n_(n), sign_(dot(i, nref) < value_type(0) ? value_type(1) : value_type(-1)) {}
    
    value_type eval_scalar(size_t idx) const {
        return sign_ * n_[idx];
    }
    
    value_type eval_scalar(size_t row, size_t col) const {
        return sign_ * n_(row, col);
    }
    
private:
    vec_storage n_;
    value_type sign_;
};

// Store vector expression types by value
template<typename Vec1, typename Vec2>
struct expression_storage<cross_product_expr<Vec1, Vec2>> {
    using type = cross_product_expr<Vec1, Vec2>;
};

template<typename Expr, size_t Size>
struct expression_storage<normalized_expr<Expr, Size>> {
    using type = normalized_expr<Expr, Size>;
};

template<typename Vec, typename Normal>
struct expression_storage<reflect_expr<Vec, Normal>> {
    using type = reflect_expr<Vec, Normal>;
};

template<typename Vec1, typename Vec2, typename T>
struct expression_storage<lerp_expr<Vec1, Vec2, T>> {
    using type = lerp_expr<Vec1, Vec2, T>;
};

template<typename Vec1, typename Vec2>
struct expression_storage<min_expr<Vec1, Vec2>> {
    using type = min_expr<Vec1, Vec2>;
};

template<typename Vec1, typename Vec2>
struct expression_storage<max_expr<Vec1, Vec2>> {
    using type = max_expr<Vec1, Vec2>;
};

template<typename Vec>
struct expression_storage<abs_expr<Vec>> {
    using type = abs_expr<Vec>;
};

template<typename Vec, typename Min, typename Max>
struct expression_storage<clamp_expr<Vec, Min, Max>> {
    using type = clamp_expr<Vec, Min, Max>;
};

template<typename Vec1, typename Vec2>
struct expression_storage<project_expr<Vec1, Vec2>> {
    using type = project_expr<Vec1, Vec2>;
};

template<typename Vec, typename Incident, typename Normal>
struct expression_storage<faceforward_expr<Vec, Incident, Normal>> {
    using type = faceforward_expr<Vec, Incident, Normal>;
};

// Expression storage specializations have been moved to expression.hh

// Expression traits for vector expressions to make them compatible with matrix expressions
// These define vector expressions as Nx1 column vectors

template<typename Expr, size_t Size>
struct expression_traits<normalized_expr<Expr, Size>> {
    using value_type = typename normalized_expr<Expr, Size>::value_type;
    static constexpr size_t rows = Size;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec1, typename Vec2>
struct expression_traits<cross_product_expr<Vec1, Vec2>> {
    using value_type = typename cross_product_expr<Vec1, Vec2>::value_type;
    static constexpr size_t rows = 3;  // Cross product always returns 3D vector
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec, typename Normal>
struct expression_traits<reflect_expr<Vec, Normal>> {
    using value_type = typename reflect_expr<Vec, Normal>::value_type;
    static constexpr size_t rows = vector_size_v<Vec>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec1, typename Vec2, typename T>
struct expression_traits<lerp_expr<Vec1, Vec2, T>> {
    using value_type = T;
    static constexpr size_t rows = vector_size_v<Vec1>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec1, typename Vec2>
struct expression_traits<min_expr<Vec1, Vec2>> {
    using value_type = typename min_expr<Vec1, Vec2>::value_type;
    static constexpr size_t rows = vector_size_v<Vec1>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec1, typename Vec2>
struct expression_traits<max_expr<Vec1, Vec2>> {
    using value_type = typename max_expr<Vec1, Vec2>::value_type;
    static constexpr size_t rows = vector_size_v<Vec1>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec>
struct expression_traits<abs_expr<Vec>> {
    using value_type = typename abs_expr<Vec>::value_type;
    static constexpr size_t rows = vector_size_v<Vec>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec, typename Min, typename Max>
struct expression_traits<clamp_expr<Vec, Min, Max>> {
    using value_type = typename clamp_expr<Vec, Min, Max>::value_type;
    static constexpr size_t rows = vector_size_v<Vec>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec1, typename Vec2>
struct expression_traits<project_expr<Vec1, Vec2>> {
    using value_type = typename project_expr<Vec1, Vec2>::value_type;
    static constexpr size_t rows = vector_size_v<Vec1>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

template<typename Vec, typename Incident, typename Normal>
struct expression_traits<faceforward_expr<Vec, Incident, Normal>> {
    using value_type = typename faceforward_expr<Vec, Incident, Normal>::value_type;
    static constexpr size_t rows = vector_size_v<Vec>;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
};

// Custom unary/binary expressions
template<typename Expr, typename Op>
struct expression_traits<custom_unary_expr<Expr, Op>> {
    using value_type = typename custom_unary_expr<Expr, Op>::value_type;
    static constexpr size_t rows = expression_traits<Expr>::rows;
    static constexpr size_t cols = expression_traits<Expr>::cols;
    static constexpr bool row_major = expression_traits<Expr>::row_major;
};

template<typename Expr1, typename Expr2, typename Op>
struct expression_traits<custom_binary_expr<Expr1, Expr2, Op>> {
    using value_type = typename custom_binary_expr<Expr1, Expr2, Op>::value_type;
    static constexpr size_t rows = expression_traits<Expr1>::rows;
    static constexpr size_t cols = expression_traits<Expr1>::cols;
    static constexpr bool row_major = expression_traits<Expr1>::row_major;
};

} // namespace euler