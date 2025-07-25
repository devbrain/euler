#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <type_traits>
#include <utility>

namespace euler {

// Forward declarations
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor> class matrix;

// CRTP base class for expression templates
template<typename derived, typename T>
class expression {
public:
    using value_type = T;
    
    // Access the derived type
    const derived& self() const { 
        return static_cast<const derived&>(*this); 
    }
    
    derived& self() { 
        return static_cast<derived&>(*this); 
    }
    
    // Size information (must be provided by derived class)
    static constexpr size_t size() { 
        return derived::size; 
    }
    
    // Element access (delegates to derived class)
    T operator[](size_t idx) const {
        return self().eval_scalar(idx);
    }
    
    // For 2D access (matrices)
    T operator()(size_t i, size_t j) const {
        return self().eval_scalar(i, j);
    }
};

// Forward declarations
template<typename T>
class scalar_expression;

template<typename L, typename R, typename Op>
class binary_expression;

template<typename E, typename Op>
class unary_expression;

// Type trait to determine if we should store by value or reference
template<typename T>
struct expression_storage {
    // By default, store lvalue references as const references
    using type = const T&;
};

// Store scalar expressions by value
template<typename T>
struct expression_storage<scalar_expression<T>> {
    using type = scalar_expression<T>;
};

// Store rvalue binary expressions by value
template<typename L, typename R, typename Op>
struct expression_storage<binary_expression<L, R, Op>> {
    using type = binary_expression<L, R, Op>;
};

// Store rvalue unary expressions by value  
template<typename E, typename Op>
struct expression_storage<unary_expression<E, Op>> {
    using type = unary_expression<E, Op>;
};

// Forward declare custom expression types from vector_expr.hh
template<typename Expr, typename Op> class custom_unary_expr;
template<typename Expr1, typename Expr2, typename Op> class custom_binary_expr;

// Store custom unary expressions by value
template<typename Expr, typename Op>
struct expression_storage<custom_unary_expr<Expr, Op>> {
    using type = custom_unary_expr<Expr, Op>;
};

// Store custom binary expressions by value
template<typename Expr1, typename Expr2, typename Op>
struct expression_storage<custom_binary_expr<Expr1, Expr2, Op>> {
    using type = custom_binary_expr<Expr1, Expr2, Op>;
};

// Forward declaration for matrix and vector
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;
template<typename T, size_t N> class vector;

// Storage specializations for concrete matrix and vector types
// These store by value to handle temporaries from functions like sqrt(v)
template<typename T, size_t M, size_t N, bool RowMajor>
struct expression_storage<matrix<T, M, N, RowMajor>> {
    using type = matrix<T, M, N, RowMajor>;
};

template<typename T, size_t N>
struct expression_storage<vector<T, N>> {
    using type = vector<T, N>;
};

// But store matrix/vector references as references (for lazy evaluation)
template<typename T, size_t M, size_t N, bool RowMajor>
struct expression_storage<matrix<T, M, N, RowMajor>&> {
    using type = const matrix<T, M, N, RowMajor>&;
};

template<typename T, size_t N>
struct expression_storage<vector<T, N>&> {
    using type = const vector<T, N>&;
};

template<typename T, size_t M, size_t N, bool RowMajor>
struct expression_storage<const matrix<T, M, N, RowMajor>&> {
    using type = const matrix<T, M, N, RowMajor>&;
};

template<typename T, size_t N>
struct expression_storage<const vector<T, N>&> {
    using type = const vector<T, N>&;
};


// Binary expression template
template<typename left_expr, typename right_expr, typename op>
class binary_expression : public expression<binary_expression<left_expr, right_expr, op>, 
                                          typename left_expr::value_type> {
public:
    using value_type = typename left_expr::value_type;
    using left_storage = typename expression_storage<left_expr>::type;
    using right_storage = typename expression_storage<right_expr>::type;
    static constexpr size_t size = left_expr::size > 0 ? left_expr::size : right_expr::size;
    
    binary_expression(const left_expr& l, const right_expr& r) 
        : left(l), right(r) {}
    
    value_type eval_scalar(size_t idx) const {
        return op::apply(left[idx], right[idx]);
    }
    
    value_type eval_scalar(size_t i, size_t j) const {
        return op::apply(left(i, j), right(i, j));
    }
    
private:
    left_storage left;
    right_storage right;
};

// Unary expression template
template<typename expr, typename op>
class unary_expression : public expression<unary_expression<expr, op>, 
                                         typename expr::value_type> {
public:
    using value_type = typename expr::value_type;
    using expr_storage = typename expression_storage<expr>::type;
    static constexpr size_t size = expr::size;
    
    unary_expression(const expr& operand_expr) : operand(operand_expr) {}
    
    value_type eval_scalar(size_t idx) const {
        return op::apply(operand[idx]);
    }
    
    value_type eval_scalar(size_t i, size_t j) const {
        return op::apply(operand(i, j));
    }
    
private:
    expr_storage operand;
};

// Scalar expression (broadcasts a scalar to all elements)
template<typename T>
class scalar_expression : public expression<scalar_expression<T>, T> {
public:
    using value_type = T;
    static constexpr size_t size = 0; // Dynamic size - means scalar
    
    scalar_expression(T value) : val(value) {}
    
    value_type eval_scalar(size_t) const {
        return val;
    }
    
    value_type eval_scalar(size_t, size_t) const {
        return val;
    }
    
    // Override operator[] to return scalar value regardless of index
    value_type operator[](size_t) const {
        return val;
    }
    
    value_type operator()(size_t, size_t) const {
        return val;
    }
    
private:
    T val;
};

// Operation functors
namespace ops {
    
    struct plus {
        template<typename T>
        static T apply(T a, T b) { return a + b; }
    };
    
    struct minus {
        template<typename T>
        static T apply(T a, T b) { return a - b; }
    };
    
    struct multiplies {
        template<typename T>
        static T apply(T a, T b) { return a * b; }
    };
    
    struct divides {
        template<typename T>
        static T apply(T a, T b) { return a / b; }
    };
    
    struct negate {
        template<typename T>
        static T apply(T a) { return -a; }
    };
    
} // namespace ops

// Expression builder functions
template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator+(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return binary_expression<E1, E2, ops::plus>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator-(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return binary_expression<E1, E2, ops::minus>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator*(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return binary_expression<E1, E2, ops::multiplies>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator/(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return binary_expression<E1, E2, ops::divides>(lhs.self(), rhs.self());
}

// Scalar operations
template<typename E, typename T = typename E::value_type>
auto operator+(const expression<E, T>& lhs, T rhs) {
    return binary_expression<E, scalar_expression<T>, ops::plus>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator+(T lhs, const expression<E, T>& rhs) {
    return binary_expression<scalar_expression<T>, E, ops::plus>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator-(const expression<E, T>& lhs, T rhs) {
    return binary_expression<E, scalar_expression<T>, ops::minus>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator-(T lhs, const expression<E, T>& rhs) {
    return binary_expression<scalar_expression<T>, E, ops::minus>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator*(const expression<E, T>& lhs, T rhs) {
    return binary_expression<E, scalar_expression<T>, ops::multiplies>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator*(T lhs, const expression<E, T>& rhs) {
    return binary_expression<scalar_expression<T>, E, ops::multiplies>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator/(const expression<E, T>& lhs, T rhs) {
    return binary_expression<E, scalar_expression<T>, ops::divides>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator/(T lhs, const expression<E, T>& rhs) {
    return binary_expression<scalar_expression<T>, E, ops::divides>(scalar_expression<T>(lhs), rhs.self());
}

// Unary operations
template<typename E, typename T = typename E::value_type>
auto operator-(const expression<E, T>& expr) {
    return unary_expression<E, ops::negate>(expr.self());
}

} // namespace euler