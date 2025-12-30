#pragma once

#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/core/temp_value.hh>
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
    const derived& self() const & {
        return static_cast<const derived&>(*this);
    }

    derived& self() & {
        return static_cast<derived&>(*this);
    }

    derived&& self() && {
        return static_cast<derived&&>(*this);
    }

    const derived&& self() const && {
        return static_cast<const derived&&>(*this);
    }
    
    // Size information (must be provided by derived class)
    static constexpr size_t size() {
        return derived::static_size;
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

// Forward declare custom expression types from vector_expr.hh
template<typename Expr, typename Op> class custom_unary_expr;
template<typename Expr1, typename Expr2, typename Op> class custom_binary_expr;

// Forward declaration for matrix and vector
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;
template<typename T, size_t N> class vector;

// =============================================================================
// Storage type selection based on is_leaf trait
// =============================================================================
// Leaf types (concrete vector/matrix) are stored by reference for efficiency.
// Non-leaf types (expressions, temp_holders) are stored by value to prevent
// dangling references.
// =============================================================================

namespace detail {

// Storage type selector: leaf types by reference, non-leaf by value
template<typename T, bool StoreByRef>
struct storage_type_impl;

template<typename T>
struct storage_type_impl<T, true> {
    using type = const T&;
};

template<typename T>
struct storage_type_impl<T, false> {
    using type = T;
};

// Determine storage type based on is_leaf trait
template<typename T>
using storage_type = typename storage_type_impl<std::decay_t<T>, is_leaf_v<T>>::type;

} // namespace detail

// =============================================================================
// Operand capture utilities for rvalue detection
// =============================================================================
// These functions detect when a temporary (rvalue) basic math object is passed
// and wrap it in a temp_holder to prevent dangling references.
// =============================================================================

namespace detail {

// Capture an operand, wrapping rvalue leaf types in temp_holder
// Uses decltype(auto) to preserve reference types for lvalues
template<typename T>
decltype(auto) capture_operand(T&& operand) {
    using DecayedT = std::decay_t<T>;
    constexpr bool is_rvalue = std::is_rvalue_reference_v<T&&>;
    constexpr bool is_leaf_type = is_leaf_v<DecayedT>;

    if constexpr (is_rvalue && is_leaf_type) {
        // Rvalue leaf (vector/matrix): wrap in temp_holder to own the temporary
        return temp_holder_t<DecayedT>(std::forward<T>(operand));
    } else {
        // Lvalue leaf: stored by reference (safe, outlives expression)
        // Expression: stored by value (lightweight wrapper)
        return std::forward<T>(operand);
    }
}

// Type alias for the result of capture_operand
template<typename T>
using captured_t = decltype(capture_operand(std::declval<T>()));

} // namespace detail

// =============================================================================
// Backward-compatible expression_storage trait
// =============================================================================
// This provides the same interface as the old expression_storage but uses
// the new is_leaf-based storage_type internally.
// =============================================================================

template<typename T>
struct expression_storage {
    using type = detail::storage_type<T>;
};

// Binary expression template
template<typename left_expr, typename right_expr, typename op>
class binary_expression : public expression<binary_expression<left_expr, right_expr, op>,
                                          typename std::decay_t<left_expr>::value_type> {
public:
    using value_type = typename std::decay_t<left_expr>::value_type;
    using left_storage = detail::storage_type<left_expr>;
    using right_storage = detail::storage_type<right_expr>;
    static constexpr size_t static_size = std::decay_t<left_expr>::static_size > 0
        ? std::decay_t<left_expr>::static_size
        : std::decay_t<right_expr>::static_size;

    binary_expression(const std::decay_t<left_expr>& l, const std::decay_t<right_expr>& r)
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
                                         typename std::decay_t<expr>::value_type> {
public:
    using value_type = typename std::decay_t<expr>::value_type;
    using expr_storage = detail::storage_type<expr>;
    static constexpr size_t static_size = std::decay_t<expr>::static_size;

    unary_expression(const std::decay_t<expr>& operand_expr) : operand(operand_expr) {}

    value_type eval_scalar(size_t idx) const {
        return op::apply(operand[idx]);
    }

    value_type eval_scalar(size_t i, size_t j) const {
        return op::apply(operand(i, j));
    }

private:
    expr_storage operand;
};

// =============================================================================
// Macro for generating expression_traits specializations
// =============================================================================
// Use this macro to avoid boilerplate when defining expression_traits for
// expression types that have value_type, rows, cols, and row_major members.
// =============================================================================

// Helper macro to allow commas in macro arguments
#define EULER_COMMA ,

#define EULER_DEFINE_EXPR_TRAITS(TEMPLATE_PARAMS, EXPR_TYPE)                     \
template<TEMPLATE_PARAMS>                                                         \
struct expression_traits<EXPR_TYPE> {                                            \
    using value_type = typename EXPR_TYPE::value_type;                           \
    static constexpr size_t rows = EXPR_TYPE::rows;                              \
    static constexpr size_t cols = EXPR_TYPE::cols;                              \
    static constexpr bool row_major = EXPR_TYPE::row_major;                      \
}

// Scalar expression (broadcasts a scalar to all elements)
template<typename T>
class scalar_expression : public expression<scalar_expression<T>, T> {
public:
    using value_type = T;
    static constexpr size_t static_size = 0; // Dynamic size - means scalar
    
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

// =============================================================================
// Factory functions for creating expressions with proper operand capture
// =============================================================================

namespace detail {

// Factory for binary expressions - handles rvalue detection and wrapping
template<typename Op, typename LHS, typename RHS>
auto make_binary_expr(LHS&& lhs, RHS&& rhs) {
    // Use decltype(auto) to preserve references when capture_operand returns a reference
    decltype(auto) captured_lhs = capture_operand(std::forward<LHS>(lhs));
    decltype(auto) captured_rhs = capture_operand(std::forward<RHS>(rhs));
    // Use std::decay_t to get the actual type (stripping references for the template parameter)
    using CapturedLHS = std::decay_t<decltype(captured_lhs)>;
    using CapturedRHS = std::decay_t<decltype(captured_rhs)>;
    return binary_expression<CapturedLHS, CapturedRHS, Op>(captured_lhs, captured_rhs);
}

// Factory for unary expressions - handles rvalue detection and wrapping
template<typename Op, typename Expr>
auto make_unary_expr(Expr&& expr) {
    decltype(auto) captured = capture_operand(std::forward<Expr>(expr));
    using CapturedExpr = std::decay_t<decltype(captured)>;
    return unary_expression<CapturedExpr, Op>(captured);
}

} // namespace detail

// =============================================================================
// Expression builder operators
// =============================================================================

// Binary operations between expressions
template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator+(expression<E1, T>&& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::plus>(
        std::move(lhs).self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator+(expression<E1, T>&& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::plus>(
        std::move(lhs).self(),
        rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator+(const expression<E1, T>& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::plus>(
        lhs.self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator+(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::plus>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator-(expression<E1, T>&& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::minus>(
        std::move(lhs).self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator-(expression<E1, T>&& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::minus>(
        std::move(lhs).self(),
        rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator-(const expression<E1, T>& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::minus>(
        lhs.self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator-(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::minus>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator*(expression<E1, T>&& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::multiplies>(
        std::move(lhs).self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator*(expression<E1, T>&& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::multiplies>(
        std::move(lhs).self(),
        rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator*(const expression<E1, T>& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::multiplies>(
        lhs.self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator*(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::multiplies>(lhs.self(), rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator/(expression<E1, T>&& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::divides>(
        std::move(lhs).self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator/(expression<E1, T>&& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::divides>(
        std::move(lhs).self(),
        rhs.self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator/(const expression<E1, T>& lhs, expression<E2, T>&& rhs) {
    return detail::make_binary_expr<ops::divides>(
        lhs.self(),
        std::move(rhs).self());
}

template<typename E1, typename E2, typename T = typename E1::value_type>
auto operator/(const expression<E1, T>& lhs, const expression<E2, T>& rhs) {
    return detail::make_binary_expr<ops::divides>(lhs.self(), rhs.self());
}

// Scalar operations - lvalue expression
template<typename E, typename T = typename E::value_type>
auto operator+(const expression<E, T>& lhs, T rhs) {
    return detail::make_binary_expr<ops::plus>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator+(T lhs, const expression<E, T>& rhs) {
    return detail::make_binary_expr<ops::plus>(scalar_expression<T>(lhs), rhs.self());
}

// Scalar operations - rvalue expression
template<typename E, typename T = typename E::value_type>
auto operator+(expression<E, T>&& lhs, T rhs) {
    return detail::make_binary_expr<ops::plus>(std::move(lhs).self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator+(T lhs, expression<E, T>&& rhs) {
    return detail::make_binary_expr<ops::plus>(scalar_expression<T>(lhs), std::move(rhs).self());
}

template<typename E, typename T = typename E::value_type>
auto operator-(const expression<E, T>& lhs, T rhs) {
    return detail::make_binary_expr<ops::minus>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator-(T lhs, const expression<E, T>& rhs) {
    return detail::make_binary_expr<ops::minus>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator-(expression<E, T>&& lhs, T rhs) {
    return detail::make_binary_expr<ops::minus>(std::move(lhs).self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator-(T lhs, expression<E, T>&& rhs) {
    return detail::make_binary_expr<ops::minus>(scalar_expression<T>(lhs), std::move(rhs).self());
}

template<typename E, typename T = typename E::value_type>
auto operator*(const expression<E, T>& lhs, T rhs) {
    return detail::make_binary_expr<ops::multiplies>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator*(T lhs, const expression<E, T>& rhs) {
    return detail::make_binary_expr<ops::multiplies>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator*(expression<E, T>&& lhs, T rhs) {
    return detail::make_binary_expr<ops::multiplies>(std::move(lhs).self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator*(T lhs, expression<E, T>&& rhs) {
    return detail::make_binary_expr<ops::multiplies>(scalar_expression<T>(lhs), std::move(rhs).self());
}

template<typename E, typename T = typename E::value_type>
auto operator/(const expression<E, T>& lhs, T rhs) {
    return detail::make_binary_expr<ops::divides>(lhs.self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator/(T lhs, const expression<E, T>& rhs) {
    return detail::make_binary_expr<ops::divides>(scalar_expression<T>(lhs), rhs.self());
}

template<typename E, typename T = typename E::value_type>
auto operator/(expression<E, T>&& lhs, T rhs) {
    return detail::make_binary_expr<ops::divides>(std::move(lhs).self(), scalar_expression<T>(rhs));
}

template<typename E, typename T = typename E::value_type>
auto operator/(T lhs, expression<E, T>&& rhs) {
    return detail::make_binary_expr<ops::divides>(scalar_expression<T>(lhs), std::move(rhs).self());
}

// Unary operations
template<typename E, typename T = typename E::value_type>
auto operator-(const expression<E, T>& expr) {
    return detail::make_unary_expr<ops::negate>(expr.self());
}

template<typename E, typename T = typename E::value_type>
auto operator-(expression<E, T>&& expr) {
    return detail::make_unary_expr<ops::negate>(std::move(expr).self());
}

} // namespace euler