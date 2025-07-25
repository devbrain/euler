#pragma once

#include <euler/core/traits.hh>
#include <type_traits>

namespace euler {

// Forward declarations
template<typename T, size_t N> class vector;
template<typename T, size_t N> class column_vector;
template<typename T, size_t N> class row_vector;
template<typename T, size_t R, size_t C, bool RowMajor> class matrix;
template<typename T> class matrix_view;
template<typename T> class const_matrix_view;

// Expression templates forward declarations
template<typename Derived, typename T> class expression;
template<typename Expr1, typename Expr2, typename Op> class binary_expression;
template<typename Expr, typename Op> class unary_expression;
template<typename Vec1, typename Vec2> class cross_product_expr;
template<typename Expr, size_t Size> class normalized_expr;
template<typename Expr1, typename Expr2, typename Op> class matrix_binary_expression;
template<typename Expr> class length_expr;
template<typename Expr> class length_squared_expr;
template<typename T> class scalar_expression;
template<typename Vec, typename Normal> class reflect_expr;
template<typename Vec1, typename Vec2, typename T> class lerp_expr;
template<typename Vec1, typename Vec2> class min_expr;
template<typename Vec1, typename Vec2> class max_expr;
template<typename Vec> class abs_expr;
template<typename Vec, typename Min, typename Max> class clamp_expr;
template<typename Vec1, typename Vec2> class project_expr;
template<typename Vec, typename Incident, typename Normal> class faceforward_expr;
template<typename Expr, typename T> class scalar_vector_multiply_expr;

// Helper to get vector dimension from either row or column vector
template<typename T>
struct vector_size_helper {
    static constexpr size_t value = 0;
};

template<typename T, size_t N>
struct vector_size_helper<matrix<T, N, 1>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_size_helper<matrix<T, 1, N>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_size_helper<vector<T, N>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_size_helper<column_vector<T, N>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_size_helper<row_vector<T, N>> {
    static constexpr size_t value = N;
};

// Helper trait to get vector dimension (including expressions)
template<typename T, typename = void>
struct get_vector_dimension {
    static constexpr size_t value = 0;
};

// Specialization for concrete vector types
template<typename T>
struct get_vector_dimension<T, std::enable_if_t<(vector_size_helper<T>::value > 0)>> {
    static constexpr size_t value = vector_size_helper<T>::value;
};

// Specialization for matrix view
template<typename T>
struct get_vector_dimension<matrix_view<T>> {
    static constexpr size_t value = 0;  // Dynamic size
};

template<typename T>
struct get_vector_dimension<const_matrix_view<T>> {
    static constexpr size_t value = 0;  // Dynamic size
};

// Specializations for known expression types
template<typename Vec1, typename Vec2>
struct get_vector_dimension<cross_product_expr<Vec1, Vec2>> {
    static constexpr size_t value = 3;  // Cross product always returns 3D
};

template<typename Expr, size_t Size>
struct get_vector_dimension<normalized_expr<Expr, Size>> {
    static constexpr size_t value = Size;
};

template<typename Expr>
struct get_vector_dimension<length_expr<Expr>> {
    static constexpr size_t value = 1;  // Length is a scalar
};

template<typename Expr>
struct get_vector_dimension<length_squared_expr<Expr>> {
    static constexpr size_t value = 1;  // Length squared is a scalar
};

// For binary expressions, use the dimension of the first operand unless it's a scalar
template<typename Expr1, typename Expr2, typename Op>
struct get_vector_dimension<binary_expression<Expr1, Expr2, Op>> {
    static constexpr size_t first_dim = get_vector_dimension<Expr1>::value;
    static constexpr size_t second_dim = get_vector_dimension<Expr2>::value;
    static constexpr size_t value = (first_dim > 0) ? first_dim : second_dim;
};

// For unary expressions, use the dimension of the operand
template<typename Expr, typename Op>
struct get_vector_dimension<unary_expression<Expr, Op>> {
    static constexpr size_t value = get_vector_dimension<Expr>::value;
};

template<typename Vec, typename Normal>
struct get_vector_dimension<reflect_expr<Vec, Normal>> {
    static constexpr size_t value = get_vector_dimension<Vec>::value;
};

template<typename Vec1, typename Vec2, typename T>
struct get_vector_dimension<lerp_expr<Vec1, Vec2, T>> {
    static constexpr size_t value = get_vector_dimension<Vec1>::value;
};

template<typename Vec1, typename Vec2>
struct get_vector_dimension<min_expr<Vec1, Vec2>> {
    static constexpr size_t value = get_vector_dimension<Vec1>::value;
};

template<typename Vec1, typename Vec2>
struct get_vector_dimension<max_expr<Vec1, Vec2>> {
    static constexpr size_t value = get_vector_dimension<Vec1>::value;
};

template<typename Vec>
struct get_vector_dimension<abs_expr<Vec>> {
    static constexpr size_t value = get_vector_dimension<Vec>::value;
};

template<typename Vec, typename Min, typename Max>
struct get_vector_dimension<clamp_expr<Vec, Min, Max>> {
    static constexpr size_t value = get_vector_dimension<Vec>::value;
};

template<typename Vec1, typename Vec2>
struct get_vector_dimension<project_expr<Vec1, Vec2>> {
    static constexpr size_t value = get_vector_dimension<Vec1>::value;
};

template<typename Vec, typename Incident, typename Normal>
struct get_vector_dimension<faceforward_expr<Vec, Incident, Normal>> {
    static constexpr size_t value = get_vector_dimension<Vec>::value;
};

// Scalar expression has no dimension
template<typename T>
struct get_vector_dimension<scalar_expression<T>> {
    static constexpr size_t value = 0;
};

// Scalar-vector multiplication inherits dimension from the vector
template<typename Expr, typename T>
struct get_vector_dimension<scalar_vector_multiply_expr<Expr, T>> {
    static constexpr size_t value = get_vector_dimension<Expr>::value;
};

// Matrix binary expressions (for vector-like matrix operations)
template<typename Expr1, typename Expr2, typename Op>
struct get_vector_dimension<matrix_binary_expression<Expr1, Expr2, Op>> {
    // If both are vectors (Nx1 or 1xN), use their dimension
    static constexpr bool is_first_vector = (expression_traits<Expr1>::rows == 1 && expression_traits<Expr1>::cols > 1) ||
                                          (expression_traits<Expr1>::rows > 1 && expression_traits<Expr1>::cols == 1);
    static constexpr bool is_second_vector = (expression_traits<Expr2>::rows == 1 && expression_traits<Expr2>::cols > 1) ||
                                           (expression_traits<Expr2>::rows > 1 && expression_traits<Expr2>::cols == 1);
    
    static constexpr size_t first_size = is_first_vector ? 
        (expression_traits<Expr1>::rows == 1 ? expression_traits<Expr1>::cols : expression_traits<Expr1>::rows) : 0;
    static constexpr size_t second_size = is_second_vector ? 
        (expression_traits<Expr2>::rows == 1 ? expression_traits<Expr2>::cols : expression_traits<Expr2>::rows) : 0;
    
    static constexpr size_t value = (first_size > 0) ? first_size : second_size;
};

// Convenience variable template
template<typename T>
constexpr size_t expression_vector_size_v = get_vector_dimension<T>::value;

// Helper to check if type is any kind of vector (row or column)
template<typename T>
struct is_any_vector : std::false_type {};

template<typename T, size_t N>
struct is_any_vector<matrix<T, N, 1>> : std::true_type {};

template<typename T, size_t N>
struct is_any_vector<matrix<T, 1, N>> : std::true_type {};

template<typename T, size_t N>
struct is_any_vector<vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_any_vector<column_vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_any_vector<row_vector<T, N>> : std::true_type {};

template<typename T>
constexpr bool is_any_vector_v = is_any_vector<T>::value;

} // namespace euler