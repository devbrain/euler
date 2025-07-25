#pragma once

#include <euler/core/types.hh>
#include <type_traits>

namespace euler {

// Forward declarations
template<typename derived, typename T> class expression;
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor> class matrix;
template<typename T, size_t N> class vector;
template<typename T, size_t N> class column_vector;
template<typename T, size_t N> class row_vector;

// Type trait to check if a type is an expression
template<typename T>
struct is_expression : std::false_type {};

template<typename D, typename T>
struct is_expression<expression<D, T>> : std::true_type {};

// Helper to detect if a type derives from expression using SFINAE
template<typename T>
struct is_expression_derived {
    // Check for vector-like expressions (1D)
    template<typename U>
    static auto test_1d(int) -> decltype(
        std::declval<const U&>().self(),
        std::declval<const U&>().eval_scalar(size_t{}),
        std::true_type{}
    );
    
    // Check for matrix-like expressions (2D)
    template<typename U>
    static auto test_2d(int) -> decltype(
        std::declval<const U&>().self(),
        std::declval<const U&>().eval_scalar(size_t{}, size_t{}),
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test_1d(...);
    
    template<typename>
    static std::false_type test_2d(...);
    
    static constexpr bool value = decltype(test_1d<T>(0))::value || decltype(test_2d<T>(0))::value;
};

// Combined check: either exact match or derived from expression
template<typename T>
constexpr bool is_expression_v = is_expression<T>::value || is_expression_derived<T>::value;

// Type trait to check if a type is a matrix
template<typename T>
struct is_matrix : std::false_type {};

template<typename T, size_t R, size_t C, bool ColumnMajor>
struct is_matrix<matrix<T, R, C, ColumnMajor>> : std::true_type {};

// Vector types are also matrices
template<typename T, size_t N>
struct is_matrix<vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_matrix<column_vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_matrix<row_vector<T, N>> : std::true_type {};

template<typename T>
constexpr bool is_matrix_v = is_matrix<T>::value;

// Get dimensions of a matrix type
template<typename T>
struct matrix_traits {
    static constexpr bool is_matrix = false;
    static constexpr size_t rows = 0;
    static constexpr size_t cols = 0;
    using value_type = void;
};

template<typename T, size_t R, size_t C, bool ColumnMajor>
struct matrix_traits<matrix<T, R, C, ColumnMajor>> {
    static constexpr bool is_matrix = true;
    static constexpr size_t rows = R;
    static constexpr size_t cols = C;
    static constexpr bool column_major = ColumnMajor;
    using value_type = T;
};

// Specialization for vector
template<typename T, size_t N>
struct matrix_traits<vector<T, N>> {
    static constexpr bool is_matrix = true;
    #ifndef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
        static constexpr size_t rows = N;
        static constexpr size_t cols = 1;
        static constexpr bool column_major = true;
    #else
        static constexpr size_t rows = 1;
        static constexpr size_t cols = N;
        static constexpr bool column_major = false;
    #endif
    using value_type = T;
};

// Specialization for column_vector
template<typename T, size_t N>
struct matrix_traits<column_vector<T, N>> {
    static constexpr bool is_matrix = true;
    static constexpr size_t rows = N;
    static constexpr size_t cols = 1;
    static constexpr bool column_major = true;
    using value_type = T;
};

// Specialization for row_vector
template<typename T, size_t N>
struct matrix_traits<row_vector<T, N>> {
    static constexpr bool is_matrix = true;
    static constexpr size_t rows = 1;
    static constexpr size_t cols = N;
    static constexpr bool column_major = false;
    using value_type = T;
};

// Check if a matrix is a vector (Nx1 or 1xN)
template<typename T>
struct is_vector {
    static constexpr bool value = matrix_traits<T>::is_matrix && 
        (matrix_traits<T>::rows == 1 || matrix_traits<T>::cols == 1);
};

template<typename T>
constexpr bool is_vector_v = is_vector<T>::value;

// Check if a matrix is square
template<typename T>
struct is_square_matrix {
    static constexpr bool value = matrix_traits<T>::is_matrix && 
        matrix_traits<T>::rows == matrix_traits<T>::cols;
};

template<typename T>
constexpr bool is_square_matrix_v = is_square_matrix<T>::value;

// Get the size of a vector (total elements)
template<typename T>
struct vector_size {
    static constexpr size_t value = matrix_traits<T>::rows * matrix_traits<T>::cols;
};

template<typename T>
constexpr size_t vector_size_v = vector_size<T>::value;

// Check if two matrix types have the same dimensions
template<typename T1, typename T2>
struct have_same_dimensions {
    static constexpr bool value = 
        matrix_traits<T1>::rows == matrix_traits<T2>::rows &&
        matrix_traits<T1>::cols == matrix_traits<T2>::cols;
};

template<typename T1, typename T2>
constexpr bool have_same_dimensions_v = have_same_dimensions<T1, T2>::value;

// Check if matrices can be multiplied (cols of first == rows of second)
template<typename T1, typename T2>
struct can_multiply {
    static constexpr bool value = 
        matrix_traits<T1>::is_matrix && 
        matrix_traits<T2>::is_matrix &&
        matrix_traits<T1>::cols == matrix_traits<T2>::rows;
};

template<typename T1, typename T2>
constexpr bool can_multiply_v = can_multiply<T1, T2>::value;

// Result type of matrix multiplication
template<typename T1, typename T2>
struct multiplication_result {
    using type = matrix<
        typename matrix_traits<T1>::value_type,
        matrix_traits<T1>::rows,
        matrix_traits<T2>::cols,
        matrix_traits<T1>::column_major  // Use the layout of the first matrix
    >;
};

template<typename T1, typename T2>
using multiplication_result_t = typename multiplication_result<T1, T2>::type;

// SFINAE helpers for C++17
template<typename T>
using enable_if_matrix_t = std::enable_if_t<is_matrix_v<T>>;

template<typename T>
using enable_if_vector_t = std::enable_if_t<is_vector_v<T>>;

template<typename T>
using enable_if_square_matrix_t = std::enable_if_t<is_square_matrix_v<T>>;

template<typename T>
using enable_if_floating_point_t = std::enable_if_t<is_floating_point_v<T>>;

// Common type promotion
template<typename T1, typename T2>
using common_type_t = std::common_type_t<T1, T2>;

// Result type for expressions
template<typename E1, typename E2>
struct expression_result {
    using type = common_type_t<
        typename E1::value_type,
        typename E2::value_type
    >;
};

template<typename E1, typename E2>
using expression_result_t = typename expression_result<E1, E2>::type;

// Expression traits for matrix expressions
template<typename T>
struct expression_traits {
    using value_type = void;
    static constexpr size_t rows = 0;
    static constexpr size_t cols = 0;
    static constexpr bool row_major = true;
};

// Specialization for matrix
template<typename T, size_t R, size_t C, bool RowMajor>
struct expression_traits<matrix<T, R, C, RowMajor>> {
    using value_type = T;
    static constexpr size_t rows = R;
    static constexpr size_t cols = C;
    static constexpr bool row_major = RowMajor;
};

// Specialization for vector (inherits matrix layout)
template<typename T, size_t N>
struct expression_traits<vector<T, N>> {
    using value_type = T;
    #ifndef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
        static constexpr size_t rows = N;
        static constexpr size_t cols = 1;
    #else
        static constexpr size_t rows = 1;
        static constexpr size_t cols = N;
    #endif
    static constexpr bool row_major = 
        #ifdef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
            true
        #else
            false
        #endif
        ;
};

// Specialization for column_vector
template<typename T, size_t N>
struct expression_traits<column_vector<T, N>> {
    using value_type = T;
    static constexpr size_t rows = N;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = false;  // Column vectors are always column-major
};

// Specialization for row_vector
template<typename T, size_t N>
struct expression_traits<row_vector<T, N>> {
    using value_type = T;
    static constexpr size_t rows = 1;
    static constexpr size_t cols = N;
    static constexpr bool row_major = true;  // Row vectors are always row-major
};

// Forward declarations for expression types
template<typename L, typename R, typename Op> class binary_expression;
template<typename E, typename Op> class unary_expression;
template<typename T> class scalar_expression;

// Expression traits for generic binary expressions
template<typename L, typename R, typename Op>
struct expression_traits<binary_expression<L, R, Op>> {
    using value_type = typename binary_expression<L, R, Op>::value_type;
    static constexpr size_t rows = expression_traits<L>::rows;
    static constexpr size_t cols = expression_traits<L>::cols;
    static constexpr bool row_major = expression_traits<L>::row_major;
};

// Expression traits for generic unary expressions
template<typename E, typename Op>
struct expression_traits<unary_expression<E, Op>> {
    using value_type = typename unary_expression<E, Op>::value_type;
    static constexpr size_t rows = expression_traits<E>::rows;
    static constexpr size_t cols = expression_traits<E>::cols;
    static constexpr bool row_major = expression_traits<E>::row_major;
};

// Expression traits for scalar expressions
template<typename T>
struct expression_traits<scalar_expression<T>> {
    using value_type = T;
    static constexpr size_t rows = 0;  // Dynamic size
    static constexpr size_t cols = 0;  // Dynamic size
    static constexpr bool row_major = true;
};

// Fix expression traits for binary expressions with scalar broadcasting
// When left operand is scalar (0x0), take dimensions from right operand
template<typename R, typename Op>
struct expression_traits<binary_expression<scalar_expression<typename R::value_type>, R, Op>> {
    using value_type = typename R::value_type;
    static constexpr size_t rows = expression_traits<R>::rows;
    static constexpr size_t cols = expression_traits<R>::cols;
    static constexpr bool row_major = expression_traits<R>::row_major;
};

// Check if a type is a matrix expression (either a matrix or derives from expression)
template<typename T>
struct is_matrix_expression {
    static constexpr bool value = is_matrix_v<T> || is_expression_v<T>;
};

template<typename T>
constexpr bool is_matrix_expression_v = is_matrix_expression<T>::value;

// Helper to check if all types in a parameter pack are the same
template<typename T, typename... Ts>
struct are_same : std::conjunction<std::is_same<T, Ts>...> {};

template<typename T, typename... Ts>
constexpr bool are_same_v = are_same<T, Ts...>::value;

// Helper to check if a type is convertible to another
template<typename From, typename To>
constexpr bool is_convertible_v = std::is_convertible_v<From, To>;

// Storage order enum
enum class storage_order {
    column_major,  // OpenGL style
    row_major      // DirectX style
};

// Default storage order
constexpr storage_order default_storage_order = storage_order::column_major;

} // namespace euler