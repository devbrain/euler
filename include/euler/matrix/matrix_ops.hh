#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/matrix/scalar_matrix_expr.hh>
#include <euler/matrix/specialized.hh>
#include <euler/vector/vector.hh>
#include <euler/core/error.hh>
#include <euler/core/simd.hh>
#include <euler/core/approx_equal.hh>
#include <algorithm>
#include <type_traits>

namespace euler {

// Forward declarations
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;
template<typename T, size_t N> class vector;
template<typename Expr> class matrix_inverse_expression;

// Matrix-matrix multiplication - direct evaluation for concrete matrices
// Note: According to user requirements, we want lazy evaluation for all operations.
// This direct multiplication is kept for cases where the result is immediately assigned.
template<typename T, size_t M, size_t N, size_t P, bool RowMajor1, bool RowMajor2>
auto multiply_direct(const matrix<T, M, N, RowMajor1>& a, const matrix<T, N, P, RowMajor2>& b) 
    -> matrix<T, M, P, RowMajor1> {
    // Use specialized implementations for small matrices
    if constexpr ((M == 2 && N == 2 && P == 2) || 
                  (M == 3 && N == 3 && P == 3) || 
                  (M == 4 && N == 4 && P == 4)) {
        if constexpr (std::is_same_v<T, float>) {
            return multiply_specialized(a, b);
        }
    }
    
    // Generic implementation for other sizes
    matrix<T, M, P, RowMajor1> result;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < P; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < N; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

// Note: Matrix multiplication with expressions creates an expression template through matrix_expr.hh

// Matrix-Vector multiplication (matrix * column vector) - creates lazy expression
// Only for actual vector types, not matrices
template<typename MatExpr, typename T, size_t N,
         typename = std::enable_if_t<
             is_matrix_expression_v<std::decay_t<MatExpr>> && 
             expression_traits<std::decay_t<MatExpr>>::cols == N &&
             !std::is_arithmetic_v<std::decay_t<MatExpr>>>>
matrix_vector_multiply_expression<MatExpr, vector<T, N>> 
operator*(const MatExpr& m, const vector<T, N>& v) {
    return matrix_vector_multiply_expression<MatExpr, vector<T, N>>(m, v);
}


// Vector-Matrix multiplication (row vector * matrix) - creates lazy expression  
// Only for actual row_vector types, not matrices
template<typename T, size_t M, typename MatExpr,
         typename = std::enable_if_t<
             is_matrix_expression_v<std::decay_t<MatExpr>> && 
             expression_traits<std::decay_t<MatExpr>>::rows == M>>
vector_matrix_multiply_expression<row_vector<T, M>, MatExpr>
operator*(const row_vector<T, M>& v, const MatExpr& m) {
    return vector_matrix_multiply_expression<row_vector<T, M>, MatExpr>(v, m);
}

// Note: Matrix-scalar multiplication creates an expression template through matrix_expr.hh
// The operator* is defined there for lazy evaluation

// Note: Matrix-scalar division creates an expression template through matrix_expr.hh
// The operator/ is defined there for lazy evaluation

// Note: Matrix addition creates an expression template through matrix_expr.hh
// The operator+ is defined there for lazy evaluation

// Note: Matrix subtraction creates an expression template through matrix_expr.hh
// The operator- is defined there for lazy evaluation

// Note: Unary negation creates an expression template through matrix_expr.hh
// The unary operator- is defined there for lazy evaluation

// Transpose - universal function that handles both expressions and concrete matrices
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr>>>
auto transpose(const Expr& expr) {
    // For expressions (not concrete matrices), return a transpose expression
    if constexpr (!is_matrix_v<Expr>) {
        return transpose_expr(expr);
    } else {
        // For concrete matrices, perform direct transpose
        using T = typename expression_traits<Expr>::value_type;
        constexpr size_t M = expression_traits<Expr>::rows;
        constexpr size_t N = expression_traits<Expr>::cols;
        constexpr bool RowMajor = expression_traits<Expr>::row_major;
        
        const auto& m = static_cast<const matrix<T, M, N, RowMajor>&>(expr);
        
        // Use specialized implementation for 4x4 float matrices
        if constexpr (M == 4 && N == 4 && std::is_same_v<T, float>) {
            return fast_transpose_4x4(m);
        }
        
        matrix<T, N, M, RowMajor> result;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result(j, i) = m(i, j);
            }
        }
        
        return result;
    }
}

// Trace (sum of diagonal elements) for concrete matrices
template<typename T, size_t N, bool RowMajor>
T trace(const matrix<T, N, N, RowMajor>& m) {
    T sum = T(0);
    for (size_t i = 0; i < N; ++i) {
        sum += m(i, i);
    }
    return sum;
}

// Trace for expressions - evaluates the expression first
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && 
                                     !is_matrix_v<Expr> &&
                                     expression_traits<Expr>::rows == expression_traits<Expr>::cols>>
auto trace(const Expr& expr) {
    using value_type = typename expression_traits<Expr>::value_type;
    constexpr size_t size = expression_traits<Expr>::rows;
    constexpr bool row_major = expression_traits<Expr>::row_major;
    
    // Evaluate the expression to a concrete matrix
    matrix<value_type, size, size, row_major> m(expr);
    
    // Call the concrete matrix version
    return trace(m);
}

// Note: Hadamard product and division create expression templates through matrix_expr.hh
// These direct evaluation functions are kept for internal use if needed

// Component-wise multiplication (Hadamard product) - direct evaluation
template<typename T, size_t M, size_t N, bool RowMajor1, bool RowMajor2>
auto hadamard_direct(const matrix<T, M, N, RowMajor1>& a, const matrix<T, M, N, RowMajor2>& b) 
    -> matrix<T, M, N, RowMajor1> {
    matrix<T, M, N, RowMajor1> result;
    
    for (size_t i = 0; i < M * N; ++i) {
        result[i] = a[i] * b[i];
    }
    
    return result;
}

// Component-wise division - direct evaluation  
template<typename T, size_t M, size_t N, bool RowMajor1, bool RowMajor2>
auto hadamard_div_direct(const matrix<T, M, N, RowMajor1>& a, const matrix<T, M, N, RowMajor2>& b) 
    -> matrix<T, M, N, RowMajor1> {
    matrix<T, M, N, RowMajor1> result;
    
    for (size_t i = 0; i < M * N; ++i) {
        EULER_CHECK(b[i] != T(0), error_code::invalid_argument, 
                    "Division by zero in Hadamard division");
        result[i] = a[i] / b[i];
    }
    
    return result;
}

// Frobenius norm (sqrt of sum of squares)
template<typename T, size_t M, size_t N, bool RowMajor>
T frobenius_norm(const matrix<T, M, N, RowMajor>& m) {
    T sum = T(0);
    for (size_t i = 0; i < M * N; ++i) {
        sum += m[i] * m[i];
    }
    return std::sqrt(sum);
}

// ============================================================================
// Determinant specializations
// ============================================================================

// 2x2 determinant
template<typename T, bool RowMajor>
T determinant(const matrix<T, 2, 2, RowMajor>& m) {
    return fast_determinant_2x2(m);
}

// 3x3 determinant
template<typename T, bool RowMajor>
T determinant(const matrix<T, 3, 3, RowMajor>& m) {
    return fast_determinant_3x3(m);
}

// 4x4 determinant
template<typename T, bool RowMajor>
T determinant(const matrix<T, 4, 4, RowMajor>& m) {
    return fast_determinant_4x4(m);
}

// Universal determinant for expressions - evaluates the expression first
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && 
                                     !is_matrix_v<Expr> &&
                                     expression_traits<Expr>::rows == expression_traits<Expr>::cols &&
                                     expression_traits<Expr>::rows >= 2 &&
                                     expression_traits<Expr>::rows <= 4>>
auto determinant(const Expr& expr) {
    using value_type = typename expression_traits<Expr>::value_type;
    constexpr size_t size = expression_traits<Expr>::rows;
    constexpr bool row_major = expression_traits<Expr>::row_major;
    
    // Evaluate the expression to a concrete matrix
    matrix<value_type, size, size, row_major> m(expr);
    
    // Call the appropriate fast determinant function
    return determinant(m);
}

// ============================================================================
// Matrix inverse - creates lazy expression
// ============================================================================

// Universal inverse function that works with any square matrix expression
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && 
                                     expression_traits<Expr>::rows == expression_traits<Expr>::cols &&
                                     expression_traits<Expr>::rows >= 2 &&
                                     expression_traits<Expr>::rows <= 4>>
auto inverse(const Expr& expr) {
    return matrix_inverse_expression<Expr>(expr);
}

// Note: Direct inverse functions are kept for internal use

// 2x2 inverse - direct evaluation
template<typename T, bool RowMajor>
auto inverse_direct(const matrix<T, 2, 2, RowMajor>& m) 
    -> matrix<T, 2, 2, RowMajor> {
    T det = fast_determinant_2x2(m);
    EULER_CHECK(std::abs(det) > constants<T>::epsilon, error_code::singular_matrix,
                "Matrix is singular (determinant = ", det, ")");
    
    return fast_inverse_2x2(m);
}

// 3x3 inverse - direct evaluation
template<typename T, bool RowMajor>
auto inverse_direct(const matrix<T, 3, 3, RowMajor>& m) 
    -> matrix<T, 3, 3, RowMajor> {
    T det = fast_determinant_3x3(m);
    EULER_CHECK(std::abs(det) > constants<T>::epsilon, error_code::singular_matrix,
                "Matrix is singular (determinant = ", det, ")");
    
    return fast_inverse_3x3(m);
}

// 4x4 inverse - direct evaluation
template<typename T, bool RowMajor>
auto inverse_direct(const matrix<T, 4, 4, RowMajor>& m) 
    -> matrix<T, 4, 4, RowMajor> {
    return fast_inverse_4x4(m);
}

// Outer product (vector ⊗ vector = matrix)
template<typename T, size_t M, size_t N>
auto outer_product(const vector<T, M>& u, const vector<T, N>& v) 
    -> matrix<T, M, N> {
    matrix<T, M, N> result;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) = u[i] * v[j];
        }
    }
    
    return result;
}

// Matrix power (for integer exponents) - works with any square matrix expression
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && 
                                     expression_traits<Expr>::rows == expression_traits<Expr>::cols>>
auto pow(const Expr& expr, int exponent) {
    using T = typename expression_traits<Expr>::value_type;
    constexpr size_t N = expression_traits<Expr>::rows;
    constexpr bool RowMajor = expression_traits<Expr>::row_major;
    
    // For expressions, we need to evaluate to a concrete matrix first
    matrix<T, N, N, RowMajor> m;
    if constexpr (is_matrix_v<Expr>) {
        m = expr;
    } else {
        // Expression - evaluate it
        m = matrix<T, N, N, RowMajor>(expr);
    }
    
    if (exponent == 0) {
        return matrix<T, N, N, RowMajor>::identity();
    }
    
    if (exponent < 0) {
        return pow(inverse(m), -exponent);
    }
    
    // Binary exponentiation for efficiency
    matrix<T, N, N, RowMajor> result = matrix<T, N, N, RowMajor>::identity();
    matrix<T, N, N, RowMajor> base = m;
    
    while (exponent > 0) {
        if (exponent & 1) {
            result = multiply_direct(result, base);  // Use direct multiplication
        }
        base = multiply_direct(base, base);  // Use direct multiplication
        exponent >>= 1;
    }
    
    return result;
}


} // namespace euler