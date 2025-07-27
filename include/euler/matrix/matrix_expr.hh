#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/core/expression.hh>
#include <euler/core/simd.hh>
#include <type_traits>

namespace euler {

// Forward declarations for expression_storage specializations
template<typename Expr1, typename Expr2, typename Op> class matrix_binary_expression;
template<typename Expr, typename Scalar, typename Op> class matrix_scalar_expression;
template<typename Expr, typename Op> class matrix_unary_expression;
template<typename Expr> class matrix_transpose_expression;
template<typename Expr1, typename Expr2> class matrix_multiply_expression;
template<typename Expr> class matrix_inverse_expression;

// Expression storage specializations for matrix expressions
template<typename Expr1, typename Expr2, typename Op>
struct expression_storage<matrix_binary_expression<Expr1, Expr2, Op>> {
    using type = matrix_binary_expression<Expr1, Expr2, Op>;
};

template<typename Expr, typename Scalar, typename Op>
struct expression_storage<matrix_scalar_expression<Expr, Scalar, Op>> {
    using type = matrix_scalar_expression<Expr, Scalar, Op>;
};

template<typename Expr, typename Op>
struct expression_storage<matrix_unary_expression<Expr, Op>> {
    using type = matrix_unary_expression<Expr, Op>;
};

template<typename Expr>
struct expression_storage<matrix_transpose_expression<Expr>> {
    using type = matrix_transpose_expression<Expr>;
};

template<typename Expr1, typename Expr2>
struct expression_storage<matrix_multiply_expression<Expr1, Expr2>> {
    using type = matrix_multiply_expression<Expr1, Expr2>;
};

template<typename Expr>
struct expression_storage<matrix_inverse_expression<Expr>> {
    using type = matrix_inverse_expression<Expr>;
};

// Helper to check if SIMD is available for a type
template<typename T>
constexpr bool has_simd_v = simd_traits<T>::has_simd;

// Forward declarations
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;

// Binary operation functors for matrices
struct matrix_add {
    template<typename T>
    static constexpr T apply(T a, T b) { return a + b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a + b; }
};

struct matrix_sub {
    template<typename T>
    static constexpr T apply(T a, T b) { return a - b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a - b; }
};

struct matrix_mul {
    template<typename T>
    static constexpr T apply(T a, T b) { return a * b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a * b; }
};

struct matrix_div {
    template<typename T>
    static constexpr T apply(T a, T b) { return a / b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a / b; }
};

// Hadamard (element-wise) operations
struct matrix_hadamard_mul {
    template<typename T>
    static constexpr T apply(T a, T b) { return a * b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a * b; }
};

struct matrix_hadamard_div {
    template<typename T>
    static constexpr T apply(T a, T b) { return a / b; }
    
    template<typename T>
    static constexpr auto apply_simd(T a, T b) { return a / b; }
};

// Matrix binary expression template
template<typename Expr1, typename Expr2, typename Op>
class matrix_binary_expression : public expression<matrix_binary_expression<Expr1, Expr2, Op>, typename std::common_type_t<
        typename expression_traits<Expr1>::value_type,
        typename expression_traits<Expr2>::value_type
    >> {
public:
    using value_type = typename std::common_type_t<
        typename expression_traits<Expr1>::value_type,
        typename expression_traits<Expr2>::value_type
    >;
    
    using expr1_storage = typename expression_storage<Expr1>::type;
    using expr2_storage = typename expression_storage<Expr2>::type;
    
    static constexpr size_t rows = expression_traits<Expr1>::rows;
    static constexpr size_t cols = expression_traits<Expr1>::cols;
    static constexpr bool row_major = expression_traits<Expr1>::row_major;
    
    static_assert(expression_traits<Expr1>::rows == expression_traits<Expr2>::rows,
                  "Matrix dimensions must match for binary operations");
    static_assert(expression_traits<Expr1>::cols == expression_traits<Expr2>::cols,
                  "Matrix dimensions must match for binary operations");
    
    // Perfect forwarding constructor for rvalue/lvalue handling
    template<typename E1, typename E2>
    constexpr matrix_binary_expression(E1&& e1, E2&& e2)
        : expr1_(std::forward<E1>(e1)), expr2_(std::forward<E2>(e2)) {}
    
    // Element access
    constexpr value_type operator()(size_t i, size_t j) const {
        return Op::apply(expr1_(i, j), expr2_(i, j));
    }
    
    // Linear access
    constexpr value_type operator[](size_t idx) const {
        if constexpr (row_major) {
            size_t i = idx / cols;
            size_t j = idx % cols;
            return (*this)(i, j);
        } else {
            size_t j = idx / rows;
            size_t i = idx % rows;
            return (*this)(i, j);
        }
    }
    
    // SIMD evaluation for contiguous memory
    template<typename Batch>
    Batch eval_simd(size_t idx) const {
        return Op::apply_simd(expr1_.template eval_simd<Batch>(idx),
                             expr2_.template eval_simd<Batch>(idx));
    }
    
    // Element evaluation for expression interface
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    constexpr value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        matrix<value_type, rows, cols, row_major> result;
        
        if constexpr (has_simd_v<value_type>) {
            constexpr size_t batch_size = simd_traits<value_type>::size;
            const size_t simd_size = size() - (size() % batch_size);
            
            // SIMD evaluation
            for (size_t i = 0; i < simd_size; i += batch_size) {
                auto batch = eval_simd<typename simd_traits<value_type>::batch_type>(i);
                batch.store_aligned(&result[i]);
            }
            
            // Handle remainder
            for (size_t i = simd_size; i < size(); ++i) {
                result[i] = (*this)[i];
            }
        } else {
            // Non-SIMD evaluation
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = (*this)(i, j);
                }
            }
        }
        
        return result;
    }
    
private:
    expr1_storage expr1_;
    expr2_storage expr2_;
};

// Matrix-scalar binary expression
template<typename Expr, typename Scalar, typename Op>
class matrix_scalar_expression : public expression<matrix_scalar_expression<Expr, Scalar, Op>, typename std::common_type_t<
        typename expression_traits<Expr>::value_type,
        Scalar
    >> {
public:
    using value_type = typename std::common_type_t<
        typename expression_traits<Expr>::value_type,
        Scalar
    >;
    
    using expr_storage = typename expression_storage<Expr>::type;
    
    static constexpr size_t rows = expression_traits<Expr>::rows;
    static constexpr size_t cols = expression_traits<Expr>::cols;
    static constexpr bool row_major = expression_traits<Expr>::row_major;
    
    constexpr matrix_scalar_expression(const Expr& expr, Scalar ascalar)
        : expr_(expr), scalar_(ascalar) {}
    
    // Element access
    constexpr value_type operator()(size_t i, size_t j) const {
        return Op::apply(expr_(i, j), scalar_);
    }
    
    // Linear access
    constexpr value_type operator[](size_t idx) const {
        return Op::apply(expr_[idx], scalar_);
    }
    
    // SIMD evaluation
    template<typename Batch>
    Batch eval_simd(size_t idx) const {
        return Op::apply_simd(expr_.template eval_simd<Batch>(idx), Batch(scalar_));
    }
    
    // Element evaluation for expression interface
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    constexpr value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        matrix<value_type, rows, cols, row_major> result;
        
        if constexpr (has_simd_v<value_type>) {
            constexpr size_t batch_size = simd_traits<value_type>::size;
            const size_t simd_size = size() - (size() % batch_size);
            
            // SIMD evaluation
            for (size_t i = 0; i < simd_size; i += batch_size) {
                auto batch = eval_simd<typename simd_traits<value_type>::batch_type>(i);
                batch.store_aligned(&result[i]);
            }
            
            // Handle remainder
            for (size_t i = simd_size; i < size(); ++i) {
                result[i] = (*this)[i];
            }
        } else {
            // Non-SIMD evaluation
            for (size_t i = 0; i < size(); ++i) {
                result[i] = (*this)[i];
            }
        }
        
        return result;
    }
    
private:
    expr_storage expr_;
    Scalar scalar_;
};

// Unary operation functors
struct matrix_negate {
    template<typename T>
    static constexpr T apply(T a) { return -a; }
    
    template<typename T>
    static constexpr auto apply_simd(T a) { return -a; }
};

// Matrix unary expression template
template<typename Expr, typename Op>
class matrix_unary_expression : public expression<matrix_unary_expression<Expr, Op>, typename expression_traits<Expr>::value_type> {
public:
    using value_type = typename expression_traits<Expr>::value_type;
    using expr_storage = typename expression_storage<Expr>::type;
    
    static constexpr size_t rows = expression_traits<Expr>::rows;
    static constexpr size_t cols = expression_traits<Expr>::cols;
    static constexpr bool row_major = expression_traits<Expr>::row_major;
    
    constexpr explicit matrix_unary_expression(const Expr& expr)
        : expr_(expr) {}
    
    // Element access
    constexpr value_type operator()(size_t i, size_t j) const {
        return Op::apply(expr_(i, j));
    }
    
    // Linear access
    constexpr value_type operator[](size_t idx) const {
        return Op::apply(expr_[idx]);
    }
    
    // SIMD evaluation
    template<typename Batch>
    Batch eval_simd(size_t idx) const {
        return Op::apply_simd(expr_.template eval_simd<Batch>(idx));
    }
    
    // Element evaluation for expression interface
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    constexpr value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        matrix<value_type, rows, cols, row_major> result;
        
        if constexpr (has_simd_v<value_type>) {
            constexpr size_t batch_size = simd_traits<value_type>::size;
            const size_t simd_size = size() - (size() % batch_size);
            
            // SIMD evaluation
            for (size_t i = 0; i < simd_size; i += batch_size) {
                auto batch = eval_simd<typename simd_traits<value_type>::batch_type>(i);
                batch.store_aligned(&result[i]);
            }
            
            // Handle remainder
            for (size_t i = simd_size; i < size(); ++i) {
                result[i] = (*this)[i];
            }
        } else {
            // Non-SIMD evaluation
            for (size_t i = 0; i < size(); ++i) {
                result[i] = (*this)[i];
            }
        }
        
        return result;
    }
    
private:
    expr_storage expr_;
};

// Transpose expression template
template<typename Expr>
class matrix_transpose_expression : public expression<matrix_transpose_expression<Expr>, typename expression_traits<Expr>::value_type> {
public:
    using value_type = typename expression_traits<Expr>::value_type;
    using expr_storage = typename expression_storage<Expr>::type;
    
    static constexpr size_t rows = expression_traits<Expr>::cols;  // Swapped
    static constexpr size_t cols = expression_traits<Expr>::rows;  // Swapped
    static constexpr bool row_major = expression_traits<Expr>::row_major;
    
    constexpr explicit matrix_transpose_expression(const Expr& expr)
        : expr_(expr) {}
    
    // Element access (swapped indices)
    constexpr value_type operator()(size_t i, size_t j) const {
        return expr_(j, i);
    }
    
    // Linear access (more complex due to transpose)
    constexpr value_type operator[](size_t idx) const {
        if constexpr (row_major) {
            size_t i = idx / cols;
            size_t j = idx % cols;
            return expr_(j, i);
        } else {
            size_t j = idx / rows;
            size_t i = idx % rows;
            return expr_(j, i);
        }
    }
    
    // Element evaluation for expression interface
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    constexpr value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        matrix<value_type, rows, cols, row_major> result;
        
        // Transpose is not easily SIMD-izable in general case
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = expr_(j, i);
            }
        }
        
        return result;
    }
    
private:
    expr_storage expr_;
};

// Expression traits specializations for matrix expressions
template<typename Expr1, typename Expr2, typename Op>
struct expression_traits<matrix_binary_expression<Expr1, Expr2, Op>> {
    using value_type = typename matrix_binary_expression<Expr1, Expr2, Op>::value_type;
    static constexpr size_t rows = matrix_binary_expression<Expr1, Expr2, Op>::rows;
    static constexpr size_t cols = matrix_binary_expression<Expr1, Expr2, Op>::cols;
    static constexpr bool row_major = matrix_binary_expression<Expr1, Expr2, Op>::row_major;
};

template<typename Expr, typename Scalar, typename Op>
struct expression_traits<matrix_scalar_expression<Expr, Scalar, Op>> {
    using value_type = typename matrix_scalar_expression<Expr, Scalar, Op>::value_type;
    static constexpr size_t rows = matrix_scalar_expression<Expr, Scalar, Op>::rows;
    static constexpr size_t cols = matrix_scalar_expression<Expr, Scalar, Op>::cols;
    static constexpr bool row_major = matrix_scalar_expression<Expr, Scalar, Op>::row_major;
};

template<typename Expr, typename Op>
struct expression_traits<matrix_unary_expression<Expr, Op>> {
    using value_type = typename matrix_unary_expression<Expr, Op>::value_type;
    static constexpr size_t rows = matrix_unary_expression<Expr, Op>::rows;
    static constexpr size_t cols = matrix_unary_expression<Expr, Op>::cols;
    static constexpr bool row_major = matrix_unary_expression<Expr, Op>::row_major;
};

template<typename Expr>
struct expression_traits<matrix_transpose_expression<Expr>> {
    using value_type = typename matrix_transpose_expression<Expr>::value_type;
    static constexpr size_t rows = matrix_transpose_expression<Expr>::rows;
    static constexpr size_t cols = matrix_transpose_expression<Expr>::cols;
    static constexpr bool row_major = matrix_transpose_expression<Expr>::row_major;
};

// Operator overloads to create expressions
// Lvalue + Lvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<Expr2>>>
auto operator+(const Expr1& e1, const Expr2& e2) {
    return matrix_binary_expression<Expr1, Expr2, matrix_add>(e1, e2);
}

// Rvalue + Lvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<std::decay_t<Expr1>> && is_matrix_expression_v<Expr2>>>
auto operator+(Expr1&& e1, const Expr2& e2) {
    return matrix_binary_expression<std::decay_t<Expr1>, Expr2, matrix_add>(std::forward<Expr1>(e1), e2);
}

// Lvalue + Rvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<std::decay_t<Expr2>>>>
auto operator+(const Expr1& e1, Expr2&& e2) {
    return matrix_binary_expression<Expr1, std::decay_t<Expr2>, matrix_add>(e1, std::forward<Expr2>(e2));
}

// Rvalue + Rvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<std::decay_t<Expr1>> && is_matrix_expression_v<std::decay_t<Expr2>>>>
auto operator+(Expr1&& e1, Expr2&& e2) {
    return matrix_binary_expression<std::decay_t<Expr1>, std::decay_t<Expr2>, matrix_add>(std::forward<Expr1>(e1), std::forward<Expr2>(e2));
}

// Lvalue - Lvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<Expr2>>>
auto operator-(const Expr1& e1, const Expr2& e2) {
    return matrix_binary_expression<Expr1, Expr2, matrix_sub>(e1, e2);
}

// Rvalue - Lvalue  
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<std::decay_t<Expr1>> && is_matrix_expression_v<Expr2>>>
auto operator-(Expr1&& e1, const Expr2& e2) {
    return matrix_binary_expression<std::decay_t<Expr1>, Expr2, matrix_sub>(std::forward<Expr1>(e1), e2);
}

// Lvalue - Rvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<std::decay_t<Expr2>>>>
auto operator-(const Expr1& e1, Expr2&& e2) {
    return matrix_binary_expression<Expr1, std::decay_t<Expr2>, matrix_sub>(e1, std::forward<Expr2>(e2));
}

// Rvalue - Rvalue
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<std::decay_t<Expr1>> && is_matrix_expression_v<std::decay_t<Expr2>>>>
auto operator-(Expr1&& e1, Expr2&& e2) {
    return matrix_binary_expression<std::decay_t<Expr1>, std::decay_t<Expr2>, matrix_sub>(std::forward<Expr1>(e1), std::forward<Expr2>(e2));
}

template<typename Expr, typename Scalar,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && std::is_arithmetic_v<Scalar>>>
auto operator*(const Expr& expr, Scalar ascalar) {
    return matrix_scalar_expression<Expr, Scalar, matrix_mul>(expr, ascalar);
}

template<typename Scalar, typename Expr,
         typename = std::enable_if_t<std::is_arithmetic_v<Scalar> && is_matrix_expression_v<Expr>>>
auto operator*(Scalar ascalar, const Expr& expr) {
    return matrix_scalar_expression<Expr, Scalar, matrix_mul>(expr, ascalar);
}

template<typename Expr, typename Scalar,
         typename = std::enable_if_t<is_matrix_expression_v<Expr> && std::is_arithmetic_v<Scalar>>>
auto operator/(const Expr& expr, Scalar ascalar) {
    return matrix_scalar_expression<Expr, Scalar, matrix_div>(expr, ascalar);
}

template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr>>>
auto operator-(const Expr& expr) {
    return matrix_unary_expression<Expr, matrix_negate>(expr);
}

// Lazy transpose function
template<typename Expr,
         typename = std::enable_if_t<is_matrix_expression_v<Expr>>>
auto transpose_expr(const Expr& expr) {
    return matrix_transpose_expression<Expr>(expr);
}

// Matrix multiplication expression template
template<typename Expr1, typename Expr2>
class matrix_multiply_expression : public expression<matrix_multiply_expression<Expr1, Expr2>, 
    typename std::common_type_t<
        typename expression_traits<Expr1>::value_type,
        typename expression_traits<Expr2>::value_type
    >> {
public:
    using value_type = typename std::common_type_t<
        typename expression_traits<Expr1>::value_type,
        typename expression_traits<Expr2>::value_type
    >;
    using expr1_storage = typename expression_storage<Expr1>::type;
    using expr2_storage = typename expression_storage<Expr2>::type;
    
    static constexpr size_t rows = expression_traits<Expr1>::rows;
    static constexpr size_t cols = expression_traits<Expr2>::cols;
    static constexpr bool row_major = expression_traits<Expr1>::row_major;
    
    // Inner dimension must match
    static_assert(expression_traits<Expr1>::cols == expression_traits<Expr2>::rows,
                  "Matrix dimensions must be compatible for multiplication");
    
    static constexpr size_t inner_dim = expression_traits<Expr1>::cols;
    
    constexpr matrix_multiply_expression(const Expr1& e1, const Expr2& e2)
        : expr1_(e1), expr2_(e2) {}
    
    // Element access - performs dot product of row i from expr1 with column j from expr2
    constexpr value_type operator()(size_t i, size_t j) const {
        value_type sum = value_type(0);
        for (size_t k = 0; k < inner_dim; ++k) {
            sum += expr1_(i, k) * expr2_(k, j);
        }
        return sum;
    }
    
    // Linear access
    constexpr value_type operator[](size_t idx) const {
        if constexpr (row_major) {
            size_t i = idx / cols;
            size_t j = idx % cols;
            return (*this)(i, j);
        } else {
            size_t j = idx / rows;
            size_t i = idx % rows;
            return (*this)(i, j);
        }
    }
    
    // Element evaluation for expression interface
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    constexpr value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        matrix<value_type, rows, cols, row_major> result;
        
        // Use specialized multiplication if available
        if constexpr (rows <= 4 && cols <= 4 && inner_dim <= 4 && 
                      std::is_same_v<value_type, float>) {
            // Could use specialized SIMD multiplication here
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = (*this)(i, j);
                }
            }
        } else {
            // General case
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = (*this)(i, j);
                }
            }
        }
        
        return result;
    }
    
    // SIMD evaluation is complex for matrix multiplication
    // For now, we'll rely on specialized implementations in eval()
    
private:
    expr1_storage expr1_;
    expr2_storage expr2_;
};

// Expression traits specialization
template<typename Expr1, typename Expr2>
struct expression_traits<matrix_multiply_expression<Expr1, Expr2>> {
    using value_type = typename matrix_multiply_expression<Expr1, Expr2>::value_type;
    static constexpr size_t rows = matrix_multiply_expression<Expr1, Expr2>::rows;
    static constexpr size_t cols = matrix_multiply_expression<Expr1, Expr2>::cols;
    static constexpr bool row_major = matrix_multiply_expression<Expr1, Expr2>::row_major;
};

// Operator overload for matrix multiplication expression - creates expression for all matrix types
// Works for any compatible matrix dimensions
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<Expr2> &&
                                     expression_traits<Expr1>::cols == expression_traits<Expr2>::rows>>
auto operator*(const Expr1& e1, const Expr2& e2) {
    return matrix_multiply_expression<Expr1, Expr2>(e1, e2);
}

// Hadamard (element-wise) multiplication - creates expression for lazy evaluation
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<Expr2> &&
                                     expression_traits<Expr1>::rows == expression_traits<Expr2>::rows &&
                                     expression_traits<Expr1>::cols == expression_traits<Expr2>::cols>>
auto hadamard(const Expr1& e1, const Expr2& e2) {
    return matrix_binary_expression<Expr1, Expr2, matrix_hadamard_mul>(e1, e2);
}

// Hadamard (element-wise) division - creates expression for lazy evaluation
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_matrix_expression_v<Expr1> && is_matrix_expression_v<Expr2> &&
                                     expression_traits<Expr1>::rows == expression_traits<Expr2>::rows &&
                                     expression_traits<Expr1>::cols == expression_traits<Expr2>::cols>>
auto hadamard_div(const Expr1& e1, const Expr2& e2) {
    return matrix_binary_expression<Expr1, Expr2, matrix_hadamard_div>(e1, e2);
}

// Matrix inverse expression template
template<typename Expr>
class matrix_inverse_expression : public expression<matrix_inverse_expression<Expr>, typename expression_traits<Expr>::value_type> {
public:
    using value_type = typename expression_traits<Expr>::value_type;
    using expr_storage = typename expression_storage<Expr>::type;
    
    static constexpr size_t rows = expression_traits<Expr>::rows;
    static constexpr size_t cols = expression_traits<Expr>::cols;
    static constexpr bool row_major = expression_traits<Expr>::row_major;
    
    static_assert(rows == cols, "Inverse is only defined for square matrices");
    
    explicit matrix_inverse_expression(const Expr& expr)
        : expr_(expr), inverse_computed_(false), cached_inverse_() {}
    
    // Element access - computes inverse on first access
    value_type operator()(size_t i, size_t j) const {
        ensure_inverse_computed();
        return cached_inverse_(i, j);
    }
    
    // Linear access
    value_type operator[](size_t idx) const {
        ensure_inverse_computed();
        return cached_inverse_[idx];
    }
    
    // Element evaluation for expression interface
    value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    // Single index access for vector-like usage
    value_type eval_scalar(size_t idx) const {
        if constexpr (cols == 1) {
            return (*this)(idx, 0);
        } else if constexpr (rows == 1) {
            return (*this)(0, idx);
        } else {
            return (*this)[idx];
        }
    }
    
    // Size for linear iteration
    static constexpr size_t size() { return rows * cols; }
    
    // Evaluate to matrix
    auto eval() const {
        ensure_inverse_computed();
        return cached_inverse_;
    }
    
private:
    expr_storage expr_;
    mutable bool inverse_computed_;
    mutable matrix<value_type, rows, cols, row_major> cached_inverse_;
    
    void ensure_inverse_computed() const {
        if (!inverse_computed_) {
            // Evaluate the expression to a concrete matrix
            matrix<value_type, rows, cols, row_major> m;
            if constexpr (is_matrix_v<Expr>) {
                m = expr_;
            } else {
                m = matrix<value_type, rows, cols, row_major>(expr_);
            }
            
            // Compute inverse based on size
            if constexpr (rows == 2) {
                #ifdef EULER_DEBUG
                value_type det = fast_determinant_2x2(m);
                EULER_CHECK(std::abs(det) > constants<value_type>::epsilon, error_code::singular_matrix,
                            "Matrix is singular (determinant = ", det, ")");
                #endif
                cached_inverse_ = fast_inverse_2x2(m);
            } else if constexpr (rows == 3) {
                #ifdef EULER_DEBUG
                value_type det = fast_determinant_3x3(m);
                EULER_CHECK(std::abs(det) > constants<value_type>::epsilon, error_code::singular_matrix,
                            "Matrix is singular (determinant = ", det, ")");
                #endif
                cached_inverse_ = fast_inverse_3x3(m);
            } else if constexpr (rows == 4) {
                cached_inverse_ = fast_inverse_4x4(m);
            } else {
                static_assert(rows <= 4, "Inverse is only implemented for matrices up to 4x4");
            }
            
            inverse_computed_ = true;
        }
    }
};

// Expression traits specialization
template<typename Expr>
struct expression_traits<matrix_inverse_expression<Expr>> {
    using value_type = typename matrix_inverse_expression<Expr>::value_type;
    static constexpr size_t rows = matrix_inverse_expression<Expr>::rows;
    static constexpr size_t cols = matrix_inverse_expression<Expr>::cols;
    static constexpr bool row_major = matrix_inverse_expression<Expr>::row_major;
};

// Forward declaration for vector
template<typename T, size_t N> class vector;

// Matrix-Vector multiplication expression
template<typename MatExpr, typename VecExpr>
class matrix_vector_multiply_expression : public expression<matrix_vector_multiply_expression<MatExpr, VecExpr>, typename expression_traits<MatExpr>::value_type> {
public:
    using mat_traits = expression_traits<MatExpr>;
    using vec_traits = expression_traits<VecExpr>;
    using value_type = typename mat_traits::value_type;
    using mat_storage = typename expression_storage<MatExpr>::type;
    using vec_storage = typename expression_storage<VecExpr>::type;
    
    static constexpr size_t rows = mat_traits::rows;
    static constexpr size_t cols = 1;  // Result is a column vector
    static constexpr bool row_major = true;
    
    static_assert(mat_traits::cols == vec_traits::rows || mat_traits::cols == vec_traits::cols,
                  "Matrix columns must match vector size");
    
    matrix_vector_multiply_expression(const MatExpr& mat, const VecExpr& vec)
        : mat_(mat), vec_(vec) {}
    
    // Compute dot product of matrix row i with vector
    value_type operator()(size_t i, size_t j) const {
        if (j != 0) return value_type(0);
        return compute_element(i);
    }
    
    value_type operator[](size_t idx) const {
        return compute_element(idx);
    }
    
    value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    value_type eval_scalar(size_t idx) const {
        return compute_element(idx);
    }
    
    static constexpr size_t size() { return rows; }
    
private:
    mat_storage mat_;
    vec_storage vec_;
    
    value_type compute_element(size_t i) const {
        value_type sum = value_type(0);
        for (size_t j = 0; j < mat_traits::cols; ++j) {
            if constexpr (vec_traits::rows == 1) {
                // Row vector case
                sum += mat_(i, j) * vec_(0, j);
            } else if constexpr (vec_traits::cols == 1) {
                // Column vector case
                sum += mat_(i, j) * vec_(j, 0);
            } else {
                // Regular vector
                sum += mat_(i, j) * vec_[j];
            }
        }
        return sum;
    }
};

// Vector-Matrix multiplication expression (row vector * matrix)
template<typename VecExpr, typename MatExpr>
class vector_matrix_multiply_expression : public expression<vector_matrix_multiply_expression<VecExpr, MatExpr>, typename expression_traits<VecExpr>::value_type> {
public:
    using vec_traits = expression_traits<VecExpr>;
    using mat_traits = expression_traits<MatExpr>;
    using value_type = typename vec_traits::value_type;
    using vec_storage = typename expression_storage<VecExpr>::type;
    using mat_storage = typename expression_storage<MatExpr>::type;
    
    static constexpr size_t rows = 1;  // Result is a row vector
    static constexpr size_t cols = mat_traits::cols;
    static constexpr bool row_major = true;
    
    static_assert(vec_traits::rows == mat_traits::rows || vec_traits::cols == mat_traits::rows,
                  "Vector size must match matrix rows");
    
    vector_matrix_multiply_expression(const VecExpr& vec, const MatExpr& mat)
        : vec_(vec), mat_(mat) {}
    
    // Compute dot product of vector with matrix column j
    value_type operator()(size_t i, size_t j) const {
        if (i != 0) return value_type(0);
        return compute_element(j);
    }
    
    value_type operator[](size_t idx) const {
        return compute_element(idx);
    }
    
    value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    value_type eval_scalar(size_t idx) const {
        return compute_element(idx);
    }
    
    static constexpr size_t size() { return cols; }
    
private:
    vec_storage vec_;
    mat_storage mat_;
    
    value_type compute_element(size_t j) const {
        value_type sum = value_type(0);
        for (size_t i = 0; i < mat_traits::rows; ++i) {
            if constexpr (vec_traits::rows == 1) {
                // Row vector case
                sum += vec_(0, i) * mat_(i, j);
            } else if constexpr (vec_traits::cols == 1) {
                // Column vector case
                sum += vec_(i, 0) * mat_(i, j);
            } else {
                // Regular vector
                sum += vec_[i] * mat_(i, j);
            }
        }
        return sum;
    }
};

// Expression storage specializations
template<typename MatExpr, typename VecExpr>
struct expression_storage<matrix_vector_multiply_expression<MatExpr, VecExpr>> {
    using type = matrix_vector_multiply_expression<MatExpr, VecExpr>;
};

template<typename VecExpr, typename MatExpr>
struct expression_storage<vector_matrix_multiply_expression<VecExpr, MatExpr>> {
    using type = vector_matrix_multiply_expression<VecExpr, MatExpr>;
};

// Expression traits specializations
template<typename MatExpr, typename VecExpr>
struct expression_traits<matrix_vector_multiply_expression<MatExpr, VecExpr>> {
    using value_type = typename matrix_vector_multiply_expression<MatExpr, VecExpr>::value_type;
    static constexpr size_t rows = matrix_vector_multiply_expression<MatExpr, VecExpr>::rows;
    static constexpr size_t cols = matrix_vector_multiply_expression<MatExpr, VecExpr>::cols;
    static constexpr bool row_major = matrix_vector_multiply_expression<MatExpr, VecExpr>::row_major;
};

template<typename VecExpr, typename MatExpr>
struct expression_traits<vector_matrix_multiply_expression<VecExpr, MatExpr>> {
    using value_type = typename vector_matrix_multiply_expression<VecExpr, MatExpr>::value_type;
    static constexpr size_t rows = vector_matrix_multiply_expression<VecExpr, MatExpr>::rows;
    static constexpr size_t cols = vector_matrix_multiply_expression<VecExpr, MatExpr>::cols;
    static constexpr bool row_major = vector_matrix_multiply_expression<VecExpr, MatExpr>::row_major;
};

} // namespace euler