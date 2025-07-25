#pragma once

#include <euler/core/expression.hh>
#include <euler/core/scalar_ops.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/matrix/matrix.hh>
#include <type_traits>
#include <cmath>

namespace euler {

// Scalar-matrix expression for operations like scalar / matrix
template<typename Scalar, typename MatExpr, typename Op>
class scalar_matrix_expression : public expression<scalar_matrix_expression<Scalar, MatExpr, Op>,
                                                  std::common_type_t<Scalar, typename expression_traits<MatExpr>::value_type>> {
    static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be arithmetic type");
    static_assert(is_matrix_expression_v<MatExpr>, "Second argument must be a matrix expression");
    
    using mat_storage = typename expression_storage<MatExpr>::type;
    
    Scalar scalar_;
    mat_storage mat_;
    
public:
    using value_type = std::common_type_t<Scalar, typename expression_traits<MatExpr>::value_type>;
    static constexpr size_t rows = expression_traits<MatExpr>::rows;
    static constexpr size_t cols = expression_traits<MatExpr>::cols;
    static constexpr bool row_major = expression_traits<MatExpr>::row_major;
    
    constexpr scalar_matrix_expression(const Scalar& s, const MatExpr& mat)
        : scalar_(s), mat_(mat) {}
    
    constexpr value_type operator()(size_t i, size_t j) const {
        return Op::apply(static_cast<value_type>(scalar_), 
                        static_cast<value_type>(mat_(i, j)));
    }
    
    // Support both row-major and column-major indexing
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
    
    // Required by expression base class
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        return (*this)(i, j);
    }
    
    constexpr value_type eval_scalar(size_t idx) const {
        return (*this)[idx];
    }
};

// Expression traits for scalar-matrix expressions
template<typename Scalar, typename MatExpr, typename Op>
struct expression_traits<scalar_matrix_expression<Scalar, MatExpr, Op>> {
    using value_type = typename scalar_matrix_expression<Scalar, MatExpr, Op>::value_type;
    static constexpr size_t rows = scalar_matrix_expression<Scalar, MatExpr, Op>::rows;
    static constexpr size_t cols = scalar_matrix_expression<Scalar, MatExpr, Op>::cols;
    static constexpr bool row_major = scalar_matrix_expression<Scalar, MatExpr, Op>::row_major;
    static constexpr bool is_vector = false;
    static constexpr bool is_matrix = true;
};

// Operator overloads for scalar / matrix
template<typename Scalar, typename MatExpr>
constexpr auto operator/(const Scalar& s, const MatExpr& mat)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && is_matrix_expression_v<MatExpr>,
        scalar_matrix_expression<Scalar, MatExpr, scalar_div_op>
    >
{
    return scalar_matrix_expression<Scalar, MatExpr, scalar_div_op>(s, mat);
}

// Operator overloads for scalar - matrix
template<typename Scalar, typename MatExpr>
constexpr auto operator-(const Scalar& s, const MatExpr& mat)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && is_matrix_expression_v<MatExpr>,
        scalar_matrix_expression<Scalar, MatExpr, scalar_sub_op>
    >
{
    return scalar_matrix_expression<Scalar, MatExpr, scalar_sub_op>(s, mat);
}

// Power operator (using ^ for element-wise power)
template<typename Scalar, typename MatExpr>
constexpr auto operator^(const Scalar& s, const MatExpr& mat)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && is_matrix_expression_v<MatExpr>,
        scalar_matrix_expression<Scalar, MatExpr, scalar_pow_op>
    >
{
    return scalar_matrix_expression<Scalar, MatExpr, scalar_pow_op>(s, mat);
}

// Helper function for element-wise reciprocal
template<typename MatExpr>
constexpr auto reciprocal(const MatExpr& mat)
    -> std::enable_if_t<
        is_matrix_expression_v<MatExpr>,
        scalar_matrix_expression<typename expression_traits<MatExpr>::value_type, MatExpr, scalar_div_op>
    >
{
    using value_type = typename expression_traits<MatExpr>::value_type;
    return value_type(1) / mat;
}

// Element-wise inverse (alias for reciprocal)
template<typename MatExpr>
constexpr auto element_inverse(const MatExpr& mat)
    -> std::enable_if_t<
        is_matrix_expression_v<MatExpr>,
        decltype(reciprocal(mat))
    >
{
    return reciprocal(mat);
}

// Expression storage specializations for scalar-matrix expressions
template<typename Scalar, typename MatExpr, typename Op>
struct expression_storage<scalar_matrix_expression<Scalar, MatExpr, Op>> {
    using type = scalar_matrix_expression<Scalar, MatExpr, Op>;
};

} // namespace euler