#pragma once

#include <euler/core/expression.hh>
#include <euler/core/scalar_ops.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/vector/vector_traits.hh>
#include <euler/vector/vector.hh>
#include <type_traits>
#include <cmath>

namespace euler {

// Scalar-vector expression for operations like scalar / vector
template<typename Scalar, typename VecExpr, typename Op>
class scalar_vector_expression : public expression<scalar_vector_expression<Scalar, VecExpr, Op>, 
                                                  std::common_type_t<Scalar, typename VecExpr::value_type>> {
    static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be arithmetic type");
    static_assert(is_vector_v<VecExpr> || is_any_vector_v<VecExpr>, "Second argument must be a vector expression");
    
    using vec_storage = typename expression_storage<VecExpr>::type;
    
    Scalar scalar_;
    vec_storage vec_;
    
public:
    using value_type = std::common_type_t<Scalar, typename VecExpr::value_type>;
    static constexpr size_t size = vector_size_v<VecExpr>;
    static constexpr size_t static_size = vector_size_v<VecExpr>;

    constexpr scalar_vector_expression(const Scalar& s, const VecExpr& vec)
        : scalar_(s), vec_(vec) {}
    
    constexpr value_type operator[](size_t i) const {
        return Op::apply(static_cast<value_type>(scalar_), 
                        static_cast<value_type>(vec_[i]));
    }
    
    constexpr value_type operator()(size_t i) const {
        return (*this)[i];
    }
    
    // For 2D access (vectors are treated as column vectors)
    constexpr value_type operator()(size_t i, size_t j) const {
        (void)j; // Unused for vectors
        return (*this)[i];
    }
    
    // Required by expression base class
    constexpr value_type eval_scalar(size_t i) const {
        return (*this)[i];
    }
    
    // For 2D access (vectors are treated as column vectors)
    constexpr value_type eval_scalar(size_t i, size_t j) const {
        (void)j; // Unused for vectors
        return (*this)[i];
    }
};

// Expression traits for scalar-vector expressions
template<typename Scalar, typename VecExpr, typename Op>
struct expression_traits<scalar_vector_expression<Scalar, VecExpr, Op>> {
    using value_type = typename scalar_vector_expression<Scalar, VecExpr, Op>::value_type;
    static constexpr size_t rows = scalar_vector_expression<Scalar, VecExpr, Op>::size;
    static constexpr size_t cols = 1;
    static constexpr bool row_major = true;
    static constexpr bool is_vector = true;
    static constexpr bool is_matrix = false;
};

// Vector traits specialization for scalar-vector expressions
template<typename Scalar, typename VecExpr, typename Op>
struct vector_size<scalar_vector_expression<Scalar, VecExpr, Op>> 
    : std::integral_constant<size_t, scalar_vector_expression<Scalar, VecExpr, Op>::size> {};


template<typename Scalar, typename VecExpr, typename Op>
struct is_vector<scalar_vector_expression<Scalar, VecExpr, Op>> : std::true_type {};

template<typename Scalar, typename VecExpr, typename Op>
struct is_any_vector<scalar_vector_expression<Scalar, VecExpr, Op>> : std::true_type {};

// Operator overloads for scalar / vector
template<typename Scalar, typename VecExpr>
constexpr auto operator/(const Scalar& s, const VecExpr& vec)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && (is_vector_v<VecExpr> || is_any_vector_v<VecExpr>),
        scalar_vector_expression<Scalar, VecExpr, scalar_div_op>
    >
{
    return scalar_vector_expression<Scalar, VecExpr, scalar_div_op>(s, vec);
}

// Operator overloads for scalar - vector
template<typename Scalar, typename VecExpr>
constexpr auto operator-(const Scalar& s, const VecExpr& vec)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && (is_vector_v<VecExpr> || is_any_vector_v<VecExpr>),
        scalar_vector_expression<Scalar, VecExpr, scalar_sub_op>
    >
{
    return scalar_vector_expression<Scalar, VecExpr, scalar_sub_op>(s, vec);
}

// Power operator (using ^ for element-wise power)
template<typename Scalar, typename VecExpr>
constexpr auto operator^(const Scalar& s, const VecExpr& vec)
    -> std::enable_if_t<
        std::is_arithmetic_v<Scalar> && (is_vector_v<VecExpr> || is_any_vector_v<VecExpr>),
        scalar_vector_expression<Scalar, VecExpr, scalar_pow_op>
    >
{
    return scalar_vector_expression<Scalar, VecExpr, scalar_pow_op>(s, vec);
}

// Helper function for element-wise reciprocal
template<typename VecExpr>
constexpr auto reciprocal(const VecExpr& vec)
    -> std::enable_if_t<
        is_vector_v<VecExpr> || is_any_vector_v<VecExpr>,
        scalar_vector_expression<typename VecExpr::value_type, VecExpr, scalar_div_op>
    >
{
    using value_type = typename VecExpr::value_type;
    return value_type(1) / vec;
}

// Element-wise inverse (alias for reciprocal)
template<typename VecExpr>
constexpr auto element_inverse(const VecExpr& vec)
    -> std::enable_if_t<
        is_vector_v<VecExpr> || is_any_vector_v<VecExpr>,
        decltype(reciprocal(vec))
    >
{
    return reciprocal(vec);
}

// Expression storage specializations for scalar-vector expressions
template<typename Scalar, typename VecExpr, typename Op>
struct expression_storage<scalar_vector_expression<Scalar, VecExpr, Op>> {
    using type = scalar_vector_expression<Scalar, VecExpr, Op>;
};

} // namespace euler