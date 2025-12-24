#pragma once

#include <euler/vector/vector.hh>
#include <euler/matrix/matrix_view.hh>
#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/vector/vector_traits.hh>
#include <euler/vector/vector_expr.hh>
#include <euler/vector/scalar_vector_expr.hh>
#include <cmath>
#include <utility>
#include <tuple>
#include <type_traits>
#ifdef EULER_HAS_SIMD
#include <immintrin.h>
#endif

namespace euler {

// Helper to check if a matrix_view is vector-like at runtime
template<typename T>
bool is_vector_view(const matrix_view<T>& view) {
    return view.rows() == 1 || view.cols() == 1;
}

template<typename T>
bool is_vector_view(const const_matrix_view<T>& view) {
    return view.rows() == 1 || view.cols() == 1;
}

// Get effective size of a vector-like view
template<typename T>
size_t vector_view_size(const matrix_view<T>& view) {
    return view.rows() == 1 ? view.cols() : view.rows();
}

template<typename T>
size_t vector_view_size(const const_matrix_view<T>& view) {
    return view.rows() == 1 ? view.cols() : view.rows();
}

// is_expression is already defined in core/traits.hh

// Cross product - returns expression for expressions, concrete result for concrete types
template<typename T>
auto cross(const vector<T, 3>& a, const vector<T, 3>& b) {
    return cross_product_expr<vector<T, 3>, vector<T, 3>>(a, b);
}

// Cross product for any expressions
template<typename Expr1, typename Expr2>
auto cross(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    cross_product_expr<Expr1, Expr2>
> {
    return cross_product_expr<Expr1, Expr2>(a, b);
}

// Cross product for expression and vector
template<typename Expr, typename T>
auto cross(const Expr& a, const vector<T, 3>& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr>,
    cross_product_expr<Expr, vector<T, 3>>
> {
    return cross_product_expr<Expr, vector<T, 3>>(a, b);
}

// Cross product for vector and expression
template<typename T, typename Expr>
auto cross(const vector<T, 3>& a, const Expr& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr>,
    cross_product_expr<vector<T, 3>, Expr>
> {
    return cross_product_expr<vector<T, 3>, Expr>(a, b);
}

// Overload for when we explicitly want immediate evaluation
template<typename T>
constexpr vector<T, 3> cross_eval(const vector<T, 3>& a, const vector<T, 3>& b) {
    return vector<T, 3>(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()
    );
}

// Normalize - returns expression for vector types
template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec> && !is_expression_v<Vec>>>
auto normalize(const Vec& v) {
    constexpr size_t Size = expression_vector_size_v<Vec>;
    return normalized_expr<Vec, Size>(v);
}

// Normalize for expressions - compose expressions
template<typename Expr,
         typename = std::enable_if_t<is_expression_v<Expr>>>
auto normalize(const expression<Expr, typename Expr::value_type>& expr) {
    constexpr size_t Size = expression_vector_size_v<Expr>;
    return normalized_expr<Expr, Size>(static_cast<const Expr&>(expr));
}

// Fast normalize for vec3 using SIMD when possible
template<typename T>
vector<T, 3> fast_normalize(const vector<T, 3>& v) {
#ifdef EULER_HAS_SIMD
    if constexpr (std::is_same_v<T, float>) {
        // Use SSE intrinsics for float
        __m128 vec = _mm_setr_ps(v[0], v[1], v[2], 0.0f);
        __m128 dot = _mm_mul_ps(vec, vec);
        
        // Horizontal sum: dot = [x²+y²+z²+0, x²+y²+z²+0, ...]
        __m128 shuf = _mm_shuffle_ps(dot, dot, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(dot, shuf);
        shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
        sums = _mm_add_ps(sums, shuf);
        
        // Fast inverse square root
        __m128 rsqrt = _mm_rsqrt_ps(sums);
        
        // Newton-Raphson refinement for better accuracy
        // rsqrt = rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three_halves = _mm_set1_ps(1.5f);
        __m128 rsqrt_sq = _mm_mul_ps(rsqrt, rsqrt);
        __m128 half_x = _mm_mul_ps(half, sums);
        __m128 nr = _mm_mul_ps(half_x, rsqrt_sq);
        nr = _mm_sub_ps(three_halves, nr);
        rsqrt = _mm_mul_ps(rsqrt, nr);
        
        // Multiply vector by inverse length
        vec = _mm_mul_ps(vec, rsqrt);
        
        alignas(16) float result[4];
        _mm_store_ps(result, vec);
        return vector<T, 3>(result[0], result[1], result[2]);
    } else if constexpr (std::is_same_v<T, double>) {
        // Use SSE2/AVX intrinsics for double
        __m256d vec = _mm256_setr_pd(v[0], v[1], v[2], 0.0);
        __m256d dot = _mm256_mul_pd(vec, vec);
        
        // Horizontal sum
        __m256d temp = _mm256_hadd_pd(dot, dot);
        __m128d low = _mm256_castpd256_pd128(temp);
        __m128d high = _mm256_extractf128_pd(temp, 1);
        __m128d sum = _mm_add_sd(low, high);
        
        // sqrt and reciprocal
        __m128d sqrt_sum = _mm_sqrt_sd(sum, sum);
        __m128d inv_len = _mm_div_sd(_mm_set_sd(1.0), sqrt_sum);
        
        // Broadcast inv_len to all elements
        __m256d inv_len_vec = _mm256_broadcast_sd(&inv_len[0]);
        
        // Multiply vector by inverse length
        vec = _mm256_mul_pd(vec, inv_len_vec);
        
        alignas(32) double result[4];
        _mm256_store_pd(result, vec);
        return vector<T, 3>(result[0], result[1], result[2]);
    } else
#endif
    {
        // Fallback to standard normalization
        T len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        T inv_len = T(1) / std::sqrt(len_sq);
        return vector<T, 3>(v[0] * inv_len, v[1] * inv_len, v[2] * inv_len);
    }
}

// Normalize for any expression-like type (including binary expressions)
template<typename Expr>
auto normalize(const Expr& expr) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr>,
    normalized_expr<Expr, expression_vector_size_v<Expr>>
> {
    constexpr size_t Size = expression_vector_size_v<Expr>;
    return normalized_expr<Expr, Size>(expr);
}

// Reflect - returns expression
template<typename Vec, typename Normal,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_any_vector_v<Normal>>>
auto reflect(const Vec& incident, const Normal& normal) {
    return reflect_expr<Vec, Normal>(incident, normal);
}

// Reflect for expressions
template<typename Expr1, typename Expr2>
auto reflect(const Expr1& incident, const Expr2& normal) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> && 
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    reflect_expr<Expr1, Expr2>
> {
    return reflect_expr<Expr1, Expr2>(incident, normal);
}

// Reflect for expression and vector
template<typename Expr, typename Vec>
auto reflect(const Expr& incident, const Vec& normal) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    reflect_expr<Expr, Vec>
> {
    return reflect_expr<Expr, Vec>(incident, normal);
}

// Reflect for vector and expression
template<typename Vec, typename Expr>
auto reflect(const Vec& incident, const Expr& normal) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    reflect_expr<Vec, Expr>
> {
    return reflect_expr<Vec, Expr>(incident, normal);
}

// Lerp - returns expression
template<typename Vec1, typename Vec2, typename T,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2>>>
auto lerp(const Vec1& a, const Vec2& b, T t) {
    return lerp_expr<Vec1, Vec2, T>(a, b, t);
}

// Lerp for any expressions
template<typename Expr1, typename Expr2, typename T>
auto lerp(const Expr1& a, const Expr2& b, T t) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    lerp_expr<Expr1, Expr2, T>
> {
    return lerp_expr<Expr1, Expr2, T>(a, b, t);
}

// Lerp for expression and vector
template<typename Expr, typename Vec, typename T>
auto lerp(const Expr& a, const Vec& b, T t) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    lerp_expr<Expr, Vec, T>
> {
    return lerp_expr<Expr, Vec, T>(a, b, t);
}

// Lerp for vector and expression
template<typename Vec, typename Expr, typename T>
auto lerp(const Vec& a, const Expr& b, T t) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    lerp_expr<Vec, Expr, T>
> {
    return lerp_expr<Vec, Expr, T>(a, b, t);
}

// Dot product - always immediate evaluation (returns scalar)
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2> &&
                                    vector_size_helper<Vec1>::value == vector_size_helper<Vec2>::value>>
constexpr auto dot(const Vec1& a, const Vec2& b) -> typename Vec1::value_type {
    using T = typename Vec1::value_type;
    constexpr size_t N = vector_size_helper<Vec1>::value;
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Helper to detect if a type is a vector expression (has eval_scalar and is vector-sized)
template<typename T>
struct is_vector_expression {
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().eval_scalar(size_t{}),
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test(...);
    
    static constexpr bool value = decltype(test<T>(0))::value && 
                                  !is_any_vector_v<T> && 
                                  !std::is_same_v<T, matrix_view<typename T::value_type>> &&
                                  !std::is_same_v<T, const_matrix_view<typename T::value_type>>;
};

template<typename T>
constexpr bool is_vector_expression_v = is_vector_expression<T>::value;

// Dot product for any expression-like types (including matrix expressions used as vectors)
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_vector_expression_v<Expr1> && is_vector_expression_v<Expr2>>>
auto dot(const Expr1& a, const Expr2& b) -> decltype(a.eval_scalar(0)) {
    using T = decltype(a.eval_scalar(0));
    
    // Get the dimension from expression traits
    constexpr size_t N = expression_vector_size_v<Expr1>;
    static_assert(expression_vector_size_v<Expr1> == expression_vector_size_v<Expr2>,
                  "Vector dimensions must match for dot product");
    
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += a.eval_scalar(i) * b.eval_scalar(i);
    }
    return result;
}

// Dot product for expression and vector
template<typename Expr, typename Vec,
         typename = std::enable_if_t<is_vector_expression_v<Expr> && is_any_vector_v<Vec>>,
         typename = void>  // Extra parameter to avoid ambiguity
auto dot(const Expr& a, const Vec& b) -> decltype(a.eval_scalar(0)) {
    using T = decltype(a.eval_scalar(0));
    const size_t N = vector_size_helper<Vec>::value;
    
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += a.eval_scalar(i) * b[i];
    }
    return result;
}

// Dot product for vector and expression
template<typename Vec, typename Expr,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_vector_expression_v<Expr>>,
         typename = void, typename = void>  // Extra parameters to avoid ambiguity
auto dot(const Vec& a, const Expr& b) -> typename Vec::value_type {
    return dot(b, a);
}

// Dot product for matrix_view
template<typename T>
T dot(const matrix_view<T>& a, const matrix_view<T>& b) {
    EULER_CHECK(is_vector_view(a) && is_vector_view(b), error_code::dimension_mismatch,
               "Both matrix views must be row or column vectors");
    const size_t n = vector_view_size(a);
    EULER_CHECK(n == vector_view_size(b), error_code::dimension_mismatch,
               "Vector dimensions must match");
    
    T result = T(0);
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Dot product for matrix_view and vector
template<typename T, typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
T dot(const matrix_view<T>& view, const Vec& vec) {
    EULER_CHECK(is_vector_view(view), error_code::dimension_mismatch,
               "Matrix view must be a row or column vector");
    const size_t n = vector_view_size(view);
    EULER_CHECK(n == vector_size_helper<Vec>::value, error_code::dimension_mismatch,
               "Vector dimensions must match");
    
    T result = T(0);
    for (size_t i = 0; i < n; ++i) {
        result += view[i] * vec[i];
    }
    return result;
}

// Dot product for vector and matrix_view
template<typename Vec, typename T,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
T dot(const Vec& vec, const matrix_view<T>& view) {
    return dot(view, vec);
}

// Length operations - immediate evaluation
template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto length_squared(const Vec& v) -> typename Vec::value_type {
    return dot(v, v);
}

// Length squared for expressions
template<typename Expr,
         typename = std::enable_if_t<is_vector_expression_v<Expr>>>
auto length_squared(const Expr& v) -> decltype(v.eval_scalar(0)) {
    return dot(v, v);
}

template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto length(const Vec& v) -> typename Vec::value_type {
    return std::sqrt(length_squared(v));
}

// Length for expressions
template<typename Expr,
         typename = std::enable_if_t<is_vector_expression_v<Expr>>>
auto length(const Expr& v) -> decltype(v.eval_scalar(0)) {
    return std::sqrt(length_squared(v));
}

// Distance operations
template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto distance(const Vec& a, const Vec& b) -> typename Vec::value_type {
    return length(b - a);
}

// Distance for expressions (both expressions)
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_expression_v<Expr1> && is_expression_v<Expr2>>>
auto distance(const expression<Expr1, typename Expr1::value_type>& a,
              const expression<Expr2, typename Expr2::value_type>& b) -> typename Expr1::value_type {
    return length(static_cast<const Expr2&>(b) - static_cast<const Expr1&>(a));
}

// Distance for mixed vector/expression cases
template<typename Vec, typename Expr,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_expression_v<Expr>>>
auto distance(const Vec& a, const expression<Expr, typename Expr::value_type>& b) -> typename Vec::value_type {
    return length(static_cast<const Expr&>(b) - a);
}

template<typename Expr, typename Vec,
         typename = std::enable_if_t<is_expression_v<Expr> && is_any_vector_v<Vec>>>
auto distance(const expression<Expr, typename Expr::value_type>& a, const Vec& b) -> typename Expr::value_type {
    return length(b - static_cast<const Expr&>(a));
}

template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto distance_squared(const Vec& a, const Vec& b) -> typename Vec::value_type {
    return length_squared(b - a);
}

// Distance squared for expressions (both expressions)
template<typename Expr1, typename Expr2,
         typename = std::enable_if_t<is_expression_v<Expr1> && is_expression_v<Expr2>>>
auto distance_squared(const expression<Expr1, typename Expr1::value_type>& a,
                      const expression<Expr2, typename Expr2::value_type>& b) -> typename Expr1::value_type {
    return length_squared(static_cast<const Expr2&>(b) - static_cast<const Expr1&>(a));
}

// Distance squared for mixed vector/expression cases
template<typename Vec, typename Expr,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_expression_v<Expr>>>
auto distance_squared(const Vec& a, const expression<Expr, typename Expr::value_type>& b) -> typename Vec::value_type {
    return length_squared(static_cast<const Expr&>(b) - a);
}

template<typename Expr, typename Vec,
         typename = std::enable_if_t<is_expression_v<Expr> && is_any_vector_v<Vec>>>
auto distance_squared(const expression<Expr, typename Expr::value_type>& a, const Vec& b) -> typename Expr::value_type {
    return length_squared(b - static_cast<const Expr&>(a));
}

// Component-wise min - returns expression
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2> &&
                                    vector_size_helper<Vec1>::value == vector_size_helper<Vec2>::value>>
auto min(const Vec1& a, const Vec2& b) {
    return min_expr<Vec1, Vec2>(a, b);
}

// Component-wise min for any expressions
template<typename Expr1, typename Expr2>
auto min(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    min_expr<Expr1, Expr2>
> {
    return min_expr<Expr1, Expr2>(a, b);
}

// Min for expression and vector
template<typename Expr, typename Vec>
auto min(const Expr& a, const Vec& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    min_expr<Expr, Vec>
> {
    return min_expr<Expr, Vec>(a, b);
}

// Min for vector and expression
template<typename Vec, typename Expr>
auto min(const Vec& a, const Expr& b) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    min_expr<Vec, Expr>
> {
    return min_expr<Vec, Expr>(a, b);
}

// Component-wise max - returns expression
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2> &&
                                    vector_size_helper<Vec1>::value == vector_size_helper<Vec2>::value>>
auto max(const Vec1& a, const Vec2& b) {
    return max_expr<Vec1, Vec2>(a, b);
}

// Component-wise max for any expressions
template<typename Expr1, typename Expr2>
auto max(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    max_expr<Expr1, Expr2>
> {
    return max_expr<Expr1, Expr2>(a, b);
}

// Max for expression and vector
template<typename Expr, typename Vec>
auto max(const Expr& a, const Vec& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    max_expr<Expr, Vec>
> {
    return max_expr<Expr, Vec>(a, b);
}

// Max for vector and expression
template<typename Vec, typename Expr>
auto max(const Vec& a, const Expr& b) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    max_expr<Vec, Expr>
> {
    return max_expr<Vec, Expr>(a, b);
}

// Component-wise abs - returns expression
template<typename Vec,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto abs(const Vec& v) {
    return abs_expr<Vec>(v);
}

// Component-wise abs for any expression
template<typename Expr>
auto abs(const Expr& expr) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr>,
    abs_expr<Expr>
> {
    return abs_expr<Expr>(expr);
}

// Clamp operations - returns expression
template<typename Vec, typename Min, typename Max,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto clamp(const Vec& v, const Min& min_val, const Max& max_val) {
    return clamp_expr<Vec, Min, Max>(v, min_val, max_val);
}

// Clamp for any expression
template<typename Expr, typename Min, typename Max>
auto clamp(const Expr& expr, const Min& min_val, const Max& max_val) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr>,
    clamp_expr<Expr, Min, Max>
> {
    return clamp_expr<Expr, Min, Max>(expr, min_val, max_val);
}

// Angle between vectors - immediate evaluation
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2>>>
auto angle_between(const Vec1& a, const Vec2& b) -> typename Vec1::value_type {
    using T = typename Vec1::value_type;
    T d = dot(a, b);
    T len_a = length(a);
    T len_b = length(b);
    
    EULER_CHECK(len_a > T(0) && len_b > T(0), error_code::invalid_argument,
               "Cannot compute angle with zero-length vector");
    
    // Clamp to handle numerical errors
    T cos_angle = std::max(T(-1), std::min(T(1), d / (len_a * len_b)));
    return std::acos(cos_angle);
}

// Angle for any expressions
template<typename Expr1, typename Expr2>
auto angle_between(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    typename Expr1::value_type
> {
    using T = typename Expr1::value_type;
    T d = dot(a, b);
    T len_a = length(a);
    T len_b = length(b);
    
    EULER_CHECK(len_a > T(0) && len_b > T(0), error_code::invalid_argument,
               "Cannot compute angle with zero-length vector");
    
    T cos_angle = std::max(T(-1), std::min(T(1), d / (len_a * len_b)));
    return std::acos(cos_angle);
}

// Angle for expression and vector
template<typename Expr, typename Vec>
auto angle_between(const Expr& a, const Vec& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    typename Expr::value_type
> {
    using T = typename Expr::value_type;
    T d = dot(a, b);
    T len_a = length(a);
    T len_b = length(b);
    
    EULER_CHECK(len_a > T(0) && len_b > T(0), error_code::invalid_argument,
               "Cannot compute angle with zero-length vector");
    
    T cos_angle = std::max(T(-1), std::min(T(1), d / (len_a * len_b)));
    return std::acos(cos_angle);
}

// Angle for vector and expression
template<typename Vec, typename Expr>
auto angle_between(const Vec& a, const Expr& b) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    typename Vec::value_type
> {
    return angle_between(b, a);
}

// Project - returns expression
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2>>>
auto project(const Vec1& a, const Vec2& b) {
    using T = typename Vec1::value_type;
    [[maybe_unused]] T b_len_sq = length_squared(b);
    EULER_CHECK(b_len_sq > T(0), error_code::invalid_argument,
               "Cannot project onto zero-length vector");
    return project_expr<Vec1, Vec2>(a, b);
}

// Project for any expressions
template<typename Expr1, typename Expr2>
auto project(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    project_expr<Expr1, Expr2>
> {
    using T = typename Expr1::value_type;
    [[maybe_unused]] T b_len_sq = length_squared(b);
    EULER_CHECK(b_len_sq > T(0), error_code::invalid_argument,
               "Cannot project onto zero-length vector");
    return project_expr<Expr1, Expr2>(a, b);
}

// Project for expression and vector
template<typename Expr, typename Vec>
auto project(const Expr& a, const Vec& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    project_expr<Expr, Vec>
> {
    using T = typename Expr::value_type;
    [[maybe_unused]] T b_len_sq = length_squared(b);
    EULER_CHECK(b_len_sq > T(0), error_code::invalid_argument,
               "Cannot project onto zero-length vector");
    return project_expr<Expr, Vec>(a, b);
}

// Project for vector and expression
template<typename Vec, typename Expr>
auto project(const Vec& a, const Expr& b) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    project_expr<Vec, Expr>
> {
    using T = typename Vec::value_type;
    [[maybe_unused]] T b_len_sq = length_squared(b);
    EULER_CHECK(b_len_sq > T(0), error_code::invalid_argument,
               "Cannot project onto zero-length vector");
    return project_expr<Vec, Expr>(a, b);
}

// Reject 
template<typename Vec1, typename Vec2,
         typename = std::enable_if_t<is_any_vector_v<Vec1> && is_any_vector_v<Vec2>>>
auto reject(const Vec1& a, const Vec2& b) {
    return a - project(a, b);
}

// Reject for any expressions
template<typename Expr1, typename Expr2>
auto reject(const Expr1& a, const Expr2& b) -> std::enable_if_t<
    is_expression_v<Expr1> && is_expression_v<Expr2> &&
    !is_any_vector_v<Expr1> && !is_any_vector_v<Expr2>,
    decltype(a - project(a, b))
> {
    return a - project(a, b);
}

// Reject for expression and vector
template<typename Expr, typename Vec>
auto reject(const Expr& a, const Vec& b) -> std::enable_if_t<
    is_expression_v<Expr> && !is_any_vector_v<Expr> && is_any_vector_v<Vec>,
    decltype(a - project(a, b))
> {
    return a - project(a, b);
}

// Reject for vector and expression
template<typename Vec, typename Expr>
auto reject(const Vec& a, const Expr& b) -> std::enable_if_t<
    is_any_vector_v<Vec> && is_expression_v<Expr> && !is_any_vector_v<Expr>,
    decltype(a - project(a, b))
> {
    return a - project(a, b);
}

// Face forward - returns expression
template<typename Vec, typename Incident, typename Normal,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_any_vector_v<Incident> && is_any_vector_v<Normal>>>
auto faceforward(const Vec& n, const Incident& i, const Normal& nref) {
    return faceforward_expr<Vec, Incident, Normal>(n, i, nref);
}

// Face forward for any expression combinations
template<typename N, typename I, typename NRef>
auto faceforward(const N& n, const I& i, const NRef& nref) -> std::enable_if_t<
    ((is_expression_v<N> && !is_any_vector_v<N>) || 
     (is_expression_v<I> && !is_any_vector_v<I>) || 
     (is_expression_v<NRef> && !is_any_vector_v<NRef>)),
    faceforward_expr<N, I, NRef>
> {
    return faceforward_expr<N, I, NRef>(n, i, nref);
}

// Refraction
template<typename Vec, typename Normal, typename T,
         typename = std::enable_if_t<is_any_vector_v<Vec> && is_any_vector_v<Normal>>>
auto refract(const Vec& incident, const Normal& normal, T eta) {
    T n_dot_i = dot(normal, incident);
    T k = T(1) - eta * eta * (T(1) - n_dot_i * n_dot_i);
    
    if (k < T(0)) {
        // Total internal reflection
        return Vec::zero();
    }
    
    // Force evaluation to ensure consistent return type
    Vec result = eta * incident - (eta * n_dot_i + std::sqrt(k)) * normal;
    return result;
}

// Smoothstep
template<typename Vec, typename Edge0, typename Edge1,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
auto smoothstep(const Edge0& edge0, const Edge1& edge1, const Vec& x) {
    auto t = clamp((x - edge0) / (edge1 - edge0), typename Vec::value_type(0), typename Vec::value_type(1));
    return t * t * (typename Vec::value_type(3) - typename Vec::value_type(2) * t);
}

// Utility functions
template<typename Vec, typename T = typename Vec::value_type,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
bool approx_equal(const Vec& a, const Vec& b, T eps = constants<T>::epsilon) {
    // Get the vector dimension
    constexpr size_t dim = expression_vector_size_v<Vec>;
    
    // Compare element by element
    for (size_t i = 0; i < dim; ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

template<typename Vec, typename T = typename Vec::value_type,
         typename = std::enable_if_t<is_any_vector_v<Vec>>>
bool approx_zero(const Vec& v, T eps = constants<T>::epsilon) {
    return length_squared(v) < eps * eps;
}

// 2D cross product (returns scalar)
template<typename T>
T cross(const vector<T, 2>& a, const vector<T, 2>& b) {
    return a.x() * b.y() - a.y() * b.x();
}

// Orthonormalization using Gram-Schmidt for 2D vectors
template<typename T>
void orthonormalize(vector<T, 2>& v0, vector<T, 2>& v1) {
    v0 = normalize(v0);
    v1 = normalize(v1 - project(v1, v0));
}

// Orthonormalization for 2D with expressions
template<typename Vec0, typename Vec1>
auto orthonormalize(const Vec0& v0, const Vec1& v1) 
    -> std::enable_if_t<
        expression_vector_size_v<Vec0> == 2 && 
        expression_vector_size_v<Vec1> == 2 &&
        (is_any_vector_v<Vec0> || is_expression_v<Vec0>) &&
        (is_any_vector_v<Vec1> || is_expression_v<Vec1>) &&
        (!std::is_same_v<Vec0, vector<typename Vec0::value_type, 2>> || 
         !std::is_same_v<Vec1, vector<typename Vec1::value_type, 2>>),
        std::pair<vector<typename Vec0::value_type, 2>, vector<typename Vec0::value_type, 2>>
    >
{
    using T = typename Vec0::value_type;
    vector<T, 2> r0 = normalize(v0);
    vector<T, 2> r1 = normalize(v1 - project(v1, r0));
    return std::make_pair(r0, r1);
}

// Orthonormalization using Gram-Schmidt for 3D vectors
template<typename T>
void orthonormalize(vector<T, 3>& v0, vector<T, 3>& v1, vector<T, 3>& v2) {
    v0 = normalize(v0);
    v1 = normalize(v1 - project(v1, v0));
    v2 = normalize(v2 - project(v2, v0) - project(v2, v1));
}

// Orthonormalization for 3D with expressions
template<typename Vec0, typename Vec1, typename Vec2>
auto orthonormalize(const Vec0& v0, const Vec1& v1, const Vec2& v2)
    -> std::enable_if_t<
        expression_vector_size_v<Vec0> == 3 && 
        expression_vector_size_v<Vec1> == 3 &&
        expression_vector_size_v<Vec2> == 3 &&
        (is_any_vector_v<Vec0> || is_expression_v<Vec0>) &&
        (is_any_vector_v<Vec1> || is_expression_v<Vec1>) &&
        (is_any_vector_v<Vec2> || is_expression_v<Vec2>) &&
        (!std::is_same_v<Vec0, vector<typename Vec0::value_type, 3>> || 
         !std::is_same_v<Vec1, vector<typename Vec1::value_type, 3>> ||
         !std::is_same_v<Vec2, vector<typename Vec2::value_type, 3>>),
        std::tuple<vector<typename Vec0::value_type, 3>, vector<typename Vec0::value_type, 3>, vector<typename Vec0::value_type, 3>>
    >
{
    using T = typename Vec0::value_type;
    vector<T, 3> r0 = normalize(v0);
    vector<T, 3> r1 = normalize(v1 - project(v1, r0));
    vector<T, 3> r2 = normalize(v2 - project(v2, r0) - project(v2, r1));
    return std::make_tuple(r0, r1, r2);
}

// Orthonormalization using Gram-Schmidt for 4D vectors
template<typename T>
void orthonormalize(vector<T, 4>& v0, vector<T, 4>& v1, vector<T, 4>& v2, vector<T, 4>& v3) {
    v0 = normalize(v0);
    v1 = normalize(v1 - project(v1, v0));
    v2 = normalize(v2 - project(v2, v0) - project(v2, v1));
    v3 = normalize(v3 - project(v3, v0) - project(v3, v1) - project(v3, v2));
}

// Orthonormalization for 4D with expressions
template<typename Vec0, typename Vec1, typename Vec2, typename Vec3>
auto orthonormalize(const Vec0& v0, const Vec1& v1, const Vec2& v2, const Vec3& v3)
    -> std::enable_if_t<
        expression_vector_size_v<Vec0> == 4 && 
        expression_vector_size_v<Vec1> == 4 &&
        expression_vector_size_v<Vec2> == 4 &&
        expression_vector_size_v<Vec3> == 4 &&
        (is_any_vector_v<Vec0> || is_expression_v<Vec0>) &&
        (is_any_vector_v<Vec1> || is_expression_v<Vec1>) &&
        (is_any_vector_v<Vec2> || is_expression_v<Vec2>) &&
        (is_any_vector_v<Vec3> || is_expression_v<Vec3>) &&
        (!std::is_same_v<Vec0, vector<typename Vec0::value_type, 4>> || 
         !std::is_same_v<Vec1, vector<typename Vec1::value_type, 4>> ||
         !std::is_same_v<Vec2, vector<typename Vec2::value_type, 4>> ||
         !std::is_same_v<Vec3, vector<typename Vec3::value_type, 4>>),
        std::tuple<vector<typename Vec0::value_type, 4>, vector<typename Vec0::value_type, 4>, vector<typename Vec0::value_type, 4>, vector<typename Vec0::value_type, 4>>
    >
{
    using T = typename Vec0::value_type;
    vector<T, 4> r0 = normalize(v0);
    vector<T, 4> r1 = normalize(v1 - project(v1, r0));
    vector<T, 4> r2 = normalize(v2 - project(v2, r0) - project(v2, r1));
    vector<T, 4> r3 = normalize(v3 - project(v3, r0) - project(v3, r1) - project(v3, r2));
    return std::make_tuple(r0, r1, r2, r3);
}

// Build orthonormal basis from single 2D vector
template<typename T>
void build_orthonormal_basis(const vector<T, 2>& n, vector<T, 2>& t) {
    // For 2D, the tangent is just the perpendicular vector
    t = vector<T, 2>(-n.y(), n.x());
}

// Build orthonormal basis from single vector with expressions - 2D version
template<typename Vec>
auto build_orthonormal_basis(const Vec& n) 
    -> std::enable_if_t<
        expression_vector_size_v<Vec> == 2,
        std::pair<vector<typename Vec::value_type, 2>, vector<typename Vec::value_type, 2>>
    >
{
    using T = typename Vec::value_type;
    vector<T, 2> normal = n;  // Assume already normalized if from expression
    T len = length(normal);
    if (len > constants<T>::epsilon) {
        normal = normal / len;
    }
    vector<T, 2> tangent(-normal.y(), normal.x());
    return std::make_pair(normal, tangent);
}

// Build orthonormal basis from single 3D vector
template<typename T>
void build_orthonormal_basis(const vector<T, 3>& n, vector<T, 3>& t, vector<T, 3>& b) {
    // Choose t perpendicular to n
    if (std::abs(n.x()) < T(0.9)) {
        t = cross(n, vector<T, 3>::unit_x());
    } else {
        t = cross(n, vector<T, 3>::unit_y());
    }
    t = normalize(t);
    b = cross(n, t);
}

// Build orthonormal basis from single vector with expressions - 3D version
template<typename Vec>
auto build_orthonormal_basis(const Vec& n) 
    -> std::enable_if_t<
        expression_vector_size_v<Vec> == 3,
        std::tuple<vector<typename Vec::value_type, 3>, vector<typename Vec::value_type, 3>, vector<typename Vec::value_type, 3>>
    >
{
    using T = typename Vec::value_type;
    vector<T, 3> normal = n;  // Assume already normalized if from expression
    T len = length(normal);
    if (len > constants<T>::epsilon) {
        normal = normal / len;
    }
    
    vector<T, 3> tangent;
    if (std::abs(normal.x()) < T(0.9)) {
        tangent = cross(normal, vector<T, 3>::unit_x());
    } else {
        tangent = cross(normal, vector<T, 3>::unit_y());
    }
    tangent = normalize(tangent);
    vector<T, 3> bitangent = cross(normal, tangent);
    
    return std::make_tuple(normal, tangent, bitangent);
}

// Build orthonormal basis from single 4D vector
template<typename T>
void build_orthonormal_basis(const vector<T, 4>& n, vector<T, 4>& t, vector<T, 4>& b, vector<T, 4>& c) {
    // For 4D, we need to find 3 orthogonal vectors to n
    // Start with standard basis vectors and orthogonalize
    vector<T, 4> candidates[4] = {
        vector<T, 4>::unit_x(),
        vector<T, 4>::unit_y(),
        vector<T, 4>::unit_z(),
        vector<T, 4>::unit_w()
    };
    
    // Find the candidate most orthogonal to n
    int best_idx = 0;
    T min_dot = std::abs(dot(n, candidates[0]));
    for (int i = 1; i < 4; ++i) {
        T d = std::abs(dot(n, candidates[i]));
        if (d < min_dot) {
            min_dot = d;
            best_idx = i;
        }
    }
    
    // Use Gram-Schmidt to build the rest
    t = normalize(candidates[best_idx] - project(candidates[best_idx], n));
    
    // Find next best candidate
    int next_idx = (best_idx + 1) % 4;
    min_dot = std::abs(dot(candidates[next_idx], n)) + std::abs(dot(candidates[next_idx], t));
    for (int i = 0; i < 4; ++i) {
        if (i != best_idx) {
            T d = std::abs(dot(candidates[i], n)) + std::abs(dot(candidates[i], t));
            if (d < min_dot) {
                min_dot = d;
                next_idx = i;
            }
        }
    }
    
    b = normalize(candidates[next_idx] - project(candidates[next_idx], n) - project(candidates[next_idx], t));
    
    // The fourth vector - find the remaining one
    for (int i = 0; i < 4; ++i) {
        if (i != best_idx && i != next_idx) {
            vector<T, 4> temp = candidates[i] - project(candidates[i], n) - project(candidates[i], t) - project(candidates[i], b);
            T temp_len = length(temp);
            if (temp_len > constants<T>::epsilon) {  // Check if not degenerate
                c = temp / temp_len;
            } else {
                // Fallback: use a different candidate
                for (int j = 0; j < 4; ++j) {
                    if (j != i && j != best_idx && j != next_idx) {
                        c = normalize(candidates[j] - project(candidates[j], n) - project(candidates[j], t) - project(candidates[j], b));
                        break;
                    }
                }
            }
            break;
        }
    }
}

// Build orthonormal basis from single vector with expressions - 4D version
template<typename Vec>
auto build_orthonormal_basis(const Vec& n) 
    -> std::enable_if_t<
        expression_vector_size_v<Vec> == 4,
        std::tuple<vector<typename Vec::value_type, 4>, vector<typename Vec::value_type, 4>, 
                   vector<typename Vec::value_type, 4>, vector<typename Vec::value_type, 4>>
    >
{
    using T = typename Vec::value_type;
    vector<T, 4> normal = n;  // Assume already normalized if from expression
    T len = length(normal);
    if (len > constants<T>::epsilon) {
        normal = normal / len;
    }
    
    vector<T, 4> candidates[4] = {
        vector<T, 4>::unit_x(),
        vector<T, 4>::unit_y(),
        vector<T, 4>::unit_z(),
        vector<T, 4>::unit_w()
    };
    
    // Find the candidate most orthogonal to normal
    int best_idx = 0;
    T min_dot = std::abs(dot(normal, candidates[0]));
    for (int i = 1; i < 4; ++i) {
        T d = std::abs(dot(normal, candidates[i]));
        if (d < min_dot) {
            min_dot = d;
            best_idx = i;
        }
    }
    
    vector<T, 4> tangent = normalize(candidates[best_idx] - project(candidates[best_idx], normal));
    
    // Find next best candidate
    int next_idx = (best_idx + 1) % 4;
    min_dot = std::abs(dot(candidates[next_idx], normal)) + std::abs(dot(candidates[next_idx], tangent));
    for (int i = 0; i < 4; ++i) {
        if (i != best_idx) {
            T d = std::abs(dot(candidates[i], normal)) + std::abs(dot(candidates[i], tangent));
            if (d < min_dot) {
                min_dot = d;
                next_idx = i;
            }
        }
    }
    
    vector<T, 4> bitangent = normalize(candidates[next_idx] - project(candidates[next_idx], normal) - project(candidates[next_idx], tangent));
    
    // Find the fourth vector
    vector<T, 4> tritangent;
    for (int i = 0; i < 4; ++i) {
        if (i != best_idx && i != next_idx) {
            vector<T, 4> temp = candidates[i] - project(candidates[i], normal) - project(candidates[i], tangent) - project(candidates[i], bitangent);
            T temp_len = length(temp);
            if (temp_len > constants<T>::epsilon) {  // Check if not degenerate
                tritangent = temp / temp_len;
            } else {
                // Fallback: use a different candidate
                for (int j = 0; j < 4; ++j) {
                    if (j != i && j != best_idx && j != next_idx) {
                        tritangent = normalize(candidates[j] - project(candidates[j], normal) - project(candidates[j], tangent) - project(candidates[j], bitangent));
                        break;
                    }
                }
            }
            break;
        }
    }
    
    return std::make_tuple(normal, tangent, bitangent, tritangent);
}

} // namespace euler