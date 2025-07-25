#pragma once

#include <type_traits>
#include <utility>

namespace euler {

// Forward declarations
template<typename T> class scalar_expression;
template<typename L, typename R, typename Op> class binary_expression;
template<typename E, typename Op> class unary_expression;
template<typename Expr, typename Op> class custom_unary_expr;
template<typename Expr1, typename Expr2, typename Op> class custom_binary_expr;
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;
template<typename T, size_t N> class vector;

// Helper to determine if a type is a concrete matrix/vector
template<typename T>
struct is_concrete_matrix : std::false_type {};

template<typename T, size_t M, size_t N, bool RowMajor>
struct is_concrete_matrix<matrix<T, M, N, RowMajor>> : std::true_type {};

template<typename T, size_t N>
struct is_concrete_matrix<vector<T, N>> : std::true_type {};

// Storage wrapper that can hold either a value or a reference
template<typename T>
class storage_wrapper {
private:
    static constexpr bool store_by_value = 
        is_concrete_matrix<std::decay_t<T>>::value && !std::is_lvalue_reference_v<T>;
    
    using storage_type = std::conditional_t<store_by_value, std::decay_t<T>, const T&>;
    storage_type data;
    
public:
    template<typename U>
    storage_wrapper(U&& u) : data(std::forward<U>(u)) {}
    
    const auto& get() const { return data; }
    auto& get() { return data; }
};

// New expression storage trait that uses the wrapper
template<typename T>
struct expression_storage_v2 {
    using type = storage_wrapper<T>;
};

// For already-wrapped types, don't double-wrap
template<typename T>
struct expression_storage_v2<storage_wrapper<T>> {
    using type = storage_wrapper<T>;
};

} // namespace euler