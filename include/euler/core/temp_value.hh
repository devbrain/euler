#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace euler {

// Forward declarations
template<typename T, size_t N> class vector;
template<typename T, size_t M, size_t N, bool RowMajor> class matrix;

// =============================================================================
// is_leaf trait - distinguishes concrete types from expression templates
// =============================================================================
//
// Leaf types (vectors, matrices) are stored by REFERENCE in expressions
// for efficiency. Expression templates are stored by VALUE since they're
// lightweight wrappers.
//
// The key insight: rvalue leaves need special handling via temp_holder
// to prevent dangling references. This is done in capture_operand().
// =============================================================================

template<typename T>
struct is_leaf : std::false_type {};

// Vectors are leaves
template<typename T, size_t N>
struct is_leaf<vector<T, N>> : std::true_type {};

// Matrices are leaves
template<typename T, size_t M, size_t N, bool RowMajor>
struct is_leaf<matrix<T, M, N, RowMajor>> : std::true_type {};

template<typename T>
inline constexpr bool is_leaf_v = is_leaf<std::decay_t<T>>::value;

// =============================================================================
// Temporary value holders - own rvalue temporaries to prevent dangling refs
// =============================================================================

template<typename T> class vector_temp_holder;
template<typename T> class matrix_temp_holder;

// Temp holders are NOT leaves - they own data and should be stored by value
// (is_leaf defaults to false, so no specialization needed)

// Holder for temporary vector values
template<typename T>
class vector_temp_holder {
    T value_;

public:
    using value_type = typename T::value_type;
    static constexpr size_t static_size = T::static_size;

    vector_temp_holder(vector_temp_holder&& other) noexcept
        : value_(std::move(other.value_)) {}

    vector_temp_holder(const vector_temp_holder& other)
        : value_(other.value_) {}

    explicit vector_temp_holder(T&& v)
        : value_(std::move(v)) {}

    value_type operator[](size_t i) const {
        return value_[i];
    }

    value_type operator()(size_t row, size_t col) const {
        return value_(row, col);
    }

    value_type eval_scalar(size_t i) const {
        return value_[i];
    }

    value_type eval_scalar(size_t row, size_t col) const {
        return value_(row, col);
    }

    static constexpr size_t size() {
        return static_size;
    }
};

// Holder for temporary matrix values
template<typename T>
class matrix_temp_holder {
    T value_;

public:
    using value_type = typename T::value_type;
    static constexpr size_t rows = T::rows;
    static constexpr size_t cols = T::cols;
    static constexpr size_t static_size = rows * cols;

    matrix_temp_holder(matrix_temp_holder&& other) noexcept
        : value_(std::move(other.value_)) {}

    matrix_temp_holder(const matrix_temp_holder& other)
        : value_(other.value_) {}

    explicit matrix_temp_holder(T&& v)
        : value_(std::move(v)) {}

    value_type operator()(size_t row, size_t col) const {
        return value_(row, col);
    }

    value_type operator[](size_t i) const {
        return value_[i];
    }

    value_type eval_scalar(size_t i) const {
        return value_[i];
    }

    value_type eval_scalar(size_t row, size_t col) const {
        return value_(row, col);
    }

    static constexpr size_t row_count() { return rows; }
    static constexpr size_t col_count() { return cols; }
};

// =============================================================================
// Type dispatch for temporary holders
// =============================================================================

// Detect if a type is a matrix (has rows and cols members)
template<typename T, typename = void>
struct is_matrix_type : std::false_type {};

template<typename T>
struct is_matrix_type<T, std::void_t<decltype(T::rows), decltype(T::cols)>> : std::true_type {};

// Detect if a type is a vector (has static_size but not rows/cols)
template<typename T, typename = void>
struct is_vector_type : std::false_type {};

template<typename T>
struct is_vector_type<T, std::void_t<decltype(T::static_size)>> :
    std::bool_constant<!is_matrix_type<T>::value> {};

// Select appropriate holder type
template<typename T, bool IsMatrix = is_matrix_type<std::decay_t<T>>::value>
struct temp_holder_selector;

template<typename T>
struct temp_holder_selector<T, false> {
    using type = vector_temp_holder<std::decay_t<T>>;
};

template<typename T>
struct temp_holder_selector<T, true> {
    using type = matrix_temp_holder<std::decay_t<T>>;
};

template<typename T>
using temp_holder_t = typename temp_holder_selector<T>::type;

} // namespace euler
