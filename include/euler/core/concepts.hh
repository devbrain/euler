#pragma once

#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <concepts>
#include <type_traits>

namespace euler {

// Basic arithmetic type concepts
template<typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;

template<typename T>
concept scalar_type = std::is_arithmetic_v<T>;

// Expression concepts
template<typename T>
concept vector_expression = requires(const T& t) {
    typename T::value_type;
    { t.eval_scalar(size_t{}) } -> std::convertible_to<typename T::value_type>;
    { t.self() } -> std::convertible_to<const T&>;
};

template<typename T>
concept matrix_expression = requires(const T& t) {
    typename T::value_type;
    { t.eval_scalar(size_t{}, size_t{}) } -> std::convertible_to<typename T::value_type>;
    { t.self() } -> std::convertible_to<const T&>;
};

template<typename T>
concept any_expression = vector_expression<T> || matrix_expression<T>;

// Matrix and vector type concepts
template<typename T>
concept matrix_type = is_matrix_v<T>;

template<typename T>
concept vector_type = is_vector_v<T>;

template<typename T>
concept square_matrix_type = is_square_matrix_v<T>;

// Dimensional compatibility concepts
template<typename T1, typename T2>
concept same_dimensions = have_same_dimensions_v<T1, T2>;

template<typename T1, typename T2>
concept multipliable = can_multiply_v<T1, T2>;

// Complex number concept
template<typename T>
concept complex_number = requires(const T& t) {
    typename T::value_type;
    { t.real() } -> std::convertible_to<typename T::value_type>;
    { t.imag() } -> std::convertible_to<typename T::value_type>;
};

// Quaternion concept
template<typename T>
concept quaternion_type = requires(const T& t) {
    typename T::value_type;
    { t.w() } -> std::convertible_to<typename T::value_type>;
    { t.x() } -> std::convertible_to<typename T::value_type>;
    { t.y() } -> std::convertible_to<typename T::value_type>;
    { t.z() } -> std::convertible_to<typename T::value_type>;
};

// Angle concept
template<typename T>
concept angle_type = requires(const T& t) {
    typename T::value_type;
    { t.value() } -> std::convertible_to<typename T::value_type>;
    { t.as_radians() } -> std::convertible_to<typename T::value_type>;
};

// Coordinate point concepts
template<typename T>
concept point2_type = requires(const T& t) {
    typename T::value_type;
    { t.x } -> std::convertible_to<typename T::value_type>;
    { t.y } -> std::convertible_to<typename T::value_type>;
};

template<typename T>
concept point3_type = requires(const T& t) {
    typename T::value_type;
    { t.x } -> std::convertible_to<typename T::value_type>;
    { t.y } -> std::convertible_to<typename T::value_type>;
    { t.z } -> std::convertible_to<typename T::value_type>;
};

// Container concepts for points
template<typename Container>
concept point2_container = requires(const Container& c) {
    typename Container::value_type;
    requires point2_type<typename Container::value_type>;
    { c.begin() } -> std::input_iterator;
    { c.end() } -> std::input_iterator;
    { c.size() } -> std::convertible_to<std::size_t>;
};

template<typename Container>
concept point3_container = requires(const Container& c) {
    typename Container::value_type;
    requires point3_type<typename Container::value_type>;
    { c.begin() } -> std::input_iterator;
    { c.end() } -> std::input_iterator;
    { c.size() } -> std::convertible_to<std::size_t>;
};

// More flexible range concept that works with arrays, vectors, spans, etc.
template<typename Range, typename PointType>
concept point_range = requires(const Range& r) {
    { std::begin(r) } -> std::input_iterator;
    { std::end(r) } -> std::input_iterator;
    requires point2_type<std::remove_cvref_t<decltype(*std::begin(r))>> ||
             point3_type<std::remove_cvref_t<decltype(*std::begin(r))>>;
};

// Concept for contiguous point storage (for SIMD optimization)
template<typename Container>
concept contiguous_point2_container = point2_container<Container> &&
    requires(const Container& c) {
        { c.data() } -> std::convertible_to<const typename Container::value_type*>;
    };

// Concept for sized point ranges
template<typename Range>
concept sized_point_range = point_range<Range, void> &&
    requires(const Range& r) {
        { std::size(r) } -> std::convertible_to<std::size_t>;
    };

// Iterator concept for DDA iterators
template<typename T>
concept dda_iterator = requires(T& t) {
    typename T::value_type;
    { *t } -> std::convertible_to<typename T::value_type>;
    { ++t } -> std::same_as<T&>;
    { t++ } -> std::same_as<T>;
    { t != T::end() } -> std::convertible_to<bool>;
};

} // namespace euler