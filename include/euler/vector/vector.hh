/**
 * @file vector.hh
 * @brief Mathematical vector class template
 */
#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <utility>
#include <array>
#include <cmath>

namespace euler {

/**
 * @class vector
 * @brief Mathematical vector of fixed size
 * 
 * Vector is implemented as a specialization of matrix with either Nx1 or 1xN dimensions.
 * The default orientation depends on EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR:
 * - Column-major layout (default): vectors are Nx1 column vectors
 * - Row-major layout: vectors are 1xN row vectors
 * 
 * @tparam T The scalar type (e.g., float, double)
 * @tparam N The number of components
 */
template<typename T, size_t N>
class vector : public matrix<T, 
                            #ifndef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
                                N, 1  // Column vector (Nx1) for column-major
                            #else
                                1, N  // Row vector (1xN) for row-major
                            #endif
                            > {
public:
    using base_type = matrix<T, 
                            #ifndef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
                                N, 1  // Column vector
                            #else
                                1, N  // Row vector
                            #endif
                            >;
    using value_type = T;              ///< Type of vector components
    using size_type = size_t;          ///< Type for sizes and indices
    using reference = T&;              ///< Reference to component
    using const_reference = const T&;  ///< Const reference to component
    
    static constexpr size_t size = N;  ///< Number of components
    
    // Inherit constructors
    using base_type::base_type;
    
    // Default constructor
    vector() = default;
    
    // Constructor from scalar (fills all components)
    explicit vector(T value) : base_type(value) {}
    
    // Variadic constructor for convenient initialization
    template<typename... Args, 
             typename = std::enable_if_t<sizeof...(Args) == N && 
                                         (std::is_convertible_v<Args, T> && ...)>>
    vector(Args... args) {
        static_assert(sizeof...(Args) == N, "Number of arguments must match vector dimension");
        std::array<T, N> values = {static_cast<T>(args)...};
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = values[i];
        }
    }
    
    // Constructor from base matrix
    explicit vector(const base_type& m) : base_type(m) {}
    
    // Implicit conversion from any matrix with matching element count
    // This allows seamless conversion between row/column vectors
    template<size_t R, size_t C, bool Layout,
             typename = std::enable_if_t<R * C == N>>
    vector(const matrix<T, R, C, Layout>& m) {
        // Just copy the elements linearly - vectors are always stored contiguously
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = m[i];
        }
    }
    
    // Element access using indices instead of (i, 0)
    reference operator[](size_t idx) {
        return base_type::operator[](idx);
    }
    
    const_reference operator[](size_t idx) const {
        return base_type::operator[](idx);
    }
    
    // Named component access for common sizes
    template<size_t M = N>
    std::enable_if_t<M >= 1, reference> x() { return (*this)[0]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 1, const_reference> x() const { return (*this)[0]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 2, reference> y() { return (*this)[1]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 2, const_reference> y() const { return (*this)[1]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 3, reference> z() { return (*this)[2]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 3, const_reference> z() const { return (*this)[2]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 4, reference> w() { return (*this)[3]; }
    
    template<size_t M = N>
    std::enable_if_t<M >= 4, const_reference> w() const { return (*this)[3]; }
    
    // Alternative names for components
    template<size_t M = N>
    std::enable_if_t<M >= 1, reference> r() { return x(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 1, const_reference> r() const { return x(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 2, reference> g() { return y(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 2, const_reference> g() const { return y(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 3, reference> b() { return z(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 3, const_reference> b() const { return z(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 4, reference> a() { return w(); }
    
    template<size_t M = N>
    std::enable_if_t<M >= 4, const_reference> a() const { return w(); }
    
    // Length/magnitude operations
    T length_squared() const {
        T sum = T(0);
        for (size_t i = 0; i < N; ++i) {
            sum += (*this)[i] * (*this)[i];
        }
        return sum;
    }
    
    T length() const {
        return std::sqrt(length_squared());
    }
    
    T magnitude() const { return length(); }
    T magnitude_squared() const { return length_squared(); }
    
    // Normalization
    vector normalized() const {
        T len = length();
        EULER_CHECK(len > T(0), error_code::invalid_argument,
                   "Cannot normalize zero-length vector");
        return *this / len;
    }
    
    void normalize() {
        *this = normalized();
    }
    
    // Factory methods
    static vector zero() {
        return vector(T(0));
    }
    
    static vector ones() {
        return vector(T(1));
    }
    
    // Unit vectors for common dimensions
    template<size_t M = N>
    static std::enable_if_t<M >= 1, vector> unit_x() {
        vector v = zero();
        v.x() = T(1);
        return v;
    }
    
    template<size_t M = N>
    static std::enable_if_t<M >= 2, vector> unit_y() {
        vector v = zero();
        v.y() = T(1);
        return v;
    }
    
    template<size_t M = N>
    static std::enable_if_t<M >= 3, vector> unit_z() {
        vector v = zero();
        v.z() = T(1);
        return v;
    }
    
    template<size_t M = N>
    static std::enable_if_t<M >= 4, vector> unit_w() {
        vector v = zero();
        v.w() = T(1);
        return v;
    }
};

// Column vector (Nx1) - explicit type
template<typename T, size_t N>
class column_vector : public matrix<T, N, 1> {
public:
    using base_type = matrix<T, N, 1>;
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    
    static constexpr size_t size = N;
    
    // Inherit constructors
    using base_type::base_type;
    
    // Default constructor
    column_vector() = default;
    
    // Conversion from row vector
    explicit column_vector(const matrix<T, 1, N>& row_vec) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = row_vec[i];
        }
    }
    
    // Conversion from generic vector
    explicit column_vector(const vector<T, N>& v) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = v[i];
        }
    }
    
    // Convenience element access
    reference operator[](size_t idx) {
        return base_type::operator[](idx);
    }
    
    const_reference operator[](size_t idx) const {
        return base_type::operator[](idx);
    }
};

// Row vector (1xN) - explicit type  
template<typename T, size_t N>
class row_vector : public matrix<T, 1, N> {
public:
    using base_type = matrix<T, 1, N>;
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    
    static constexpr size_t size = N;
    
    // Inherit constructors
    using base_type::base_type;
    
    // Default constructor
    row_vector() = default;
    
    // Variadic constructor
    template<typename... Args, 
             std::enable_if_t<sizeof...(Args) == N && 
                              std::conjunction_v<std::is_convertible<Args, T>...>, int> = 0>
    constexpr row_vector(Args... args) : base_type{} {
        T values[] = {static_cast<T>(args)...};
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = values[i];
        }
    }
    
    // Conversion from column vector
    explicit row_vector(const matrix<T, N, 1>& col_vec) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = col_vec[i];
        }
    }
    
    // Conversion from generic vector
    explicit row_vector(const vector<T, N>& v) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = v[i];
        }
    }
    
    // Convenience element access
    reference operator[](size_t idx) {
        return base_type::operator[](idx);
    }
    
    const_reference operator[](size_t idx) const {
        return base_type::operator[](idx);
    }
};

// Common vector type aliases
template<typename T> using vec2 = vector<T, 2>;
template<typename T> using vec3 = vector<T, 3>;
template<typename T> using vec4 = vector<T, 4>;

// Explicit column vectors
template<typename T> using cvec2 = column_vector<T, 2>;
template<typename T> using cvec3 = column_vector<T, 3>;
template<typename T> using cvec4 = column_vector<T, 4>;

// Explicit row vectors
template<typename T> using rvec2 = row_vector<T, 2>;
template<typename T> using rvec3 = row_vector<T, 3>;
template<typename T> using rvec4 = row_vector<T, 4>;

// Default precision vectors
using vector2 = vec2<scalar>;
using vector3 = vec3<scalar>;
using vector4 = vec4<scalar>;

// Explicit precision vectors
using vec2f = vec2<float>;
using vec3f = vec3<float>;
using vec4f = vec4<float>;

using vec2d = vec2<double>;
using vec3d = vec3<double>;
using vec4d = vec4<double>;

using vec2i = vec2<int32_t>;
using vec3i = vec3<int32_t>;
using vec4i = vec4<int32_t>;

using vec2u = vec2<uint32_t>;
using vec3u = vec3<uint32_t>;
using vec4u = vec4<uint32_t>;

// Extend existing is_vector trait for our vector types
template<typename T, size_t N>
struct is_vector<vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_vector<column_vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct is_vector<row_vector<T, N>> : std::true_type {};

// Vector dimension
template<typename T>
struct vector_dimension {
    static constexpr size_t value = 0;
};

template<typename T, size_t N>
struct vector_dimension<vector<T, N>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_dimension<column_vector<T, N>> {
    static constexpr size_t value = N;
};

template<typename T, size_t N>
struct vector_dimension<row_vector<T, N>> {
    static constexpr size_t value = N;
};

template<typename T>
constexpr size_t vector_dimension_v = vector_dimension<T>::value;

} // namespace euler