#pragma once

#include <euler/core/traits.hh>
#include <euler/core/types.hh>
#include <type_traits>
#include <cmath>

namespace euler {

// Forward declarations
template<typename T, size_t N> class vector;
template<typename T, size_t Rows, size_t Cols, bool RowMajor> class matrix;
template<typename T, typename Unit> class angle;
template<typename T> class complex;

// Helper to check if a matrix expression is vector-like (1xN or Nx1)
template<typename T>
constexpr bool is_vector_like_matrix() {
    if constexpr (is_matrix_expression_v<T>) {
        return expression_traits<T>::rows == 1 || expression_traits<T>::cols == 1;
    }
    return false;
}

// Combined check for vector or vector-like matrix expression
template<typename T>
constexpr bool is_vector_or_vector_like_v = is_vector_v<T> || is_vector_like_matrix<T>();

// Remove conflicting are_dimensions_compatible template
template<typename T1, typename T2>
struct are_dimensions_compatible;


// Trait to check if type is an angle
template<typename T>
struct is_angle_type : std::false_type {};

template<typename T, typename Unit>
struct is_angle_type<angle<T, Unit>> : std::true_type {};

template<typename T>
constexpr bool is_angle_type_v = is_angle_type<std::decay_t<T>>::value;

// Trait to check if type is complex
template<typename T>
struct is_complex_type : std::false_type {};

template<typename T>
struct is_complex_type<complex<T>> : std::true_type {};

template<typename T>
constexpr bool is_complex_type_v = is_complex_type<std::decay_t<T>>::value;

// Helper to check scalar types (including angles and complex)
template<typename T>
constexpr bool is_scalar_v = std::is_arithmetic_v<std::decay_t<T>> || is_angle_type_v<T> || is_complex_type_v<T>;

// Helper to get the effective size of a vector or vector-like expression
template<typename T>
constexpr size_t get_vector_size() {
    if constexpr (is_vector_v<T>) {
        return vector_size_v<T>;
    }
    else if constexpr (is_matrix_expression_v<T>) {
        if constexpr (expression_traits<T>::rows == 1) {
            return expression_traits<T>::cols;
        }
        else if constexpr (expression_traits<T>::cols == 1) {
            return expression_traits<T>::rows;
        }
    }
    return 0;
}

// Check if two types have compatible dimensions
template<typename T1, typename T2>
constexpr bool have_compatible_dimensions() {
    if constexpr (is_scalar_v<T1> && is_scalar_v<T2>) {
        return true;
    }
    else if constexpr (is_vector_or_vector_like_v<T1> && is_vector_or_vector_like_v<T2>) {
        return get_vector_size<T1>() == get_vector_size<T2>();
    }
    else if constexpr (is_matrix_expression_v<T1> && is_matrix_expression_v<T2> && 
                       !is_vector_or_vector_like_v<T1> && !is_vector_or_vector_like_v<T2>) {
        return expression_traits<T1>::rows == expression_traits<T2>::rows &&
               expression_traits<T1>::cols == expression_traits<T2>::cols;
    }
    else {
        return false;
    }
}

// Generic approx_equal function
// Helper to get value type
template<typename T, typename = void>
struct get_value_type {
    using type = T;  // For scalars
};

// For matrix expressions
template<typename T>
struct get_value_type<T, std::enable_if_t<is_matrix_expression_v<T>>> {
    using type = typename expression_traits<T>::value_type;
};

// For angle types
template<typename T>
struct get_value_type<T, std::enable_if_t<is_angle_type_v<T>>> {
    using type = typename std::decay_t<T>::value_type;
};

// For complex types
template<typename T>
struct get_value_type<T, std::enable_if_t<is_complex_type_v<T>>> {
    using type = typename std::decay_t<T>::value_type;
};


template<typename T>
using get_value_type_t = typename get_value_type<T>::type;

// Approximate equality comparison for arrays
template<typename T>
bool approx_equal_array(const T* a, const T* b, size_t size, T tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Generic approximate equality comparison
// This function evaluates expressions and compares element by element.
// Since this is typically the last point where expressions are used, evaluation is performed immediately
template<typename T1, typename T2,
         typename = std::enable_if_t<have_compatible_dimensions<T1, T2>()>>
bool approx_equal(const T1& a, const T2& b,
                  std::common_type_t<get_value_type_t<T1>, get_value_type_t<T2>> tolerance =
                      constants<std::common_type_t<get_value_type_t<T1>, get_value_type_t<T2>>>::epsilon) {
    
    using value_type = std::common_type_t<get_value_type_t<T1>, get_value_type_t<T2>>;
    
    // Static asserts for better error messages
    static_assert(have_compatible_dimensions<T1, T2>(), 
                  "approx_equal: Incompatible dimensions for comparison");
    
    if constexpr (is_vector_or_vector_like_v<T1> && is_vector_or_vector_like_v<T2>) {
        static_assert(get_vector_size<T1>() == get_vector_size<T2>(),
                      "approx_equal: Vector dimensions must match");
    }
    else if constexpr (is_matrix_expression_v<T1> && is_matrix_expression_v<T2> && 
                       !is_vector_or_vector_like_v<T1> && !is_vector_or_vector_like_v<T2>) {
        static_assert(expression_traits<T1>::rows == expression_traits<T2>::rows,
                      "approx_equal: Matrix row dimensions must match");
        static_assert(expression_traits<T1>::cols == expression_traits<T2>::cols,
                      "approx_equal: Matrix column dimensions must match");
    }
    
    // Handle scalar comparison
    if constexpr (is_scalar_v<T1> && is_scalar_v<T2>) {
        // Handle complex comparison
        if constexpr (is_complex_type_v<T1> || is_complex_type_v<T2>) {
            // If both are complex
            if constexpr (is_complex_type_v<T1> && is_complex_type_v<T2>) {
                return std::abs(a.real() - b.real()) <= tolerance && 
                       std::abs(a.imag() - b.imag()) <= tolerance;
            }
            // If only one is complex, convert the other to complex
            else if constexpr (is_complex_type_v<T1>) {
                value_type b_val;
                if constexpr (is_angle_type_v<T2>) {
                    b_val = static_cast<value_type>(b.value());
                } else {
                    b_val = static_cast<value_type>(b);
                }
                return std::abs(a.real() - b_val) <= tolerance && 
                       std::abs(a.imag()) <= tolerance;
            }
            else { // T2 is complex
                value_type a_val;
                if constexpr (is_angle_type_v<T1>) {
                    a_val = static_cast<value_type>(a.value());
                } else {
                    a_val = static_cast<value_type>(a);
                }
                return std::abs(a_val - b.real()) <= tolerance && 
                       std::abs(b.imag()) <= tolerance;
            }
        }
        // Handle angle/scalar comparison
        else {
            // Extract values, handling both regular scalars and angles
            value_type val_a, val_b;
            if constexpr (is_angle_type_v<T1>) {
                val_a = static_cast<value_type>(a.value());
            } else {
                val_a = static_cast<value_type>(a);
            }
            
            if constexpr (is_angle_type_v<T2>) {
                val_b = static_cast<value_type>(b.value());
            } else {
                val_b = static_cast<value_type>(b);
            }
            
            return std::abs(val_a - val_b) <= tolerance;
        }
    }
    // Handle vector-vector comparison
    else if constexpr (is_vector_or_vector_like_v<T1> && is_vector_or_vector_like_v<T2>) {
        constexpr size_t size = get_vector_size<T1>();
        
        if constexpr (is_vector_v<T1> && is_vector_v<T2>) {
            // Direct vector comparison with SIMD optimization
            // Evaluate to ensure contiguous memory access
            vector<value_type, size> vec_a(a);
            vector<value_type, size> vec_b(b);
            
            return approx_equal_array(vec_a.data(), vec_b.data(), size, tolerance);
        }
        else if constexpr (is_vector_v<T1> && !is_vector_v<T2>) {
            // Vector vs matrix expression (1xN or Nx1)
            // Evaluate both to ensure contiguous memory
            vector<value_type, size> vec_a(a);
            
            if constexpr (expression_traits<T2>::rows == 1) {
                matrix<value_type, 1, size, true> mat_b(b);
                return approx_equal_array(vec_a.data(), mat_b.data(), size, tolerance);
            } else {
                matrix<value_type, size, 1, true> mat_b(b);
                return approx_equal_array(vec_a.data(), mat_b.data(), size, tolerance);
            }

        }
        else if constexpr (!is_vector_v<T1> && is_vector_v<T2>) {
            // Matrix expression vs vector
            return approx_equal(b, a, tolerance);
        }
        else {
            // Both are matrix expressions that are vector-like
            // Evaluate both to matrices and compare
            if constexpr (expression_traits<T1>::rows == 1 && expression_traits<T2>::rows == 1) {
                // Both row vectors
                matrix<value_type, 1, size, true> mat_a(a);
                matrix<value_type, 1, size, true> mat_b(b);
                return approx_equal_array(mat_a.data(), mat_b.data(), size, tolerance);
            }
            else if constexpr (expression_traits<T1>::cols == 1 && expression_traits<T2>::cols == 1) {
                // Both column vectors
                matrix<value_type, size, 1, true> mat_a(a);
                matrix<value_type, size, 1, true> mat_b(b);
                return approx_equal_array(mat_a.data(), mat_b.data(), size, tolerance);
            }
            else {
                // Mixed row/column - evaluate to same format
                matrix<value_type, size, 1, true> mat_a = expression_traits<T1>::rows == 1 ? 
                    matrix<value_type, size, 1, true>(transpose(a)) : matrix<value_type, size, 1, true>(a);
                matrix<value_type, size, 1, true> mat_b = expression_traits<T2>::rows == 1 ? 
                    matrix<value_type, size, 1, true>(transpose(b)) : matrix<value_type, size, 1, true>(b);
                return approx_equal_array(mat_a.data(), mat_b.data(), size, tolerance);
            }

        }
    }
    // Handle matrix-matrix comparison (non-vector-like)
    else if constexpr (is_matrix_expression_v<T1> && is_matrix_expression_v<T2> && 
                       !is_vector_or_vector_like_v<T1> && !is_vector_or_vector_like_v<T2>) {
        constexpr size_t rows = expression_traits<T1>::rows;
        constexpr size_t cols = expression_traits<T1>::cols;
        constexpr bool row_major = expression_traits<T1>::row_major;
        
        // Evaluate expressions to concrete matrices
        matrix<value_type, rows, cols, row_major> mat_a(a);
        matrix<value_type, rows, cols, row_major> mat_b(b);
        
        // Use SIMD optimized comparison for contiguous data
        return approx_equal_array(mat_a.data(), mat_b.data(), rows * cols, tolerance);
    }
    else {
        // This should never be reached due to SFINAE, but needed for completeness
        return false;
    }
}

} // namespace euler

// Include headers after namespace to avoid circular dependencies
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_expr.hh>
#include <euler/angles/angle.hh>
#include <euler/complex/complex.hh>