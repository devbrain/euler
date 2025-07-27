/**
 * @file io.hh
 * @brief Stream input/output operators for Euler types with pretty printing
 * @ingroup CoreModule
 */
#pragma once

#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
#include <euler/quaternion/quaternion.hh>
#include <euler/angles/angle.hh>
#include <euler/core/traits.hh>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>

namespace euler {

namespace detail {

// Helper to get current stream width, with default
inline int get_stream_width(std::ostream& os, int default_width = 8) {
    auto width = os.width();
    return width > 0 ? static_cast<int>(width) : default_width;
}

// Helper to format a single value using current stream state
template<typename T>
std::string format_value(std::ostream& os, const T& value) {
    std::ostringstream oss;
    // Copy formatting state from original stream
    oss.copyfmt(os);
    oss << value;
    return oss.str();
}

// Helper to calculate column widths for matrix output
template<typename MatrixLike>
std::vector<size_t> calculate_column_widths(std::ostream& os, const MatrixLike& m, 
                                           size_t rows, size_t cols) {
    std::vector<size_t> widths(cols, 0);
    
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            std::string formatted = format_value(os, m(i, j));
            widths[j] = std::max(widths[j], formatted.length());
        }
    }
    
    // Ensure minimum width from stream settings
    int min_width = get_stream_width(os, 0);
    if (min_width > 0) {
        for (auto& w : widths) {
            w = std::max(w, static_cast<size_t>(min_width));
        }
    }
    
    return widths;
}

// Pretty print a matrix-like object
template<typename MatrixLike>
void pretty_print_matrix(std::ostream& os, const MatrixLike& m, 
                        size_t rows, size_t cols,
                        const std::string& indent = "") {
    // Calculate column widths
    auto widths = calculate_column_widths(os, m, rows, cols);
    
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    
    // Opening bracket
    os << "[";
    
    for (size_t i = 0; i < rows; ++i) {
        if (i > 0) {
            os << ",\n" << indent << " ";
        }
        
        os << "[";
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) os << ", ";
            
            // Format the value
            std::string formatted = format_value(os, m(i, j));
            
            // Right-align numbers by default
            os << std::setw(static_cast<int>(widths[j])) << formatted;
        }
        os << "]";
    }
    
    os << "]";
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
}

// Pretty print a vector-like object (normal version)
template<typename VectorLike>
void pretty_print_vector(std::ostream& os, const VectorLike& v, size_t size) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = get_stream_width(os, 0);
    
    os << "(";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) os << ", ";
        
        if (width > 0) {
            // Format with specified width
            std::string formatted = format_value(os, v[i]);
            os << std::setw(width) << formatted;
        } else {
            os << v[i];
        }
    }
    os << ")";
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
}

// Pretty print a vector-like object (callable version for views)
template<typename Func>
void pretty_print_vector_func(std::ostream& os, const Func& get_elem, size_t size) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = get_stream_width(os, 0);
    
    os << "(";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) os << ", ";
        
        if (width > 0) {
            // Format with specified width
            std::string formatted = format_value(os, get_elem(i));
            os << std::setw(width) << formatted;
        } else {
            os << get_elem(i);
        }
    }
    os << ")";
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
}

} // namespace detail

// Vector output operator with pretty printing
template<typename T, size_t N>
inline std::ostream& operator<<(std::ostream& os, const vector<T, N>& v) {
    detail::pretty_print_vector(os, v, N);
    return os;
}

// Generic expression output - evaluates and prints based on expression traits
template<typename Expr, typename T>
inline auto operator<<(std::ostream& os, const expression<Expr, T>& expr) 
    -> std::enable_if_t<std::is_base_of_v<expression<Expr, T>, Expr>, std::ostream&>
{
    const auto& derived = static_cast<const Expr&>(expr);
    
    // Use expression_traits to get dimensions safely
    using traits = expression_traits<Expr>;
    
    if constexpr (traits::rows == 0 || traits::cols == 0) {
        // Dynamic size or scalar expression - cannot print safely
        os << "<expression>";
    } else if constexpr (traits::cols == 1 && traits::rows > 1) {
        // Column vector - use vector printing
        detail::pretty_print_vector_func(os, [&](size_t i) { return derived(i, 0); }, traits::rows);
    } else if constexpr (traits::rows == 1 && traits::cols > 1) {
        // Row vector - use vector printing
        detail::pretty_print_vector_func(os, [&](size_t j) { return derived(0, j); }, traits::cols);
    } else if constexpr (traits::rows == 1 && traits::cols == 1) {
        // Scalar-like (1x1 matrix)
        os << derived(0, 0);
    } else {
        // General matrix
        detail::pretty_print_matrix(os, derived, traits::rows, traits::cols);
    }
    return os;
}

// Matrix output operator with pretty printing
template<typename T, size_t R, size_t C, bool ColumnMajor>
inline std::ostream& operator<<(std::ostream& os, const matrix<T, R, C, ColumnMajor>& m) {
    detail::pretty_print_matrix(os, m, R, C);
    return os;
}

// Matrix view output operator with pretty printing
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const matrix_view<T>& mv) {
    const size_t rows = mv.rows();
    const size_t cols = mv.cols();
    
    if (cols == 1 && rows > 1) {
        // Column vector view
        detail::pretty_print_vector_func(os, [&](size_t i) { return mv(i, 0); }, rows);
    } else if (rows == 1 && cols > 1) {
        // Row vector view
        detail::pretty_print_vector_func(os, [&](size_t j) { return mv(0, j); }, cols);
    } else if (rows == 1 && cols == 1) {
        // Single element
        os << mv(0, 0);
    } else {
        // General matrix view
        detail::pretty_print_matrix(os, mv, rows, cols);
    }
    return os;
}

// Const matrix view output operator with pretty printing
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const const_matrix_view<T>& mv) {
    const size_t rows = mv.rows();
    const size_t cols = mv.cols();
    
    if (cols == 1 && rows > 1) {
        // Column vector view
        detail::pretty_print_vector_func(os, [&](size_t i) { return mv(i, 0); }, rows);
    } else if (rows == 1 && cols > 1) {
        // Row vector view
        detail::pretty_print_vector_func(os, [&](size_t j) { return mv(0, j); }, cols);
    } else if (rows == 1 && cols == 1) {
        // Single element
        os << mv(0, 0);
    } else {
        // General matrix view
        detail::pretty_print_matrix(os, mv, rows, cols);
    }
    return os;
}

// Quaternion output operator with pretty printing
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const quaternion<T>& q) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    os << "(";
    
    auto output_component = [&](const T& val, const std::string& suffix = "") {
        if (width > 0) {
            std::string formatted = detail::format_value(os, val) + suffix;
            os << std::setw(width + static_cast<int>(suffix.length())) << formatted;
        } else {
            os << val << suffix;
        }
    };
    
    output_component(q.w());
    os << ", ";
    output_component(q.x(), "i");
    os << ", ";
    output_component(q.y(), "j");
    os << ", ";
    output_component(q.z(), "k");
    
    os << ")";
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

// Angle output operators with pretty printing
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const radian<T>& r) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    if (width > 0) {
        std::string formatted = detail::format_value(os, r.value()) + " rad";
        os << std::setw(width + 4) << formatted;
    } else {
        os << r.value() << " rad";
    }
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const degree<T>& d) {
    // Save stream state
    std::ios_base::fmtflags flags(os.flags());
    std::streamsize precision = os.precision();
    int width = detail::get_stream_width(os, 0);
    
    if (width > 0) {
        std::string formatted = detail::format_value(os, d.value()) + "°";
        os << std::setw(width + 1) << formatted;
    } else {
        os << d.value() << "°";
    }
    
    // Restore stream state
    os.flags(flags);
    os.precision(precision);
    
    return os;
}

} // namespace euler