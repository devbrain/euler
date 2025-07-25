/**
 * @file matrix.hh
 * @brief Fixed-size matrix class with expression templates
 * @ingroup MatrixModule
 */
#pragma once

#include <euler/core/expression.hh>
#include <euler/core/types.hh>
#include <euler/core/traits.hh>
#include <euler/core/error.hh>
#include <euler/core/simd.hh>
#include <algorithm>
#include <array>
#include <initializer_list>
#include <utility>
#include <tuple>

namespace euler {

// Forward declarations
template<typename T> class matrix_view;
template<typename T> class const_matrix_view;

/**
 * @brief Default matrix storage layout configuration
 * 
 * Can be overridden by defining EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
 * before including any Euler headers.
 */
#ifndef EULER_DEFAULT_MATRIX_LAYOUT_ROW_MAJOR
    /// Default to column-major for OpenGL compatibility
    constexpr bool default_column_major = true;
#else
    /// Row-major layout (DirectX, traditional C arrays)
    constexpr bool default_column_major = false;
#endif

/**
 * @class matrix
 * @brief Fixed-size matrix with configurable storage layout
 * 
 * This class represents a dense matrix with compile-time known dimensions.
 * It supports both row-major and column-major storage layouts and integrates
 * with the expression template system for efficient compound operations.
 * 
 * @tparam T The scalar type (e.g., float, double)
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @tparam ColumnMajor If true, uses column-major storage; otherwise row-major
 * 
 * @ingroup MatrixModule
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor = default_column_major>
class matrix : public expression<matrix<T, Rows, Cols, ColumnMajor>, T> {
public:
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    static constexpr size_t size = Rows * Cols;
    
    // Storage order
    static constexpr bool column_major = ColumnMajor;
    
    // Constructors
    constexpr matrix() noexcept : data_{} {}
    
    // Fill constructor
    explicit constexpr matrix(T value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    // Nested initializer list constructor (row-major input)
    constexpr matrix(std::initializer_list<std::initializer_list<T>> init) {
        EULER_CHECK(init.size() == rows, error_code::dimension_mismatch,
                   "Number of rows (", init.size(), ") doesn't match matrix rows (", rows, ")");
        
        size_t i = 0;
        for (const auto& row : init) {
            EULER_CHECK(row.size() == cols, error_code::dimension_mismatch,
                       "Row ", i, " has ", row.size(), " elements, expected ", cols);
            size_t j = 0;
            for (const auto& val : row) {
                (*this)(i, j) = val;
                ++j;
            }
            ++i;
        }
    }
    
    // Expression template constructor
    template<typename Expr>
    matrix(const expression<Expr, T>& expr) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                (*this)(i, j) = expr(i, j);
            }
        }
    }
    
    // Constructor from pair of vectors (for 2x2 matrices from build_orthonormal_basis)
    template<typename Vec1, typename Vec2>
    matrix(const std::pair<Vec1, Vec2>& basis) {
        static_assert(Rows == 2 && Cols == 2, 
                      "Pair constructor only valid for 2x2 matrices");
        static_assert(Vec1::size == 2 && Vec2::size == 2,
                      "Pair must contain 2D vectors");
        
        // Store vectors as columns (standard basis matrix format)
        for (size_t i = 0; i < 2; ++i) {
            (*this)(i, 0) = basis.first[i];
            (*this)(i, 1) = basis.second[i];
        }
    }
    
    // Constructor from tuple of 3 vectors (for 3x3 matrices from build_orthonormal_basis)
    template<typename Vec1, typename Vec2, typename Vec3>
    matrix(const std::tuple<Vec1, Vec2, Vec3>& basis) {
        static_assert(Rows == 3 && Cols == 3, 
                      "3-tuple constructor only valid for 3x3 matrices");
        static_assert(Vec1::size == 3 && Vec2::size == 3 && Vec3::size == 3,
                      "Tuple must contain 3D vectors");
        
        // Store vectors as columns (standard basis matrix format)
        for (size_t i = 0; i < 3; ++i) {
            (*this)(i, 0) = std::get<0>(basis)[i];
            (*this)(i, 1) = std::get<1>(basis)[i];
            (*this)(i, 2) = std::get<2>(basis)[i];
        }
    }
    
    // Constructor from tuple of 4 vectors (for 4x4 matrices from build_orthonormal_basis)
    template<typename Vec1, typename Vec2, typename Vec3, typename Vec4>
    matrix(const std::tuple<Vec1, Vec2, Vec3, Vec4>& basis) {
        static_assert(Rows == 4 && Cols == 4, 
                      "4-tuple constructor only valid for 4x4 matrices");
        static_assert(Vec1::size == 4 && Vec2::size == 4 && 
                      Vec3::size == 4 && Vec4::size == 4,
                      "Tuple must contain 4D vectors");
        
        // Store vectors as columns (standard basis matrix format)
        for (size_t i = 0; i < 4; ++i) {
            (*this)(i, 0) = std::get<0>(basis)[i];
            (*this)(i, 1) = std::get<1>(basis)[i];
            (*this)(i, 2) = std::get<2>(basis)[i];
            (*this)(i, 3) = std::get<3>(basis)[i];
        }
    }
    
    // Copy constructor
    matrix(const matrix&) = default;
    
    // Move constructor
    matrix(matrix&&) noexcept = default;
    
    // Assignment operators
    matrix& operator=(const matrix&) = default;
    matrix& operator=(matrix&&) noexcept = default;
    
    // Expression template assignment
    template<typename Expr>
    matrix& operator=(const expression<Expr, T>& expr) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                (*this)(i, j) = expr(i, j);
            }
        }
        return *this;
    }
    
    // Element access (row, col)
    constexpr reference operator()(size_t row, size_t col) {
        EULER_DEBUG_CHECK(row < rows, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows, ")");
        EULER_DEBUG_CHECK(col < cols, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols, ")");
        if constexpr (column_major) {
            return data_[col * rows + row];  // Column-major indexing
        } else {
            return data_[row * cols + col];  // Row-major indexing
        }
    }
    
    constexpr const_reference operator()(size_t row, size_t col) const {
        EULER_DEBUG_CHECK(row < rows, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows, ")");
        EULER_DEBUG_CHECK(col < cols, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols, ")");
        if constexpr (column_major) {
            return data_[col * rows + row];  // Column-major indexing
        } else {
            return data_[row * cols + col];  // Row-major indexing
        }
    }
    
    // Linear access (for 1D operations)
    constexpr reference operator[](size_t idx) {
        EULER_DEBUG_CHECK(idx < size, error_code::index_out_of_bounds,
                         "Index ", idx, " out of range [0, ", size, ")");
        return data_[idx];
    }
    
    constexpr const_reference operator[](size_t idx) const {
        EULER_DEBUG_CHECK(idx < size, error_code::index_out_of_bounds,
                         "Index ", idx, " out of range [0, ", size, ")");
        return data_[idx];
    }
    
    // Expression template interface
    T eval_scalar(size_t idx) const {
        return data_[idx];
    }
    
    T eval_scalar(size_t row, size_t col) const {
        return (*this)(row, col);
    }
    
    // Data access
    pointer data() noexcept { return data_.data(); }
    const_pointer data() const noexcept { return data_.data(); }
    
    // View creation methods
    matrix_view<T> view() {
        return matrix_view<T>(*this);
    }
    
    const_matrix_view<T> view() const {
        return const_matrix_view<T>(*this);
    }
    
    matrix_view<T> submatrix(size_t start_row, size_t start_col,
                            size_t num_rows, size_t num_cols) {
        matrix_view<T> full_view(*this);
        return full_view.submatrix(start_row, start_col, num_rows, num_cols);
    }
    
    const_matrix_view<T> submatrix(size_t start_row, size_t start_col,
                                  size_t num_rows, size_t num_cols) const {
        const_matrix_view<T> full_view(*this);
        return full_view.submatrix(start_row, start_col, num_rows, num_cols);
    }
    
    matrix_view<T> row(size_t row_idx) {
        matrix_view<T> full_view(*this);
        return full_view.row(row_idx);
    }
    
    const_matrix_view<T> row(size_t row_idx) const {
        const_matrix_view<T> full_view(*this);
        return full_view.row(row_idx);
    }
    
    matrix_view<T> col(size_t col_idx) {
        matrix_view<T> full_view(*this);
        return full_view.col(col_idx);
    }
    
    const_matrix_view<T> col(size_t col_idx) const {
        const_matrix_view<T> full_view(*this);
        return full_view.col(col_idx);
    }
    
    matrix_view<T> diagonal(int offset = 0) {
        matrix_view<T> full_view(*this);
        return full_view.diagonal(offset);
    }
    
    const_matrix_view<T> diagonal(int offset = 0) const {
        const_matrix_view<T> full_view(*this);
        return full_view.diagonal(offset);
    }
    
    // Column access (only available for column-major matrices)
    pointer col_data(size_t col) {
        static_assert(column_major, "col_data() is only available for column-major matrices");
        EULER_DEBUG_CHECK(col < cols, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols, ")");
        return &data_[col * rows];
    }
    
    const_pointer col_data(size_t col) const {
        static_assert(column_major, "col_data() is only available for column-major matrices");
        EULER_DEBUG_CHECK(col < cols, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols, ")");
        return &data_[col * rows];
    }
    
    // Row access (only available for row-major matrices)
    pointer row_data(size_t row) {
        static_assert(!column_major, "row_data() is only available for row-major matrices");
        EULER_DEBUG_CHECK(row < rows, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows, ")");
        return &data_[row * cols];
    }
    
    const_pointer row_data(size_t row) const {
        static_assert(!column_major, "row_data() is only available for row-major matrices");
        EULER_DEBUG_CHECK(row < rows, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows, ")");
        return &data_[row * cols];
    }
    
    // Size queries
    static constexpr size_type row_count() noexcept { return rows; }
    static constexpr size_type col_count() noexcept { return cols; }
    static constexpr size_type element_count() noexcept { return size; }
    
    // Comparison operators
    bool operator==(const matrix& other) const {
        return std::equal(data_.begin(), data_.end(), other.data_.begin());
    }
    
    bool operator!=(const matrix& other) const {
        return !(*this == other);
    }
    
    // Convert to different storage layout
    template<bool OtherColumnMajor>
    constexpr matrix<T, Rows, Cols, OtherColumnMajor> to_layout() const {
        if constexpr (column_major == OtherColumnMajor) {
            // Same layout, just copy
            return matrix<T, Rows, Cols, OtherColumnMajor>(*this);
        } else {
            // Different layout, need to transpose storage
            matrix<T, Rows, Cols, OtherColumnMajor> result;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = (*this)(i, j);
                }
            }
            return result;
        }
    }
    
    // Convenience methods
    constexpr auto to_column_major() const {
        return to_layout<true>();
    }
    
    constexpr auto to_row_major() const {
        return to_layout<false>();
    }
    
    // Static factory methods
    static constexpr matrix zero() noexcept {
        return matrix(T(0));
    }
    
    static constexpr matrix identity() noexcept {
        static_assert(Rows == Cols, "Identity matrix requires square dimensions");
        matrix result(T(0));
        for (size_t i = 0; i < Rows; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }
    
    // Create matrix from rows
    static constexpr matrix from_rows(std::initializer_list<std::initializer_list<T>> rows_data) {
        return matrix(rows_data);
    }
    
    // Create matrix from columns
    static constexpr matrix from_cols(std::initializer_list<std::initializer_list<T>> cols_data) {
        EULER_CHECK(cols_data.size() == cols, error_code::dimension_mismatch,
                   "Number of columns (", cols_data.size(), ") doesn't match matrix columns (", cols, ")");
        
        matrix result;
        size_t j = 0;
        for (const auto& col : cols_data) {
            EULER_CHECK(col.size() == rows, error_code::dimension_mismatch,
                       "Column ", j, " has ", col.size(), " elements, expected ", rows);
            size_t i = 0;
            for (const auto& val : col) {
                result(i, j) = val;
                ++i;
            }
            ++j;
        }
        return result;
    }
    
    // Create matrix from flat array in row-major order
    static constexpr matrix from_row_major(std::initializer_list<T> data) {
        EULER_CHECK(data.size() == size, error_code::dimension_mismatch,
                   "Data size (", data.size(), ") doesn't match matrix size (", size, ")");
        
        matrix result;
        auto it = data.begin();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = *it++;
            }
        }
        return result;
    }
    
    // Create matrix from flat array in column-major order
    static constexpr matrix from_col_major(std::initializer_list<T> data) {
        EULER_CHECK(data.size() == size, error_code::dimension_mismatch,
                   "Data size (", data.size(), ") doesn't match matrix size (", size, ")");
        
        matrix result;
        if constexpr (column_major) {
            // Direct copy for column-major matrices
            std::copy(data.begin(), data.end(), result.data_.begin());
        } else {
            // Need to transpose for row-major matrices
            auto it = data.begin();
            for (size_t j = 0; j < cols; ++j) {
                for (size_t i = 0; i < rows; ++i) {
                    result(i, j) = *it++;
                }
            }
        }
        return result;
    }
    
private:
    // Aligned storage for SIMD operations
    alignas(simd_alignment_v<T>::value) std::array<T, size> data_;
};

// Type aliases for common matrix types (using default layout)
template<typename T, bool ColumnMajor = default_column_major> 
using matrix2 = matrix<T, 2, 2, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix3 = matrix<T, 3, 3, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix4 = matrix<T, 4, 4, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix2x3 = matrix<T, 2, 3, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix2x4 = matrix<T, 2, 4, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix3x2 = matrix<T, 3, 2, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix3x4 = matrix<T, 3, 4, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix4x2 = matrix<T, 4, 2, ColumnMajor>;
template<typename T, bool ColumnMajor = default_column_major> 
using matrix4x3 = matrix<T, 4, 3, ColumnMajor>;

// Default precision aliases (using default layout)
using mat2 = matrix2<scalar>;
using mat3 = matrix3<scalar>;
using mat4 = matrix4<scalar>;
using mat2x3 = matrix2x3<scalar>;
using mat2x4 = matrix2x4<scalar>;
using mat3x2 = matrix3x2<scalar>;
using mat3x4 = matrix3x4<scalar>;
using mat4x2 = matrix4x2<scalar>;
using mat4x3 = matrix4x3<scalar>;

// Explicit layout aliases for when you need specific storage
template<typename T> using matrix2_row = matrix<T, 2, 2, false>;
template<typename T> using matrix3_row = matrix<T, 3, 3, false>;
template<typename T> using matrix4_row = matrix<T, 4, 4, false>;
template<typename T> using matrix2_col = matrix<T, 2, 2, true>;
template<typename T> using matrix3_col = matrix<T, 3, 3, true>;
template<typename T> using matrix4_col = matrix<T, 4, 4, true>;

// Generic layout aliases
template<typename T, size_t R, size_t C> using matrix_row = matrix<T, R, C, false>;
template<typename T, size_t R, size_t C> using matrix_col = matrix<T, R, C, true>;


} // namespace euler