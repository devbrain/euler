#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/core/expression.hh>
#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <euler/core/simd.hh>
#include <algorithm>
#include <cmath>

namespace euler {

// Forward declarations
template<typename T> class const_matrix_view;

// Matrix view provides a zero-copy view into a matrix or sub-matrix
template<typename T>
class matrix_view : public expression<matrix_view<T>, T> {
public:
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Constructor from raw data with dimensions and strides
    matrix_view(T* data, size_t rows, size_t cols, 
                size_t row_stride = 1, size_t col_stride = 0,
                [[maybe_unused]] bool expect_aligned = true) noexcept
        : data_(data), rows_(rows), cols_(cols),
          row_stride_(row_stride),
          col_stride_(col_stride == 0 ? rows : col_stride) {
        EULER_CHECK_NOT_NULL(data);
        EULER_CHECK_POSITIVE(rows, "rows");
        EULER_CHECK_POSITIVE(cols, "cols");
        
        // Warn if data is not aligned for SIMD (in debug mode)
        // Only check if we expect alignment (base views should be aligned, subviews may not be)
        // Also skip check for small matrices where SIMD isn't beneficial
        EULER_DEBUG_CHECK(!expect_aligned || is_aligned(data) || !simd_traits<T>::has_simd || (rows * cols < 4),
                         error_code::performance_warning,
                         "Matrix view data is not aligned for SIMD operations");
    }
    
    // Constructor from matrix
    template<size_t Rows, size_t Cols, bool ColumnMajor>
    explicit matrix_view(matrix<T, Rows, Cols, ColumnMajor>& mat) noexcept
        : data_(mat.data()), rows_(Rows), cols_(Cols),
          row_stride_(ColumnMajor ? 1 : Cols), 
          col_stride_(ColumnMajor ? Rows : 1) {}
    
    // Constructor from const matrix (creates const_matrix_view)
    template<size_t Rows, size_t Cols, bool ColumnMajor>
    explicit matrix_view(const matrix<T, Rows, Cols, ColumnMajor>& mat) = delete;
    
    // Sub-view creation
    matrix_view submatrix(size_t start_row, size_t start_col,
                         size_t num_rows, size_t num_cols) {
        EULER_CHECK(start_row + num_rows <= rows_, error_code::index_out_of_bounds,
                   "Submatrix row range [", start_row, ", ", start_row + num_rows,
                   ") exceeds matrix rows ", rows_);
        EULER_CHECK(start_col + num_cols <= cols_, error_code::index_out_of_bounds,
                   "Submatrix column range [", start_col, ", ", start_col + num_cols,
                   ") exceeds matrix columns ", cols_);
        
        T* sub_data = &(*this)(start_row, start_col);
        // Subviews may not be aligned, so pass false for expect_aligned
        return matrix_view(sub_data, num_rows, num_cols, row_stride_, col_stride_, false);
    }
    
    const_matrix_view<T> submatrix(size_t start_row, size_t start_col,
                                   size_t num_rows, size_t num_cols) const {
        EULER_CHECK(start_row + num_rows <= rows_, error_code::index_out_of_bounds,
                   "Submatrix row range [", start_row, ", ", start_row + num_rows,
                   ") exceeds matrix rows ", rows_);
        EULER_CHECK(start_col + num_cols <= cols_, error_code::index_out_of_bounds,
                   "Submatrix column range [", start_col, ", ", start_col + num_cols,
                   ") exceeds matrix columns ", cols_);
        
        const T* sub_data = &(*this)(start_row, start_col);
        return const_matrix_view<T>(sub_data, num_rows, num_cols, row_stride_, col_stride_, false);
    }
    
    // Row view
    matrix_view row(size_t row_idx) {
        EULER_CHECK(row_idx < rows_, error_code::index_out_of_bounds,
                   "Row index ", row_idx, " out of range [0, ", rows_, ")");
        return matrix_view(&(*this)(row_idx, 0), 1, cols_, col_stride_, col_stride_, false);
    }
    
    const_matrix_view<T> row(size_t row_idx) const {
        EULER_CHECK(row_idx < rows_, error_code::index_out_of_bounds,
                   "Row index ", row_idx, " out of range [0, ", rows_, ")");
        return const_matrix_view<T>(&(*this)(row_idx, 0), 1, cols_, col_stride_, col_stride_, false);
    }
    
    // Column view
    matrix_view col(size_t col_idx) {
        EULER_CHECK(col_idx < cols_, error_code::index_out_of_bounds,
                   "Column index ", col_idx, " out of range [0, ", cols_, ")");
        return matrix_view(&(*this)(0, col_idx), rows_, 1, row_stride_, row_stride_, false);
    }
    
    const_matrix_view<T> col(size_t col_idx) const {
        EULER_CHECK(col_idx < cols_, error_code::index_out_of_bounds,
                   "Column index ", col_idx, " out of range [0, ", cols_, ")");
        return const_matrix_view<T>(&(*this)(0, col_idx), rows_, 1, row_stride_, row_stride_, false);
    }
    
    // Diagonal view
    matrix_view diagonal(int offset = 0) {
        const size_t row_start = offset < 0 ? static_cast<size_t>(-offset) : 0;
        const size_t col_start = offset > 0 ? static_cast<size_t>(offset) : 0;
        const size_t diag_len = std::min(
            rows_ - row_start,
            cols_ - col_start
        );
        
        EULER_CHECK(diag_len > 0, error_code::invalid_argument,
                   "Diagonal offset ", offset, " is out of range");
        
        T* diag_data = &(*this)(row_start, col_start);
        return matrix_view(diag_data, diag_len, 1, 
                          row_stride_ + col_stride_, row_stride_ + col_stride_, false);
    }
    
    const_matrix_view<T> diagonal(int offset = 0) const {
        const size_t row_start = offset < 0 ? static_cast<size_t>(-offset) : 0;
        const size_t col_start = offset > 0 ? static_cast<size_t>(offset) : 0;
        const size_t diag_len = std::min(
            rows_ - row_start,
            cols_ - col_start
        );
        
        EULER_CHECK(diag_len > 0, error_code::invalid_argument,
                   "Diagonal offset ", offset, " is out of range");
        
        const T* diag_data = &(*this)(row_start, col_start);
        return const_matrix_view<T>(diag_data, diag_len, 1, 
                                   row_stride_ + col_stride_, row_stride_ + col_stride_, false);
    }
    
    // Element access
    reference operator()(size_t row, size_t col) {
        EULER_DEBUG_CHECK(row < rows_, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows_, ")");
        EULER_DEBUG_CHECK(col < cols_, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols_, ")");
        return data_[row * row_stride_ + col * col_stride_];
    }
    
    const_reference operator()(size_t row, size_t col) const {
        EULER_DEBUG_CHECK(row < rows_, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows_, ")");
        EULER_DEBUG_CHECK(col < cols_, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols_, ")");
        return data_[row * row_stride_ + col * col_stride_];
    }
    
    // Linear access for vector views
    reference operator[](size_t idx) {
        if (is_vector()) {
            // Vector element access - works for any vector view
            EULER_DEBUG_CHECK(idx < vector_size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", vector_size(), ")");
            if (rows_ == 1) {
                // Row vector
                return (*this)(0, idx);
            } else {
                // Column vector
                return (*this)(idx, 0);
            }
        } else {
            // Matrix linear access requires contiguous storage
            EULER_CHECK(is_contiguous(), error_code::invalid_argument,
                       "Linear access requires contiguous storage for matrices");
            EULER_DEBUG_CHECK(idx < size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", size(), ")");
            return data_[idx];
        }
    }
    
    const_reference operator[](size_t idx) const {
        if (is_vector()) {
            // Vector element access - works for any vector view
            EULER_DEBUG_CHECK(idx < vector_size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", vector_size(), ")");
            if (rows_ == 1) {
                // Row vector
                return (*this)(0, idx);
            } else {
                // Column vector
                return (*this)(idx, 0);
            }
        } else {
            // Matrix linear access requires contiguous storage
            EULER_CHECK(is_contiguous(), error_code::invalid_argument,
                       "Linear access requires contiguous storage for matrices");
            EULER_DEBUG_CHECK(idx < size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", size(), ")");
            return data_[idx];
        }
    }
    
    // Expression template interface
    T eval_scalar(size_t idx) const {
        return (*this)[idx];
    }
    
    T eval_scalar(size_t row, size_t col) const {
        return (*this)(row, col);
    }
    
    // Properties
    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    
    size_t row_stride() const noexcept { return row_stride_; }
    size_t col_stride() const noexcept { return col_stride_; }
    
    bool is_contiguous() const noexcept {
        return (row_stride_ == 1 && col_stride_ == rows_) ||
               (col_stride_ == 1 && row_stride_ == cols_);
    }
    
    bool is_column_major() const noexcept {
        return row_stride_ == 1 && col_stride_ == rows_;
    }
    
    bool is_row_major() const noexcept {
        return col_stride_ == 1 && row_stride_ == cols_;
    }
    
    bool is_simd_aligned() const noexcept {
        return is_aligned(data_);
    }
    
    // Data access
    pointer data() noexcept { return data_; }
    const_pointer data() const noexcept { return data_; }
    
    // Assignment from expression
    template<typename Expr>
    matrix_view& operator=(const expression<Expr, T>& expr) {
        for (size_t j = 0; j < cols_; ++j) {
            for (size_t i = 0; i < rows_; ++i) {
                (*this)(i, j) = expr(i, j);
            }
        }
        return *this;
    }
    
    // Assignment from scalar
    matrix_view& operator=(T value) {
        for (size_t j = 0; j < cols_; ++j) {
            for (size_t i = 0; i < rows_; ++i) {
                (*this)(i, j) = value;
            }
        }
        return *this;
    }
    
    // Check if a subview would be SIMD aligned
    bool is_subview_aligned(size_t start_row, size_t start_col) const noexcept {
        const T* sub_data = &(*this)(start_row, start_col);
        return is_aligned(sub_data);
    }
    
    // Get the alignment offset for a given element
    size_t alignment_offset(size_t row, size_t col) const noexcept {
        const T* elem_ptr = &(*this)(row, col);
        return reinterpret_cast<uintptr_t>(elem_ptr) % simd_alignment<T>();
    }
    
    // Check if this view represents a vector (single row or column)
    bool is_vector() const noexcept {
        return rows_ == 1 || cols_ == 1;
    }
    
    // Get vector length (only valid if is_vector() is true)
    size_t vector_size() const noexcept {
        return rows_ == 1 ? cols_ : rows_;
    }
    
    // Length/magnitude operations for vector views
    T length_squared() const {
        EULER_CHECK(is_vector(), error_code::invalid_argument,
                   "length_squared() only valid for vector views");
        T sum = T(0);
        const size_t n = vector_size();
        if (is_contiguous()) {
            // Fast path for contiguous storage
            for (size_t i = 0; i < n; ++i) {
                sum += data_[i] * data_[i];
            }
        } else {
            // Strided access
            if (rows_ == 1) {
                // Row vector
                for (size_t i = 0; i < cols_; ++i) {
                    const T val = (*this)(0, i);
                    sum += val * val;
                }
            } else {
                // Column vector
                for (size_t i = 0; i < rows_; ++i) {
                    const T val = (*this)(i, 0);
                    sum += val * val;
                }
            }
        }
        return sum;
    }
    
    T length() const {
        return std::sqrt(length_squared());
    }

private:
    T* data_;
    size_t rows_;
    size_t cols_;
    size_t row_stride_;
    size_t col_stride_;
};

// Const matrix view
template<typename T>
class const_matrix_view : public expression<const_matrix_view<T>, T> {
public:
    using value_type = T;
    using size_type = size_t;
    using const_reference = const T&;
    using const_pointer = const T*;
    
    // Constructor from raw data
    const_matrix_view(const T* data, size_t rows, size_t cols,
                     size_t row_stride = 1, size_t col_stride = 0,
                     bool expect_aligned = true) noexcept
        : data_(data), rows_(rows), cols_(cols),
          row_stride_(row_stride),
          col_stride_(col_stride == 0 ? rows : col_stride) {
        EULER_CHECK_NOT_NULL(data);
        EULER_CHECK_POSITIVE(rows, "rows");
        EULER_CHECK_POSITIVE(cols, "cols");
        
        // Warn if data is not aligned for SIMD (in debug mode)
        // Only check if we expect alignment (base views should be aligned, subviews may not be)
        EULER_DEBUG_CHECK(!expect_aligned || is_aligned(data) || !simd_traits<T>::has_simd, 
                         error_code::performance_warning,
                         "Matrix view data is not aligned for SIMD operations");
    }
    
    // Constructor from matrix
    template<size_t Rows, size_t Cols, bool ColumnMajor>
    explicit const_matrix_view(const matrix<T, Rows, Cols, ColumnMajor>& mat) noexcept
        : data_(mat.data()), rows_(Rows), cols_(Cols),
          row_stride_(ColumnMajor ? 1 : Cols), 
          col_stride_(ColumnMajor ? Rows : 1) {}
    
    // Constructor from matrix_view
    explicit const_matrix_view(const matrix_view<T>& view) noexcept
        : data_(view.data()), rows_(view.rows()), cols_(view.cols()),
          row_stride_(view.row_stride()), col_stride_(view.col_stride()) {}
    
    // Element access
    const_reference operator()(size_t row, size_t col) const {
        EULER_DEBUG_CHECK(row < rows_, error_code::index_out_of_bounds,
                         "Row index ", row, " out of range [0, ", rows_, ")");
        EULER_DEBUG_CHECK(col < cols_, error_code::index_out_of_bounds,
                         "Column index ", col, " out of range [0, ", cols_, ")");
        return data_[row * row_stride_ + col * col_stride_];
    }
    
    // Expression template interface
    T eval_scalar(size_t row, size_t col) const {
        return (*this)(row, col);
    }
    
    // Properties
    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    
    size_t row_stride() const noexcept { return row_stride_; }
    size_t col_stride() const noexcept { return col_stride_; }
    
    // Check if data is SIMD aligned
    bool is_simd_aligned() const noexcept {
        return is_aligned(data_);
    }
    
    // Sub-view creation methods (const versions)
    const_matrix_view submatrix(size_t start_row, size_t start_col,
                               size_t num_rows, size_t num_cols) const {
        EULER_CHECK(start_row + num_rows <= rows_, error_code::index_out_of_bounds,
                   "Submatrix row range [", start_row, ", ", start_row + num_rows,
                   ") exceeds matrix rows ", rows_);
        EULER_CHECK(start_col + num_cols <= cols_, error_code::index_out_of_bounds,
                   "Submatrix column range [", start_col, ", ", start_col + num_cols,
                   ") exceeds matrix columns ", cols_);
        
        const T* sub_data = &(*this)(start_row, start_col);
        return const_matrix_view(sub_data, num_rows, num_cols, row_stride_, col_stride_, false);
    }
    
    const_matrix_view row(size_t row_idx) const {
        EULER_CHECK(row_idx < rows_, error_code::index_out_of_bounds,
                   "Row index ", row_idx, " out of range [0, ", rows_, ")");
        return const_matrix_view(&(*this)(row_idx, 0), 1, cols_, col_stride_, col_stride_, false);
    }
    
    const_matrix_view col(size_t col_idx) const {
        EULER_CHECK(col_idx < cols_, error_code::index_out_of_bounds,
                   "Column index ", col_idx, " out of range [0, ", cols_, ")");
        return const_matrix_view(&(*this)(0, col_idx), rows_, 1, row_stride_, row_stride_, false);
    }
    
    const_matrix_view diagonal(int offset = 0) const {
        const size_t row_start = offset < 0 ? static_cast<size_t>(-offset) : 0;
        const size_t col_start = offset > 0 ? static_cast<size_t>(offset) : 0;
        const size_t diag_len = std::min(
            rows_ - row_start,
            cols_ - col_start
        );
        
        EULER_CHECK(diag_len > 0, error_code::invalid_argument,
                   "Diagonal offset ", offset, " is out of range");
        
        const T* diag_data = &(*this)(row_start, col_start);
        return const_matrix_view(diag_data, diag_len, 1, 
                                row_stride_ + col_stride_, row_stride_ + col_stride_, false);
    }
    
    // Check if this view represents a vector (single row or column)
    bool is_vector() const noexcept {
        return rows_ == 1 || cols_ == 1;
    }
    
    // Get vector length (only valid if is_vector() is true)
    size_t vector_size() const noexcept {
        return rows_ == 1 ? cols_ : rows_;
    }
    
    // Linear access for vector views
    const_reference operator[](size_t idx) const {
        if (is_vector()) {
            // Vector element access - works for any vector view
            EULER_DEBUG_CHECK(idx < vector_size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", vector_size(), ")");
            if (rows_ == 1) {
                // Row vector
                return (*this)(0, idx);
            } else {
                // Column vector
                return (*this)(idx, 0);
            }
        } else {
            // Matrix linear access requires contiguous storage
            EULER_CHECK(is_contiguous(), error_code::invalid_argument,
                       "Linear access requires contiguous storage for matrices");
            EULER_DEBUG_CHECK(idx < size(), error_code::index_out_of_bounds,
                             "Index ", idx, " out of range [0, ", size(), ")");
            return data_[idx];
        }
    }
    
    // Check if contiguous
    bool is_contiguous() const noexcept {
        return (row_stride_ == 1 && col_stride_ == rows_) ||
               (col_stride_ == 1 && row_stride_ == cols_);
    }
    
    // Length/magnitude operations for vector views
    T length_squared() const {
        EULER_CHECK(is_vector(), error_code::invalid_argument,
                   "length_squared() only valid for vector views");
        T sum = T(0);
        const size_t n = vector_size();
        if (is_contiguous()) {
            // Fast path for contiguous storage
            for (size_t i = 0; i < n; ++i) {
                sum += data_[i] * data_[i];
            }
        } else {
            // Strided access
            if (rows_ == 1) {
                // Row vector
                for (size_t i = 0; i < cols_; ++i) {
                    const T val = (*this)(0, i);
                    sum += val * val;
                }
            } else {
                // Column vector
                for (size_t i = 0; i < rows_; ++i) {
                    const T val = (*this)(i, 0);
                    sum += val * val;
                }
            }
        }
        return sum;
    }
    
    T length() const {
        return std::sqrt(length_squared());
    }
    
    // Check if a subview would be SIMD aligned
    bool is_subview_aligned(size_t start_row, size_t start_col) const noexcept {
        const T* sub_data = &(*this)(start_row, start_col);
        return is_aligned(sub_data);
    }
    
    // Get the alignment offset for a given element
    size_t alignment_offset(size_t row, size_t col) const noexcept {
        const T* elem_ptr = &(*this)(row, col);
        return reinterpret_cast<uintptr_t>(elem_ptr) % simd_alignment<T>();
    }

private:
    const T* data_;
    size_t rows_;
    size_t cols_;
    size_t row_stride_;
    size_t col_stride_;
};

// Helper functions to create views
template<typename T, size_t R, size_t C, bool ColumnMajor>
matrix_view<T> make_view(matrix<T, R, C, ColumnMajor>& mat) {
    return matrix_view<T>(mat);
}

template<typename T, size_t R, size_t C, bool ColumnMajor>
const_matrix_view<T> make_view(const matrix<T, R, C, ColumnMajor>& mat) {
    return const_matrix_view<T>(mat);
}

// Type aliases
template<typename T> using mat_view = matrix_view<T>;
template<typename T> using const_mat_view = const_matrix_view<T>;

} // namespace euler