/**
 * @file matrix_ops.hh
 * @brief Direct SIMD operations for matrices
 * @ingroup DirectModule
 * 
 * This header provides high-performance direct operations on matrices
 * that bypass the expression template system for maximum performance.
 * 
 * @section matrix_ops_features Key Features
 * - SIMD-optimized implementations for element-wise operations
 * - Optimized matrix multiplication with cache-friendly algorithms
 * - Support for both row-major and column-major layouts
 * - In-place operation support (result can alias inputs)
 * - Specialized operations for square matrices
 * 
 * @section matrix_ops_usage Usage Example
 * @code{.cpp}
 * #include <euler/direct/matrix_ops.hh>
 * 
 * using namespace euler;
 * using namespace euler::direct;
 * 
 * mat3<float> A, B, C;
 * // ... initialize matrices ...
 * 
 * // Element-wise operations
 * add(A, B, C);           // C = A + B
 * mul(2.0f, A, C);        // C = 2.0 * A (scalar multiplication)
 * 
 * // Matrix multiplication
 * matmul(A, B, C);        // C = A * B (matrix product)
 * 
 * // In-place operations
 * add(A, B, A);           // A = A + B (aliasing safe)
 * @endcode
 * 
 * @section matrix_ops_performance Performance Notes
 * - Direct operations avoid temporary allocations
 * - Matrix multiplication uses blocking for cache efficiency
 * - Small matrices use unrolled loops for optimal performance
 */
#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
#include <euler/core/simd.hh>
#include <euler/core/compiler.hh>
#include <euler/core/types.hh>
#include <euler/core/error.hh>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <limits>

namespace euler::direct {

// =============================================================================
// Binary Operations
// =============================================================================

/**
 * @defgroup matrix_binary_ops Binary Matrix Operations
 * @ingroup DirectModule
 * @brief Element-wise operations between two matrices
 * @{
 */

/**
 * @brief Matrix addition: result = op1 + op2
 * 
 * Performs element-wise addition of two matrices with the same dimensions
 * and layout. Uses SIMD instructions when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @tparam ColumnMajor Storage layout (true for column-major, false for row-major)
 * @param op1 First operand matrix
 * @param op2 Second operand matrix
 * @param result Output matrix (can be same as op1 or op2)
 * 
 * @code
 * mat3<float> A = mat3<float>::identity();
 * mat3<float> B = mat3<float>::identity() * 2;
 * mat3<float> C;
 * add(A, B, C);  // C = A + B
 * @endcode
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void add(const matrix<T, Rows, Cols, ColumnMajor>& op1, 
                            const matrix<T, Rows, Cols, ColumnMajor>& op2, 
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT p1 = op1.data();
    const T* EULER_RESTRICT p2 = op2.data();
    T* EULER_RESTRICT pr = result.data();
    
    constexpr size_t size = Rows * Cols;
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        if constexpr (size >= simd_size) {
            constexpr size_t vec_size = size - (size % simd_size);
            
            // Main SIMD loop
            for (size_t i = 0; i < vec_size; i += simd_size) {
                batch v1 = batch::load_unaligned(&p1[i]);
                batch v2 = batch::load_unaligned(&p2[i]);
                batch vr = v1 + v2;
                vr.store_unaligned(&pr[i]);
            }
            
            // Handle remainder
            if constexpr (size % simd_size != 0) {
                for (size_t i = vec_size; i < size; ++i) {
                    pr[i] = p1[i] + p2[i];
                }
            }
        } else {
            // Matrix too small for SIMD
            for (size_t i = 0; i < size; ++i) {
                pr[i] = p1[i] + p2[i];
            }
        }
    } else
#endif
    {
        // Fallback with compiler optimization hints
        EULER_LOOP_VECTORIZE
        EULER_LOOP_UNROLL(8)
        for (size_t i = 0; i < size; ++i) {
            pr[i] = p1[i] + p2[i];
        }
    }
}

/**
 * @brief Matrix subtraction: result = op1 - op2
 * 
 * Performs element-wise subtraction of two matrices with the same dimensions
 * and layout. Uses SIMD instructions when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @tparam ColumnMajor Storage layout
 * @param op1 First operand matrix (minuend)
 * @param op2 Second operand matrix (subtrahend)
 * @param result Output matrix (can be same as op1 or op2)
 * 
 * @code
 * mat3<float> A, B, C;
 * sub(A, B, C);  // C = A - B
 * @endcode
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void sub(const matrix<T, Rows, Cols, ColumnMajor>& op1, 
                            const matrix<T, Rows, Cols, ColumnMajor>& op2, 
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT p1 = op1.data();
    const T* EULER_RESTRICT p2 = op2.data();
    T* EULER_RESTRICT pr = result.data();
    
    constexpr size_t size = Rows * Cols;
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        if constexpr (size >= simd_size) {
            constexpr size_t vec_size = size - (size % simd_size);
            
            for (size_t i = 0; i < vec_size; i += simd_size) {
                batch v1 = batch::load_unaligned(&p1[i]);
                batch v2 = batch::load_unaligned(&p2[i]);
                batch vr = v1 - v2;
                vr.store_unaligned(&pr[i]);
            }
            
            // Handle remainder
            if constexpr (size % simd_size != 0) {
                for (size_t i = vec_size; i < size; ++i) {
                    pr[i] = p1[i] - p2[i];
                }
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                pr[i] = p1[i] - p2[i];
            }
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < size; ++i) {
            pr[i] = p1[i] - p2[i];
        }
    }
}

/** @} */ // end of matrix_binary_ops

// =============================================================================
// Scalar Operations
// =============================================================================

/**
 * @defgroup matrix_scalar_ops Scalar-Matrix Operations
 * @ingroup DirectModule
 * @brief Operations between scalars and matrices
 * @{
 */

/**
 * @brief Scalar multiplication: result = scalar * matrix
 * 
 * Multiplies each element of a matrix by a scalar value.
 * Uses SIMD broadcast and multiplication when available.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @tparam ColumnMajor Storage layout
 * @param m Input matrix
 * @param scalar Scalar multiplier
 * @param result Output matrix (can be same as m)
 * 
 * @code
 * mat3<float> A = mat3<float>::identity();
 * mat3<float> B;
 * scale(A, 2.0f, B);  // B = 2.0 * A
 * @endcode
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_HOT void scale(const matrix<T, Rows, Cols, ColumnMajor>& m, T scalar, 
                    matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT pm = m.data();
    T* EULER_RESTRICT pr = result.data();
    
    constexpr size_t size = Rows * Cols;
    
#ifdef EULER_HAS_XSIMD
    if constexpr (simd_traits<T>::has_simd) {
        using batch = typename simd_traits<T>::batch_type;
        constexpr size_t simd_size = batch::size;
        
        if constexpr (size >= simd_size) {
            batch scalar_vec = batch::broadcast(scalar);
            constexpr size_t vec_size = size - (size % simd_size);
            
            for (size_t i = 0; i < vec_size; i += simd_size) {
                batch vm = batch::load_unaligned(&pm[i]);
                batch vr = scalar_vec * vm;
                vr.store_unaligned(&pr[i]);
            }
            
            // Handle remainder
            if constexpr (size % simd_size != 0) {
                for (size_t i = vec_size; i < size; ++i) {
                    pr[i] = scalar * pm[i];
                }
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                pr[i] = scalar * pm[i];
            }
        }
    } else
#endif
    {
        EULER_LOOP_VECTORIZE
        for (size_t i = 0; i < size; ++i) {
            pr[i] = scalar * pm[i];
        }
    }
}

/**
 * @brief Scalar multiplication (alias): result = scalar * matrix
 * 
 * Alias for scale() function for consistency with vector operations.
 * 
 * @see scale()
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void mul(T scalar, const matrix<T, Rows, Cols, ColumnMajor>& m, 
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    scale(m, scalar, result);
}

/**
 * @brief Scalar multiplication: result = matrix * scalar
 * 
 * Commutative version of scalar multiplication.
 * 
 * @see scale()
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void mul(const matrix<T, Rows, Cols, ColumnMajor>& m, T scalar, 
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    scale(m, scalar, result);
}

/** @} */ // end of matrix_scalar_ops

// =============================================================================
// Matrix Multiplication
// =============================================================================

/**
 * @defgroup matrix_multiplication Matrix Multiplication
 * @ingroup DirectModule
 * @brief Optimized matrix multiplication operations
 * @{
 */

/**
 * @brief General matrix multiplication: result = a * b
 * 
 * Performs matrix multiplication with automatic handling of aliasing.
 * Uses cache-friendly algorithms for larger matrices.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam M Number of rows in matrix a (and result)
 * @tparam N Number of columns in matrix b (and result)
 * @tparam K Number of columns in a / rows in b
 * @tparam ColumnMajor Storage layout
 * @param a Left matrix (M×K)
 * @param b Right matrix (K×N)
 * @param result Output matrix (M×N)
 * 
 * @note If result aliases with a or b, uses temporary storage
 * @note Specialized implementations exist for small matrices (2×2, 3×3, 4×4)
 * 
 * @code
 * mat3<float> A, B, C;
 * mul(A, B, C);      // C = A * B
 * mul(A, B, A);      // A = A * B (aliasing handled)
 * @endcode
 */
template<typename T, size_t M, size_t N, size_t K, bool ColumnMajor>
void mul(const matrix<T, M, K, ColumnMajor>& a, 
         const matrix<T, K, N, ColumnMajor>& b,
         matrix<T, M, N, ColumnMajor>& result) {
    // Check for aliasing - only possible if dimensions match
    bool aliased = false;
    if constexpr (N == K) {
        // result (M×N) and a (M×K) have same dimensions when N == K
        aliased = (static_cast<const void*>(&result) == static_cast<const void*>(&a));
    }
    if constexpr (M == K) {
        // result (M×N) and b (K×N) have same dimensions when M == K
        aliased = aliased || (static_cast<const void*>(&result) == static_cast<const void*>(&b));
    }
    
    if (aliased) {
        // Use temporary storage
        matrix<T, M, N, ColumnMajor> temp;
        
        // Perform multiplication into temp
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T(0);
                EULER_LOOP_VECTORIZE
                EULER_LOOP_UNROLL(4)
                for (size_t k = 0; k < K; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                temp(i, j) = sum;
            }
        }
        
        // Copy result
        result = std::move(temp);
    } else {
        // Direct computation
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = T(0);
                EULER_LOOP_VECTORIZE
                EULER_LOOP_UNROLL(4)
                for (size_t k = 0; k < K; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
    }
}

// =============================================================================
// Optimized Square Matrix Multiplication
// =============================================================================

/**
 * @brief Optimized 2x2 matrix multiplication
 */
template<typename T, bool ColumnMajor>
EULER_HOT void mul(const matrix<T, 2, 2, ColumnMajor>& a,
                   const matrix<T, 2, 2, ColumnMajor>& b,
                   matrix<T, 2, 2, ColumnMajor>& result) {
    // Extract elements
    const T a00 = a(0,0), a01 = a(0,1);
    const T a10 = a(1,0), a11 = a(1,1);
    const T b00 = b(0,0), b01 = b(0,1);
    const T b10 = b(1,0), b11 = b(1,1);
    
    // Compute result
    result(0,0) = a00 * b00 + a01 * b10;
    result(0,1) = a00 * b01 + a01 * b11;
    result(1,0) = a10 * b00 + a11 * b10;
    result(1,1) = a10 * b01 + a11 * b11;
}

/**
 * @brief Optimized 3x3 matrix multiplication
 */
template<typename T, bool ColumnMajor>
EULER_HOT void mul(const matrix<T, 3, 3, ColumnMajor>& a,
                   const matrix<T, 3, 3, ColumnMajor>& b,
                   matrix<T, 3, 3, ColumnMajor>& result) {
    // Use temporary storage for aliasing safety
    T r[9];
    
    // Row 0
    r[0] = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
    r[1] = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
    r[2] = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);
    
    // Row 1
    r[3] = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
    r[4] = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
    r[5] = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);
    
    // Row 2
    r[6] = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
    r[7] = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
    r[8] = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);
    
    // Copy to result
    result(0,0) = r[0]; result(0,1) = r[1]; result(0,2) = r[2];
    result(1,0) = r[3]; result(1,1) = r[4]; result(1,2) = r[5];
    result(2,0) = r[6]; result(2,1) = r[7]; result(2,2) = r[8];
}

/**
 * @brief Optimized 4x4 matrix multiplication with SIMD
 */
template<typename T, bool ColumnMajor>
EULER_HOT void mul(const matrix<T, 4, 4, ColumnMajor>& a,
                   const matrix<T, 4, 4, ColumnMajor>& b,
                   matrix<T, 4, 4, ColumnMajor>& result) {
#ifdef EULER_HAS_SIMD
    if constexpr (std::is_same_v<T, float>) {
        // Use the existing optimized SIMD implementation from specialized.hh
        // Load columns of a (these are reused for all result columns)
        __m128 a_col0 = _mm_setr_ps(a(0,0), a(1,0), a(2,0), a(3,0));
        __m128 a_col1 = _mm_setr_ps(a(0,1), a(1,1), a(2,1), a(3,1));
        __m128 a_col2 = _mm_setr_ps(a(0,2), a(1,2), a(2,2), a(3,2));
        __m128 a_col3 = _mm_setr_ps(a(0,3), a(1,3), a(2,3), a(3,3));
        
        // Process each column of the result
        for (size_t j = 0; j < 4; ++j) {
            // Broadcast each element of b's column j
            __m128 b0 = _mm_set1_ps(b(0, j));
            __m128 b1 = _mm_set1_ps(b(1, j));
            __m128 b2 = _mm_set1_ps(b(2, j));
            __m128 b3 = _mm_set1_ps(b(3, j));
            
            // Compute result column j = a * b_col_j
            __m128 result_col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(a_col0, b0), _mm_mul_ps(a_col1, b1)),
                _mm_add_ps(_mm_mul_ps(a_col2, b2), _mm_mul_ps(a_col3, b3))
            );
            
            // Store result column
            result(0, j) = result_col[0];
            result(1, j) = result_col[1];
            result(2, j) = result_col[2];
            result(3, j) = result_col[3];
        }
        return;
    }
#endif
    
    // Fallback implementation with temporary storage for aliasing
    T r[16];
    
    // Unrolled multiplication
    for (size_t i = 0; i < 4; ++i) {
        r[i*4 + 0] = a(i,0)*b(0,0) + a(i,1)*b(1,0) + a(i,2)*b(2,0) + a(i,3)*b(3,0);
        r[i*4 + 1] = a(i,0)*b(0,1) + a(i,1)*b(1,1) + a(i,2)*b(2,1) + a(i,3)*b(3,1);
        r[i*4 + 2] = a(i,0)*b(0,2) + a(i,1)*b(1,2) + a(i,2)*b(2,2) + a(i,3)*b(3,2);
        r[i*4 + 3] = a(i,0)*b(0,3) + a(i,1)*b(1,3) + a(i,2)*b(2,3) + a(i,3)*b(3,3);
    }
    
    // Copy to result
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result(i, j) = r[i*4 + j];
        }
    }
}

/** @} */ // end of matrix_multiplication

// =============================================================================
// Matrix Transpose
// =============================================================================

/**
 * @defgroup matrix_utilities Matrix Utility Operations
 * @ingroup DirectModule
 * @brief Common matrix operations like transpose, trace, determinant
 * @{
 */

/**
 * @brief Matrix transpose: result = transpose(m)
 * 
 * Computes the transpose of a matrix, swapping rows and columns.
 * For square matrices, supports efficient in-place transposition.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam Rows Number of rows in input matrix
 * @tparam Cols Number of columns in input matrix
 * @tparam ColumnMajor Storage layout
 * @param m Input matrix (Rows×Cols)
 * @param result Output transposed matrix (Cols×Rows)
 * 
 * @note For square matrices, can be done in-place (result = m)
 * @note Optimized for cache efficiency based on storage layout
 * 
 * @code
 * mat3x2<float> A;
 * mat2x3<float> At;
 * transpose(A, At);  // At = transpose of A
 * 
 * mat3<float> B;
 * transpose(B, B);   // In-place transpose
 * @endcode
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
void transpose(const matrix<T, Rows, Cols, ColumnMajor>& m,
               matrix<T, Cols, Rows, ColumnMajor>& result) {
    
    if constexpr (Rows == Cols) {
        // Square matrix - check for in-place transpose
        if (&m == &result) {
            // In-place transpose
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = i + 1; j < Cols; ++j) {
                    std::swap(result(i, j), result(j, i));
                }
            }
            return;
        }
    }
    
    // Out-of-place transpose
    if constexpr (ColumnMajor) {
        // Column-major: optimize cache access
        for (size_t j = 0; j < Cols; ++j) {
            EULER_LOOP_VECTORIZE
            for (size_t i = 0; i < Rows; ++i) {
                result(j, i) = m(i, j);
            }
        }
    } else {
        // Row-major: optimize cache access
        for (size_t i = 0; i < Rows; ++i) {
            EULER_LOOP_VECTORIZE
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = m(i, j);
            }
        }
    }
}

// =============================================================================
// Scalar-returning Operations
// =============================================================================

/**
 * @brief Matrix trace: returns sum of diagonal elements
 */
template<typename T, size_t N, bool ColumnMajor>
EULER_HOT T trace(const matrix<T, N, N, ColumnMajor>& m) {
    T sum = T(0);
    EULER_LOOP_VECTORIZE
    EULER_LOOP_UNROLL(4)
    for (size_t i = 0; i < N; ++i) {
        sum += m(i, i);
    }
    return sum;
}

/**
 * @brief 2x2 matrix determinant
 * 
 * Computes the determinant of a 2×2 matrix using the standard formula.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 2×2 matrix
 * @return Determinant: ad - bc
 */
template<typename T, bool ColumnMajor>
EULER_ALWAYS_INLINE T determinant(const matrix<T, 2, 2, ColumnMajor>& m) {
    return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

/**
 * @brief 3x3 matrix determinant
 * 
 * Computes the determinant of a 3×3 matrix using cofactor expansion
 * along the first row.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 3×3 matrix
 * @return Determinant value
 */
template<typename T, bool ColumnMajor>
EULER_ALWAYS_INLINE T determinant(const matrix<T, 3, 3, ColumnMajor>& m) {
    return m(0,0) * (m(1,1)*m(2,2) - m(1,2)*m(2,1)) -
           m(0,1) * (m(1,0)*m(2,2) - m(1,2)*m(2,0)) +
           m(0,2) * (m(1,0)*m(2,1) - m(1,1)*m(2,0));
}

/**
 * @brief 4x4 matrix determinant
 * 
 * Computes the determinant of a 4×4 matrix using an optimized
 * algorithm that reuses 2×2 sub-determinants for efficiency.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 4×4 matrix
 * @return Determinant value
 * 
 * @note Uses Laplace expansion with sub-determinant caching
 */
template<typename T, bool ColumnMajor>
T determinant(const matrix<T, 4, 4, ColumnMajor>& m) {
    // Compute 2x2 sub-determinants
    T d2_01_01 = m(0,0)*m(1,1) - m(0,1)*m(1,0);
    T d2_01_02 = m(0,0)*m(1,2) - m(0,2)*m(1,0);
    T d2_01_03 = m(0,0)*m(1,3) - m(0,3)*m(1,0);
    T d2_01_12 = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    T d2_01_13 = m(0,1)*m(1,3) - m(0,3)*m(1,1);
    T d2_01_23 = m(0,2)*m(1,3) - m(0,3)*m(1,2);
    
    // Compute 3x3 sub-determinants (expanding along row 3)
    T d3_012_012 = d2_01_01 * m(2,2) - d2_01_02 * m(2,1) + d2_01_12 * m(2,0);
    T d3_012_013 = d2_01_01 * m(2,3) - d2_01_03 * m(2,1) + d2_01_13 * m(2,0);
    T d3_012_023 = d2_01_02 * m(2,3) - d2_01_03 * m(2,2) + d2_01_23 * m(2,0);
    T d3_012_123 = d2_01_12 * m(2,3) - d2_01_13 * m(2,2) + d2_01_23 * m(2,1);
    
    // Final 4x4 determinant
    return d3_012_012 * m(3,3) - d3_012_013 * m(3,2) + d3_012_023 * m(3,1) - d3_012_123 * m(3,0);
}

// =============================================================================
// Matrix-Vector Operations
// =============================================================================

/**
 * @defgroup matrix_vector_ops Matrix-Vector Operations
 * @ingroup DirectModule
 * @brief Operations between matrices and vectors
 * @{
 */

/**
 * @brief Matrix-vector multiplication: result = matrix * vector
 * 
 * Multiplies a matrix by a column vector, producing a new vector.
 * Uses cache-friendly access patterns based on matrix layout.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam M Number of rows in matrix (and result vector dimension)
 * @tparam N Number of columns in matrix (and input vector dimension)
 * @tparam ColumnMajor Storage layout
 * @param mat Input matrix (M×N)
 * @param vec Input vector (N-dimensional)
 * @param result Output vector (M-dimensional)
 * 
 * @code
 * mat3x2<float> A;
 * vec2<float> v;
 * vec3<float> result;
 * mul(A, v, result);  // result = A * v
 * @endcode
 */
template<typename T, size_t M, size_t N, bool ColumnMajor>
void matvec(const matrix<T, M, N, ColumnMajor>& mat,
            const vector<T, N>& vec,
            vector<T, M>& result) {
    const T* EULER_RESTRICT pv = vec.data();
    T* EULER_RESTRICT pr = result.data();
    
    // Handle aliasing by using temporary if needed
    if (pv == pr) {
        vector<T, M> temp;
        T* EULER_RESTRICT pt = temp.data();
        
        for (size_t i = 0; i < M; ++i) {
            T sum = T(0);
            EULER_LOOP_VECTORIZE
            EULER_LOOP_UNROLL(4)
            for (size_t j = 0; j < N; ++j) {
                sum += mat(i, j) * pv[j];
            }
            pt[i] = sum;
        }
        
        // Copy result
        for (size_t i = 0; i < M; ++i) {
            pr[i] = pt[i];
        }
    } else {
        // Direct computation
        for (size_t i = 0; i < M; ++i) {
            T sum = T(0);
            EULER_LOOP_VECTORIZE
            EULER_LOOP_UNROLL(4)
            for (size_t j = 0; j < N; ++j) {
                sum += mat(i, j) * pv[j];
            }
            pr[i] = sum;
        }
    }
}

/**
 * @brief Vector-matrix multiplication: result = vector * matrix (row vector)
 * 
 * Treats the input vector as a row vector and multiplies it by a matrix.
 * Handles aliasing by using temporary storage when needed.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam M Dimension of input vector (and number of rows in matrix)
 * @tparam N Number of columns in matrix (and dimension of result vector)
 * @tparam ColumnMajor Storage layout
 * @param vec Input row vector (M-dimensional)
 * @param mat Input matrix (M×N)
 * @param result Output vector (N-dimensional)
 * 
 * @code
 * vec3<float> v;
 * mat3x2<float> A;
 * vec2<float> result;
 * vecmat(v, A, result);  // result = v * A (row vector * matrix)
 * @endcode
 */
template<typename T, size_t M, size_t N, bool ColumnMajor>
void vecmat(const vector<T, M>& vec,
            const matrix<T, M, N, ColumnMajor>& mat,
            vector<T, N>& result) {
    const T* EULER_RESTRICT pv = vec.data();
    T* EULER_RESTRICT pr = result.data();
    
    // Handle aliasing by using temporary if needed
    if (pv == pr) {
        vector<T, N> temp;
        T* EULER_RESTRICT pt = temp.data();
        
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            EULER_LOOP_VECTORIZE
            EULER_LOOP_UNROLL(4)
            for (size_t i = 0; i < M; ++i) {
                sum += pv[i] * mat(i, j);
            }
            pt[j] = sum;
        }
        
        // Copy result
        for (size_t j = 0; j < N; ++j) {
            pr[j] = pt[j];
        }
    } else {
        // Direct computation
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            EULER_LOOP_VECTORIZE
            EULER_LOOP_UNROLL(4)
            for (size_t i = 0; i < M; ++i) {
                sum += pv[i] * mat(i, j);
            }
            pr[j] = sum;
        }
    }
}

/** @} */ // end of matrix_vector_ops

// =============================================================================
// Matrix Inverse and Related Operations
// =============================================================================

/**
 * @defgroup matrix_inverse Matrix Inverse Operations
 * @ingroup DirectModule
 * @brief Matrix inversion and related operations (adjugate, cofactor)
 * @{
 */

/**
 * @brief 2x2 matrix inverse
 * 
 * Computes the inverse of a 2×2 matrix using the closed-form solution.
 * Checks for singularity before computing.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 2×2 matrix
 * @param result Output inverse matrix
 * 
 * @throws error_code::singular_matrix if determinant is near zero
 * 
 * @code
 * mat2<float> A, Ainv;
 * inverse(A, Ainv);  // Ainv * A = I
 * @endcode
 */
template<typename T, bool ColumnMajor>
void inverse(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    const T det = determinant(m);
    #ifdef EULER_DEBUG
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(10),
                error_code::singular_matrix, "Matrix is singular");
    #endif
    
    const T inv_det = T(1) / det;
    
    // Adjugate matrix for 2x2: swap diagonal, negate off-diagonal
    result(0, 0) = m(1, 1) * inv_det;
    result(0, 1) = -m(0, 1) * inv_det;
    result(1, 0) = -m(1, 0) * inv_det;
    result(1, 1) = m(0, 0) * inv_det;
}

/**
 * @brief 3x3 matrix inverse
 * 
 * Computes the inverse of a 3×3 matrix using the adjugate method.
 * Optimized for 3D graphics and physics applications.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 3×3 matrix
 * @param result Output inverse matrix
 * 
 * @throws error_code::singular_matrix if determinant is near zero
 * 
 * @code
 * mat3<float> R = rotation_matrix(angle);
 * mat3<float> Rinv;
 * inverse(R, Rinv);  // Rinv = R^-1 = R^T for rotation matrices
 * @endcode
 */
template<typename T, bool ColumnMajor>
void inverse(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
    const T det = determinant(m);
    #ifdef EULER_DEBUG
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(100),
                error_code::singular_matrix, "Matrix is singular");
    #endif
    
    const T inv_det = T(1) / det;
    
    // Compute adjugate matrix (transpose of cofactor matrix)
    T cofactors[9];
    
    // Row 0 cofactors
    cofactors[0] = m(1,1)*m(2,2) - m(1,2)*m(2,1);   // C00
    cofactors[1] = -(m(0,1)*m(2,2) - m(0,2)*m(2,1)); // C01
    cofactors[2] = m(0,1)*m(1,2) - m(0,2)*m(1,1);   // C02
    
    // Row 1 cofactors
    cofactors[3] = -(m(1,0)*m(2,2) - m(1,2)*m(2,0)); // C10
    cofactors[4] = m(0,0)*m(2,2) - m(0,2)*m(2,0);   // C11
    cofactors[5] = -(m(0,0)*m(1,2) - m(0,2)*m(1,0)); // C12
    
    // Row 2 cofactors
    cofactors[6] = m(1,0)*m(2,1) - m(1,1)*m(2,0);   // C20
    cofactors[7] = -(m(0,0)*m(2,1) - m(0,1)*m(2,0)); // C21
    cofactors[8] = m(0,0)*m(1,1) - m(0,1)*m(1,0);   // C22
    
    // Transpose and scale to get inverse
    result(0,0) = cofactors[0] * inv_det;
    result(0,1) = cofactors[1] * inv_det;
    result(0,2) = cofactors[2] * inv_det;
    result(1,0) = cofactors[3] * inv_det;
    result(1,1) = cofactors[4] * inv_det;
    result(1,2) = cofactors[5] * inv_det;
    result(2,0) = cofactors[6] * inv_det;
    result(2,1) = cofactors[7] * inv_det;
    result(2,2) = cofactors[8] * inv_det;
}

/**
 * @brief 4x4 matrix inverse using cofactor expansion
 */
template<typename T, bool ColumnMajor>
void inverse(const matrix<T, 4, 4, ColumnMajor>& m, matrix<T, 4, 4, ColumnMajor>& result) {
    // Helper lambda to compute 3x3 determinant
    auto det3x3 = [](T a00, T a01, T a02,
                     T a10, T a11, T a12,
                     T a20, T a21, T a22) -> T {
        return a00 * (a11*a22 - a12*a21) 
             - a01 * (a10*a22 - a12*a20)
             + a02 * (a10*a21 - a11*a20);
    };
    
    // Compute all 16 cofactors
    // C[i][j] = (-1)^(i+j) * M[i][j] where M[i][j] is minor with row i, col j removed
    
    // Row 0 cofactors
    const T c00 = det3x3(m(1,1), m(1,2), m(1,3),
                         m(2,1), m(2,2), m(2,3),
                         m(3,1), m(3,2), m(3,3));
    
    const T c01 = -det3x3(m(1,0), m(1,2), m(1,3),
                          m(2,0), m(2,2), m(2,3),
                          m(3,0), m(3,2), m(3,3));
    
    const T c02 = det3x3(m(1,0), m(1,1), m(1,3),
                         m(2,0), m(2,1), m(2,3),
                         m(3,0), m(3,1), m(3,3));
    
    const T c03 = -det3x3(m(1,0), m(1,1), m(1,2),
                          m(2,0), m(2,1), m(2,2),
                          m(3,0), m(3,1), m(3,2));
    
    // Compute determinant using first row
    const T det = m(0,0)*c00 + m(0,1)*c01 + m(0,2)*c02 + m(0,3)*c03;
    
    #ifdef EULER_DEBUG
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(1000),
                error_code::singular_matrix, "Matrix is singular");
    #endif
    
    const T inv_det = T(1) / det;
    
    // Row 1 cofactors
    const T c10 = -det3x3(m(0,1), m(0,2), m(0,3),
                          m(2,1), m(2,2), m(2,3),
                          m(3,1), m(3,2), m(3,3));
    
    const T c11 = det3x3(m(0,0), m(0,2), m(0,3),
                         m(2,0), m(2,2), m(2,3),
                         m(3,0), m(3,2), m(3,3));
    
    const T c12 = -det3x3(m(0,0), m(0,1), m(0,3),
                          m(2,0), m(2,1), m(2,3),
                          m(3,0), m(3,1), m(3,3));
    
    const T c13 = det3x3(m(0,0), m(0,1), m(0,2),
                         m(2,0), m(2,1), m(2,2),
                         m(3,0), m(3,1), m(3,2));
    
    // Row 2 cofactors
    const T c20 = det3x3(m(0,1), m(0,2), m(0,3),
                         m(1,1), m(1,2), m(1,3),
                         m(3,1), m(3,2), m(3,3));
    
    const T c21 = -det3x3(m(0,0), m(0,2), m(0,3),
                          m(1,0), m(1,2), m(1,3),
                          m(3,0), m(3,2), m(3,3));
    
    const T c22 = det3x3(m(0,0), m(0,1), m(0,3),
                         m(1,0), m(1,1), m(1,3),
                         m(3,0), m(3,1), m(3,3));
    
    const T c23 = -det3x3(m(0,0), m(0,1), m(0,2),
                          m(1,0), m(1,1), m(1,2),
                          m(3,0), m(3,1), m(3,2));
    
    // Row 3 cofactors
    const T c30 = -det3x3(m(0,1), m(0,2), m(0,3),
                          m(1,1), m(1,2), m(1,3),
                          m(2,1), m(2,2), m(2,3));
    
    const T c31 = det3x3(m(0,0), m(0,2), m(0,3),
                         m(1,0), m(1,2), m(1,3),
                         m(2,0), m(2,2), m(2,3));
    
    const T c32 = -det3x3(m(0,0), m(0,1), m(0,3),
                          m(1,0), m(1,1), m(1,3),
                          m(2,0), m(2,1), m(2,3));
    
    const T c33 = det3x3(m(0,0), m(0,1), m(0,2),
                         m(1,0), m(1,1), m(1,2),
                         m(2,0), m(2,1), m(2,2));
    
    // Fill result with transpose of cofactor matrix / det
    // result[i][j] = cofactor[j][i] / det
    result(0,0) = c00 * inv_det;
    result(0,1) = c10 * inv_det;
    result(0,2) = c20 * inv_det;
    result(0,3) = c30 * inv_det;
    
    result(1,0) = c01 * inv_det;
    result(1,1) = c11 * inv_det;
    result(1,2) = c21 * inv_det;
    result(1,3) = c31 * inv_det;
    
    result(2,0) = c02 * inv_det;
    result(2,1) = c12 * inv_det;
    result(2,2) = c22 * inv_det;
    result(2,3) = c32 * inv_det;
    
    result(3,0) = c03 * inv_det;
    result(3,1) = c13 * inv_det;
    result(3,2) = c23 * inv_det;
    result(3,3) = c33 * inv_det;
}

/**
 * @brief Compute adjugate matrix (transpose of cofactor matrix)
 * @note For 2x2 matrices
 */
template<typename T, bool ColumnMajor>
void adjugate(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    result(0, 0) = m(1, 1);
    result(0, 1) = -m(0, 1);
    result(1, 0) = -m(1, 0);
    result(1, 1) = m(0, 0);
}

/**
 * @brief Compute adjugate matrix (transpose of cofactor matrix)
 * @note For 3x3 matrices
 */
template<typename T, bool ColumnMajor>
void adjugate(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
    // Compute cofactor matrix and transpose in one step
    result(0,0) = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    result(0,1) = -(m(0,1)*m(2,2) - m(0,2)*m(2,1));
    result(0,2) = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    
    result(1,0) = -(m(1,0)*m(2,2) - m(1,2)*m(2,0));
    result(1,1) = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    result(1,2) = -(m(0,0)*m(1,2) - m(0,2)*m(1,0));
    
    result(2,0) = m(1,0)*m(2,1) - m(1,1)*m(2,0);
    result(2,1) = -(m(0,0)*m(2,1) - m(0,1)*m(2,0));
    result(2,2) = m(0,0)*m(1,1) - m(0,1)*m(1,0);
}

/**
 * @brief Compute cofactor matrix
 * @note For 2x2 matrices
 */
template<typename T, bool ColumnMajor>
void cofactor(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    result(0, 0) = m(1, 1);
    result(0, 1) = -m(1, 0);
    result(1, 0) = -m(0, 1);
    result(1, 1) = m(0, 0);
}

/**
 * @brief Compute cofactor matrix
 * 
 * Computes the cofactor matrix for a 3×3 matrix, where each element
 * is the signed minor of the corresponding element.
 * 
 * @tparam T Element type (float, double, etc.)
 * @tparam ColumnMajor Storage layout
 * @param m Input 3×3 matrix
 * @param result Output cofactor matrix
 * 
 * @note For 3x3 matrices
 * @note The transpose of the cofactor matrix is the adjugate matrix
 */
template<typename T, bool ColumnMajor>
void cofactor(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
    // Row 0 cofactors
    result(0,0) = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    result(0,1) = -(m(1,0)*m(2,2) - m(1,2)*m(2,0));
    result(0,2) = m(1,0)*m(2,1) - m(1,1)*m(2,0);
    
    // Row 1 cofactors
    result(1,0) = -(m(0,1)*m(2,2) - m(0,2)*m(2,1));
    result(1,1) = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    result(1,2) = -(m(0,0)*m(2,1) - m(0,1)*m(2,0));
    
    // Row 2 cofactors
    result(2,0) = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    result(2,1) = -(m(0,0)*m(1,2) - m(0,2)*m(1,0));
    result(2,2) = m(0,0)*m(1,1) - m(0,1)*m(1,0);
}

/** @} */ // end of matrix_inverse

/** @} */ // end of matrix_utilities

} // namespace euler::direct