/**
 * @file matrix_ops.hh
 * @brief Direct operations for matrices
 * @ingroup DirectModule
 *
 * This header provides high-performance direct operations on matrices
 * that bypass the expression template system for maximum performance.
 *
 * @section matrix_ops_features Key Features
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
 * mul(A, B, C);           // C = A * B (matrix product)
 *
 * // In-place operations
 * add(A, B, A);           // A = A + B (aliasing safe)
 * @endcode
 */
#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_view.hh>
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
 * @brief Matrix addition: result = op1 + op2
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void add(const matrix<T, Rows, Cols, ColumnMajor>& op1,
                            const matrix<T, Rows, Cols, ColumnMajor>& op2,
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT p1 = op1.data();
    const T* EULER_RESTRICT p2 = op2.data();
    T* EULER_RESTRICT pr = result.data();

    constexpr size_t size = Rows * Cols;

    EULER_LOOP_VECTORIZE
    EULER_LOOP_UNROLL(8)
    for (size_t i = 0; i < size; ++i) {
        pr[i] = p1[i] + p2[i];
    }
}

/**
 * @brief Matrix subtraction: result = op1 - op2
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void sub(const matrix<T, Rows, Cols, ColumnMajor>& op1,
                            const matrix<T, Rows, Cols, ColumnMajor>& op2,
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT p1 = op1.data();
    const T* EULER_RESTRICT p2 = op2.data();
    T* EULER_RESTRICT pr = result.data();

    constexpr size_t size = Rows * Cols;

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < size; ++i) {
        pr[i] = p1[i] - p2[i];
    }
}

// =============================================================================
// Scalar Operations
// =============================================================================

/**
 * @brief Scalar multiplication: result = scalar * matrix
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_HOT void scale(const matrix<T, Rows, Cols, ColumnMajor>& m, T scalar,
                    matrix<T, Rows, Cols, ColumnMajor>& result) {
    const T* EULER_RESTRICT pm = m.data();
    T* EULER_RESTRICT pr = result.data();

    constexpr size_t size = Rows * Cols;

    EULER_LOOP_VECTORIZE
    for (size_t i = 0; i < size; ++i) {
        pr[i] = scalar * pm[i];
    }
}

/**
 * @brief Scalar multiplication (alias): result = scalar * matrix
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void mul(T scalar, const matrix<T, Rows, Cols, ColumnMajor>& m,
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    scale(m, scalar, result);
}

/**
 * @brief Scalar multiplication: result = matrix * scalar
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
EULER_ALWAYS_INLINE void mul(const matrix<T, Rows, Cols, ColumnMajor>& m, T scalar,
                            matrix<T, Rows, Cols, ColumnMajor>& result) {
    scale(m, scalar, result);
}

// =============================================================================
// Matrix Multiplication
// =============================================================================

/**
 * @brief General matrix multiplication: result = a * b
 */
template<typename T, size_t M, size_t N, size_t K, bool ColumnMajor>
void mul(const matrix<T, M, K, ColumnMajor>& a,
         const matrix<T, K, N, ColumnMajor>& b,
         matrix<T, M, N, ColumnMajor>& result) {
    // Check for aliasing
    bool aliased = false;
    if constexpr (N == K) {
        aliased = (static_cast<const void*>(&result) == static_cast<const void*>(&a));
    }
    if constexpr (M == K) {
        aliased = aliased || (static_cast<const void*>(&result) == static_cast<const void*>(&b));
    }

    if (aliased) {
        matrix<T, M, N, ColumnMajor> temp;

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

        result = std::move(temp);
    } else {
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
    const T a00 = a(0,0), a01 = a(0,1);
    const T a10 = a(1,0), a11 = a(1,1);
    const T b00 = b(0,0), b01 = b(0,1);
    const T b10 = b(1,0), b11 = b(1,1);

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
    T r[9];

    r[0] = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
    r[1] = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
    r[2] = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);

    r[3] = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
    r[4] = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
    r[5] = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);

    r[6] = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
    r[7] = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
    r[8] = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);

    result(0,0) = r[0]; result(0,1) = r[1]; result(0,2) = r[2];
    result(1,0) = r[3]; result(1,1) = r[4]; result(1,2) = r[5];
    result(2,0) = r[6]; result(2,1) = r[7]; result(2,2) = r[8];
}

/**
 * @brief Optimized 4x4 matrix multiplication
 */
template<typename T, bool ColumnMajor>
EULER_HOT void mul(const matrix<T, 4, 4, ColumnMajor>& a,
                   const matrix<T, 4, 4, ColumnMajor>& b,
                   matrix<T, 4, 4, ColumnMajor>& result) {
    T r[16];

    for (size_t i = 0; i < 4; ++i) {
        r[i*4 + 0] = a(i,0)*b(0,0) + a(i,1)*b(1,0) + a(i,2)*b(2,0) + a(i,3)*b(3,0);
        r[i*4 + 1] = a(i,0)*b(0,1) + a(i,1)*b(1,1) + a(i,2)*b(2,1) + a(i,3)*b(3,1);
        r[i*4 + 2] = a(i,0)*b(0,2) + a(i,1)*b(1,2) + a(i,2)*b(2,2) + a(i,3)*b(3,2);
        r[i*4 + 3] = a(i,0)*b(0,3) + a(i,1)*b(1,3) + a(i,2)*b(2,3) + a(i,3)*b(3,3);
    }

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result(i, j) = r[i*4 + j];
        }
    }
}

// =============================================================================
// Matrix Transpose
// =============================================================================

/**
 * @brief Matrix transpose: result = transpose(m)
 */
template<typename T, size_t Rows, size_t Cols, bool ColumnMajor>
void transpose(const matrix<T, Rows, Cols, ColumnMajor>& m,
               matrix<T, Cols, Rows, ColumnMajor>& result) {

    if constexpr (Rows == Cols) {
        if (&m == &result) {
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = i + 1; j < Cols; ++j) {
                    std::swap(result(i, j), result(j, i));
                }
            }
            return;
        }
    }

    if constexpr (ColumnMajor) {
        for (size_t j = 0; j < Cols; ++j) {
            EULER_LOOP_VECTORIZE
            for (size_t i = 0; i < Rows; ++i) {
                result(j, i) = m(i, j);
            }
        }
    } else {
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
[[nodiscard]] EULER_HOT T trace(const matrix<T, N, N, ColumnMajor>& m) {
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
 */
template<typename T, bool ColumnMajor>
[[nodiscard]] EULER_ALWAYS_INLINE T determinant(const matrix<T, 2, 2, ColumnMajor>& m) {
    return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

/**
 * @brief 3x3 matrix determinant
 */
template<typename T, bool ColumnMajor>
[[nodiscard]] EULER_ALWAYS_INLINE T determinant(const matrix<T, 3, 3, ColumnMajor>& m) {
    return m(0,0) * (m(1,1)*m(2,2) - m(1,2)*m(2,1)) -
           m(0,1) * (m(1,0)*m(2,2) - m(1,2)*m(2,0)) +
           m(0,2) * (m(1,0)*m(2,1) - m(1,1)*m(2,0));
}

/**
 * @brief 4x4 matrix determinant
 */
template<typename T, bool ColumnMajor>
[[nodiscard]] T determinant(const matrix<T, 4, 4, ColumnMajor>& m) {
    T d2_01_01 = m(0,0)*m(1,1) - m(0,1)*m(1,0);
    T d2_01_02 = m(0,0)*m(1,2) - m(0,2)*m(1,0);
    T d2_01_03 = m(0,0)*m(1,3) - m(0,3)*m(1,0);
    T d2_01_12 = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    T d2_01_13 = m(0,1)*m(1,3) - m(0,3)*m(1,1);
    T d2_01_23 = m(0,2)*m(1,3) - m(0,3)*m(1,2);

    T d3_012_012 = d2_01_01 * m(2,2) - d2_01_02 * m(2,1) + d2_01_12 * m(2,0);
    T d3_012_013 = d2_01_01 * m(2,3) - d2_01_03 * m(2,1) + d2_01_13 * m(2,0);
    T d3_012_023 = d2_01_02 * m(2,3) - d2_01_03 * m(2,2) + d2_01_23 * m(2,0);
    T d3_012_123 = d2_01_12 * m(2,3) - d2_01_13 * m(2,2) + d2_01_23 * m(2,1);

    return d3_012_012 * m(3,3) - d3_012_013 * m(3,2) + d3_012_023 * m(3,1) - d3_012_123 * m(3,0);
}

// =============================================================================
// Matrix-Vector Operations
// =============================================================================

/**
 * @brief Matrix-vector multiplication: result = matrix * vector
 */
template<typename T, size_t M, size_t N, bool ColumnMajor>
void matvec(const matrix<T, M, N, ColumnMajor>& mat,
            const vector<T, N>& vec,
            vector<T, M>& result) {
    const T* EULER_RESTRICT pv = vec.data();
    T* EULER_RESTRICT pr = result.data();

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

        for (size_t i = 0; i < M; ++i) {
            pr[i] = pt[i];
        }
    } else {
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
 */
template<typename T, size_t M, size_t N, bool ColumnMajor>
void vecmat(const vector<T, M>& vec,
            const matrix<T, M, N, ColumnMajor>& mat,
            vector<T, N>& result) {
    const T* EULER_RESTRICT pv = vec.data();
    T* EULER_RESTRICT pr = result.data();

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

        for (size_t j = 0; j < N; ++j) {
            pr[j] = pt[j];
        }
    } else {
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

// =============================================================================
// Matrix Inverse and Related Operations
// =============================================================================

/**
 * @brief 2x2 matrix inverse
 *
 * @note The singularity check uses epsilon * 10 as threshold. The multiplier
 *       accounts for numerical error accumulation in the determinant calculation
 *       (2 multiplications, 1 subtraction). Larger matrices use larger multipliers:
 *       2x2: 10, 3x3: 100, 4x4: 1000.
 */
template<typename T, bool ColumnMajor>
void inverse(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    const T det = determinant(m);
    // Tolerance scales with matrix size due to accumulated numerical error
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(10),
                error_code::singular_matrix, "Matrix is singular");

    const T inv_det = T(1) / det;

    result(0, 0) = m(1, 1) * inv_det;
    result(0, 1) = -m(0, 1) * inv_det;
    result(1, 0) = -m(1, 0) * inv_det;
    result(1, 1) = m(0, 0) * inv_det;
}

/**
 * @brief 3x3 matrix inverse
 */
template<typename T, bool ColumnMajor>
void inverse(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
    const T det = determinant(m);
    // Tolerance scales with matrix size due to accumulated numerical error
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(100),
                error_code::singular_matrix, "Matrix is singular");

    const T inv_det = T(1) / det;

    T cofactors[9];

    cofactors[0] = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    cofactors[1] = -(m(0,1)*m(2,2) - m(0,2)*m(2,1));
    cofactors[2] = m(0,1)*m(1,2) - m(0,2)*m(1,1);

    cofactors[3] = -(m(1,0)*m(2,2) - m(1,2)*m(2,0));
    cofactors[4] = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    cofactors[5] = -(m(0,0)*m(1,2) - m(0,2)*m(1,0));

    cofactors[6] = m(1,0)*m(2,1) - m(1,1)*m(2,0);
    cofactors[7] = -(m(0,0)*m(2,1) - m(0,1)*m(2,0));
    cofactors[8] = m(0,0)*m(1,1) - m(0,1)*m(1,0);

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
    auto det3x3 = [](T a00, T a01, T a02,
                     T a10, T a11, T a12,
                     T a20, T a21, T a22) -> T {
        return a00 * (a11*a22 - a12*a21)
             - a01 * (a10*a22 - a12*a20)
             + a02 * (a10*a21 - a11*a20);
    };

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

    const T det = m(0,0)*c00 + m(0,1)*c01 + m(0,2)*c02 + m(0,3)*c03;

    // Tolerance scales with matrix size due to accumulated numerical error
    EULER_CHECK(std::abs(det) > std::numeric_limits<T>::epsilon() * T(1000),
                error_code::singular_matrix, "Matrix is singular");

    const T inv_det = T(1) / det;

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
 * @brief Compute adjugate matrix (2x2)
 */
template<typename T, bool ColumnMajor>
void adjugate(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    result(0, 0) = m(1, 1);
    result(0, 1) = -m(0, 1);
    result(1, 0) = -m(1, 0);
    result(1, 1) = m(0, 0);
}

/**
 * @brief Compute adjugate matrix (3x3)
 */
template<typename T, bool ColumnMajor>
void adjugate(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
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
 * @brief Compute cofactor matrix (2x2)
 */
template<typename T, bool ColumnMajor>
void cofactor(const matrix<T, 2, 2, ColumnMajor>& m, matrix<T, 2, 2, ColumnMajor>& result) {
    result(0, 0) = m(1, 1);
    result(0, 1) = -m(1, 0);
    result(1, 0) = -m(0, 1);
    result(1, 1) = m(0, 0);
}

/**
 * @brief Compute cofactor matrix (3x3)
 */
template<typename T, bool ColumnMajor>
void cofactor(const matrix<T, 3, 3, ColumnMajor>& m, matrix<T, 3, 3, ColumnMajor>& result) {
    result(0,0) = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    result(0,1) = -(m(1,0)*m(2,2) - m(1,2)*m(2,0));
    result(0,2) = m(1,0)*m(2,1) - m(1,1)*m(2,0);

    result(1,0) = -(m(0,1)*m(2,2) - m(0,2)*m(2,1));
    result(1,1) = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    result(1,2) = -(m(0,0)*m(2,1) - m(0,1)*m(2,0));

    result(2,0) = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    result(2,1) = -(m(0,0)*m(1,2) - m(0,2)*m(1,0));
    result(2,2) = m(0,0)*m(1,1) - m(0,1)*m(1,0);
}

} // namespace euler::direct
