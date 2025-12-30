#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/core/types.hh>

namespace euler {

// Specialized operations for small matrices with unrolled loops

// 2x2 Matrix multiplication (unrolled)
template<typename T>
constexpr matrix<T, 2, 2> multiply_2x2(const matrix<T, 2, 2>& a, const matrix<T, 2, 2>& b) {
    return matrix<T, 2, 2>{
        {a(0,0) * b(0,0) + a(0,1) * b(1,0), a(0,0) * b(0,1) + a(0,1) * b(1,1)},
        {a(1,0) * b(0,0) + a(1,1) * b(1,0), a(1,0) * b(0,1) + a(1,1) * b(1,1)}
    };
}

// 3x3 Matrix multiplication (unrolled)
template<typename T>
constexpr matrix<T, 3, 3> multiply_3x3(const matrix<T, 3, 3>& a, const matrix<T, 3, 3>& b) {
    return matrix<T, 3, 3>{
        {a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0),
         a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1),
         a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2)},
        {a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0),
         a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1),
         a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2)},
        {a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0),
         a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1),
         a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2)}
    };
}

// 4x4 Matrix multiplication (unrolled)
template<typename T>
constexpr matrix<T, 4, 4> multiply_4x4(const matrix<T, 4, 4>& a, const matrix<T, 4, 4>& b) {
    return matrix<T, 4, 4>{
        {a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0) + a(0,3)*b(3,0),
         a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1) + a(0,3)*b(3,1),
         a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2) + a(0,3)*b(3,2),
         a(0,0)*b(0,3) + a(0,1)*b(1,3) + a(0,2)*b(2,3) + a(0,3)*b(3,3)},
        {a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0) + a(1,3)*b(3,0),
         a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1) + a(1,3)*b(3,1),
         a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2) + a(1,3)*b(3,2),
         a(1,0)*b(0,3) + a(1,1)*b(1,3) + a(1,2)*b(2,3) + a(1,3)*b(3,3)},
        {a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0) + a(2,3)*b(3,0),
         a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1) + a(2,3)*b(3,1),
         a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2) + a(2,3)*b(3,2),
         a(2,0)*b(0,3) + a(2,1)*b(1,3) + a(2,2)*b(2,3) + a(2,3)*b(3,3)},
        {a(3,0)*b(0,0) + a(3,1)*b(1,0) + a(3,2)*b(2,0) + a(3,3)*b(3,0),
         a(3,0)*b(0,1) + a(3,1)*b(1,1) + a(3,2)*b(2,1) + a(3,3)*b(3,1),
         a(3,0)*b(0,2) + a(3,1)*b(1,2) + a(3,2)*b(2,2) + a(3,3)*b(3,2),
         a(3,0)*b(0,3) + a(3,1)*b(1,3) + a(3,2)*b(2,3) + a(3,3)*b(3,3)}
    };
}

// Fast 2x2 determinant
template<typename T>
constexpr T fast_determinant_2x2(const matrix<T, 2, 2>& m) {
    return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

// Fast 3x3 determinant
template<typename T>
constexpr T fast_determinant_3x3(const matrix<T, 3, 3>& m) {
    return m(0,0) * (m(1,1) * m(2,2) - m(1,2) * m(2,1)) -
           m(0,1) * (m(1,0) * m(2,2) - m(1,2) * m(2,0)) +
           m(0,2) * (m(1,0) * m(2,1) - m(1,1) * m(2,0));
}

// Fast 4x4 determinant using sub-determinants
template<typename T>
constexpr T fast_determinant_4x4(const matrix<T, 4, 4>& m) {
    // Pre-compute 2x2 sub-determinants for efficiency
    T sub23_01 = m(2,0) * m(3,1) - m(2,1) * m(3,0);
    T sub23_02 = m(2,0) * m(3,2) - m(2,2) * m(3,0);
    T sub23_03 = m(2,0) * m(3,3) - m(2,3) * m(3,0);
    T sub23_12 = m(2,1) * m(3,2) - m(2,2) * m(3,1);
    T sub23_13 = m(2,1) * m(3,3) - m(2,3) * m(3,1);
    T sub23_23 = m(2,2) * m(3,3) - m(2,3) * m(3,2);

    // Compute 3x3 cofactors
    T cof0 = m(1,1) * sub23_23 - m(1,2) * sub23_13 + m(1,3) * sub23_12;
    T cof1 = m(1,0) * sub23_23 - m(1,2) * sub23_03 + m(1,3) * sub23_02;
    T cof2 = m(1,0) * sub23_13 - m(1,1) * sub23_03 + m(1,3) * sub23_01;
    T cof3 = m(1,0) * sub23_12 - m(1,1) * sub23_02 + m(1,2) * sub23_01;

    // Final determinant
    return m(0,0) * cof0 - m(0,1) * cof1 + m(0,2) * cof2 - m(0,3) * cof3;
}

// Fast 2x2 inverse
template<typename T>
constexpr matrix<T, 2, 2> fast_inverse_2x2(const matrix<T, 2, 2>& m) {
    T det = fast_determinant_2x2(m);
    T inv_det = T(1) / det;

    return matrix<T, 2, 2>{
        { m(1,1) * inv_det, -m(0,1) * inv_det},
        {-m(1,0) * inv_det,  m(0,0) * inv_det}
    };
}

// Fast 3x3 inverse
template<typename T>
constexpr matrix<T, 3, 3> fast_inverse_3x3(const matrix<T, 3, 3>& m) {
    T det = fast_determinant_3x3(m);
    T inv_det = T(1) / det;

    return matrix<T, 3, 3>{
        {(m(1,1) * m(2,2) - m(1,2) * m(2,1)) * inv_det,
         -(m(0,1) * m(2,2) - m(0,2) * m(2,1)) * inv_det,
         (m(0,1) * m(1,2) - m(0,2) * m(1,1)) * inv_det},
        {-(m(1,0) * m(2,2) - m(1,2) * m(2,0)) * inv_det,
         (m(0,0) * m(2,2) - m(0,2) * m(2,0)) * inv_det,
         -(m(0,0) * m(1,2) - m(0,2) * m(1,0)) * inv_det},
        {(m(1,0) * m(2,1) - m(1,1) * m(2,0)) * inv_det,
         -(m(0,0) * m(2,1) - m(0,1) * m(2,0)) * inv_det,
         (m(0,0) * m(1,1) - m(0,1) * m(1,0)) * inv_det}
    };
}

// Fast 4x4 inverse using Gauss-Jordan elimination with partial pivoting
template<typename T>
auto fast_inverse_4x4(const matrix<T, 4, 4>& m) {
    // Create augmented matrix [A|I]
    T aug[4][8];

    // Initialize augmented matrix
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            aug[i][j] = m(i,j);
            aug[i][j+4] = (i == j) ? T(1) : T(0);
        }
    }

    // Gauss-Jordan elimination
    for (size_t i = 0; i < 4; ++i) {
        // Find pivot
        size_t pivot = i;
        T max_val = std::abs(aug[i][i]);
        for (size_t k = i + 1; k < 4; ++k) {
            if (std::abs(aug[k][i]) > max_val) {
                max_val = std::abs(aug[k][i]);
                pivot = k;
            }
        }

        // Swap rows if needed
        if (pivot != i) {
            for (size_t j = 0; j < 8; ++j) {
                std::swap(aug[i][j], aug[pivot][j]);
            }
        }

        // Scale pivot row
        T pivot_val = aug[i][i];
        if (std::abs(pivot_val) < constants<T>::epsilon) {
            EULER_CHECK(false, error_code::singular_matrix, "Matrix is singular");
        }

        T inv_pivot = T(1) / pivot_val;
        for (size_t j = 0; j < 8; ++j) {
            aug[i][j] *= inv_pivot;
        }

        // Eliminate column
        for (size_t k = 0; k < 4; ++k) {
            if (k != i) {
                T factor = aug[k][i];
                for (size_t j = 0; j < 8; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    matrix<T, 4, 4> result;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result(i,j) = aug[i][j+4];
        }
    }

    return result;
}

// Fast transpose for 4x4 matrices
template<typename T>
constexpr matrix<T, 4, 4> fast_transpose_4x4(const matrix<T, 4, 4>& m) {
    return matrix<T, 4, 4>{
        {m(0,0), m(1,0), m(2,0), m(3,0)},
        {m(0,1), m(1,1), m(2,1), m(3,1)},
        {m(0,2), m(1,2), m(2,2), m(3,2)},
        {m(0,3), m(1,3), m(2,3), m(3,3)}
    };
}

// Specialization selector based on matrix size
template<typename T, size_t M, size_t N, size_t P, bool RowMajor1, bool RowMajor2>
constexpr auto multiply_specialized(const matrix<T, M, N, RowMajor1>& a, const matrix<T, N, P, RowMajor2>& b) {
    if constexpr (M == 2 && N == 2 && P == 2) {
        return multiply_2x2(a, b);
    } else if constexpr (M == 3 && N == 3 && P == 3) {
        return multiply_3x3(a, b);
    } else if constexpr (M == 4 && N == 4 && P == 4) {
        return multiply_4x4(a, b);
    } else {
        // Generic fallback
        matrix<T, M, P, RowMajor1> result;
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < N; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
}

// Batch matrix operations for animation/physics
template<typename T>
class matrix_batch_processor {
public:
    // Process multiple 4x4 matrix multiplications
    static void multiply_4x4_batch(const matrix<T, 4, 4>* a_array,
                                  const matrix<T, 4, 4>* b_array,
                                  matrix<T, 4, 4>* result_array,
                                  size_t count) {
        for (size_t i = 0; i < count; ++i) {
            result_array[i] = multiply_4x4(a_array[i], b_array[i]);
        }
    }

    // Batch transpose operations
    static void transpose_4x4_batch(const matrix<T, 4, 4>* input,
                                   matrix<T, 4, 4>* output,
                                   size_t count) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = fast_transpose_4x4(input[i]);
        }
    }
};

} // namespace euler
