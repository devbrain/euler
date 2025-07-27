#pragma once

#include <euler/matrix/matrix.hh>
#include <euler/core/simd.hh>
#include <euler/core/types.hh>
#ifdef EULER_HAS_SIMD
#include <immintrin.h>
#endif

namespace euler {

// Specialized operations for small matrices using SIMD intrinsics

// 2x2 Matrix multiplication with SIMD
template<typename T>
auto multiply_2x2_simd(const matrix<T, 2, 2>& a, const matrix<T, 2, 2>& b) 
    -> std::enable_if_t<std::is_same_v<T, float>, matrix<T, 2, 2>> {
    matrix<T, 2, 2> result;
    
#ifdef EULER_HAS_SIMD
    // Load matrix a rows
    __m128 a_row0 = _mm_set_ps(0, 0, a(0,1), a(0,0));
    __m128 a_row1 = _mm_set_ps(0, 0, a(1,1), a(1,0));
    
    // Compute result row 0
    __m128 b_col0 = _mm_set_ps(0, 0, b(1,0), b(0,0));
    __m128 b_col1 = _mm_set_ps(0, 0, b(1,1), b(0,1));
    
    __m128 prod0 = _mm_mul_ps(a_row0, b_col0);
    __m128 prod1 = _mm_mul_ps(a_row0, b_col1);
    
    result(0,0) = prod0[0] + prod0[1];
    result(0,1) = prod1[0] + prod1[1];
    
    // Compute result row 1
    prod0 = _mm_mul_ps(a_row1, b_col0);
    prod1 = _mm_mul_ps(a_row1, b_col1);
    
    result(1,0) = prod0[0] + prod0[1];
    result(1,1) = prod1[0] + prod1[1];
#else
    // Fallback to standard multiplication
    result(0,0) = a(0,0) * b(0,0) + a(0,1) * b(1,0);
    result(0,1) = a(0,0) * b(0,1) + a(0,1) * b(1,1);
    result(1,0) = a(1,0) * b(0,0) + a(1,1) * b(1,0);
    result(1,1) = a(1,0) * b(0,1) + a(1,1) * b(1,1);
#endif
    
    return result;
}

// 3x3 Matrix multiplication with SIMD
template<typename T>
auto multiply_3x3_simd(const matrix<T, 3, 3>& a, const matrix<T, 3, 3>& b) 
    -> std::enable_if_t<std::is_same_v<T, float>, matrix<T, 3, 3>> {
    matrix<T, 3, 3> result;
    
#ifdef EULER_HAS_SIMD
    // Process each row of result
    for (size_t i = 0; i < 3; ++i) {
        __m128 row = _mm_set_ps(0, a(i,2), a(i,1), a(i,0));
        
        for (size_t j = 0; j < 3; ++j) {
            __m128 col = _mm_set_ps(0, b(2,j), b(1,j), b(0,j));
            __m128 prod = _mm_mul_ps(row, col);
            
            // Horizontal add
            __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 sums = _mm_add_ps(prod, shuf);
            shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 3, 0, 1));
            sums = _mm_add_ps(sums, shuf);
            
            result(i,j) = _mm_cvtss_f32(sums);
        }
    }
#else
    // Standard multiplication
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < 3; ++k) {
                sum += a(i,k) * b(k,j);
            }
            result(i,j) = sum;
        }
    }
#endif
    
    return result;
}

// 4x4 Matrix multiplication with SIMD (most common in graphics)
template<typename T>
auto multiply_4x4_simd(const matrix<T, 4, 4>& a, const matrix<T, 4, 4>& b) 
    -> std::enable_if_t<std::is_same_v<T, float>, matrix<T, 4, 4>> {
    matrix<T, 4, 4> result;
    
#ifdef EULER_HAS_SIMD
    // More efficient approach: compute all 4 columns at once
    // Since matrices are column-major, b's columns are contiguous
    
    // Load all columns of b
    __m128 b_col0 = _mm_setr_ps(b(0,0), b(1,0), b(2,0), b(3,0));
    __m128 b_col1 = _mm_setr_ps(b(0,1), b(1,1), b(2,1), b(3,1));
    __m128 b_col2 = _mm_setr_ps(b(0,2), b(1,2), b(2,2), b(3,2));
    __m128 b_col3 = _mm_setr_ps(b(0,3), b(1,3), b(2,3), b(3,3));
    
    // Load columns of a (these are reused for all result columns)
    __m128 a_col0 = _mm_setr_ps(a(0,0), a(1,0), a(2,0), a(3,0));
    __m128 a_col1 = _mm_setr_ps(a(0,1), a(1,1), a(2,1), a(3,1));
    __m128 a_col2 = _mm_setr_ps(a(0,2), a(1,2), a(2,2), a(3,2));
    __m128 a_col3 = _mm_setr_ps(a(0,3), a(1,3), a(2,3), a(3,3));
    
    // Process each column of the result
    for (size_t j = 0; j < 4; ++j) {
        // Select the right column of b
        __m128 b_col;
        switch(j) {
            case 0: b_col = b_col0; break;
            case 1: b_col = b_col1; break;
            case 2: b_col = b_col2; break;
            case 3: b_col = b_col3; break;
            default: b_col = b_col0; break; // Should never happen
        }
        
        // Compute result column j by linear combination of a's columns
        // result(:,j) = a(:,0)*b(0,j) + a(:,1)*b(1,j) + a(:,2)*b(2,j) + a(:,3)*b(3,j)
        
        // Broadcast each element of b_col and multiply with corresponding column of a
        __m128 result_col = _mm_mul_ps(a_col0, _mm_shuffle_ps(b_col, b_col, 0x00));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(a_col1, _mm_shuffle_ps(b_col, b_col, 0x55)));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(a_col2, _mm_shuffle_ps(b_col, b_col, 0xAA)));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(a_col3, _mm_shuffle_ps(b_col, b_col, 0xFF)));
        
        // Store result column
        result(0,j) = result_col[0];
        result(1,j) = result_col[1];
        result(2,j) = result_col[2];
        result(3,j) = result_col[3];
    }
#else
    // Standard multiplication
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < 4; ++k) {
                sum += a(i,k) * b(k,j);
            }
            result(i,j) = sum;
        }
    }
#endif
    
    return result;
}

// Fast 2x2 determinant
template<typename T>
constexpr T fast_determinant_2x2(const matrix<T, 2, 2>& m) {
    return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

// Fast 3x3 determinant with co-factor expansion
template<typename T>
constexpr T fast_determinant_3x3(const matrix<T, 3, 3>& m) {
#ifdef EULER_HAS_SIMD
    if constexpr (std::is_same_v<T, float>) {
        // SIMD optimized version
        __m128 m0 = _mm_set_ps(0, m(0,2), m(0,1), m(0,0));
        __m128 m1 = _mm_set_ps(0, m(1,2), m(1,1), m(1,0));
        __m128 m2 = _mm_set_ps(0, m(2,2), m(2,1), m(2,0));
        
        // Compute cross products
        __m128 c0 = _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(m1, m1, 0xC9), _mm_shuffle_ps(m2, m2, 0xD2)),
            _mm_mul_ps(_mm_shuffle_ps(m1, m1, 0xD2), _mm_shuffle_ps(m2, m2, 0xC9))
        );
        
        // Dot product with first row
        __m128 prod = _mm_mul_ps(m0, c0);
        
        // Horizontal add
        __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 sums = _mm_add_ps(prod, shuf);
        shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(2, 3, 0, 1));
        sums = _mm_add_ps(sums, shuf);
        
        return _mm_cvtss_f32(sums);
    } else 
#endif
    {
        // Standard computation
        return m(0,0) * (m(1,1) * m(2,2) - m(1,2) * m(2,1)) -
               m(0,1) * (m(1,0) * m(2,2) - m(1,2) * m(2,0)) +
               m(0,2) * (m(1,0) * m(2,1) - m(1,1) * m(2,0));
    }
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
auto fast_inverse_2x2(const matrix<T, 2, 2>& m) {
    T det = fast_determinant_2x2(m);
    T inv_det = T(1) / det;
    
    return matrix<T, 2, 2>{
        { m(1,1) * inv_det, -m(0,1) * inv_det},
        {-m(1,0) * inv_det,  m(0,0) * inv_det}
    };
}

// Fast 3x3 inverse with SIMD
template<typename T>
auto fast_inverse_3x3(const matrix<T, 3, 3>& m) {
    T det = fast_determinant_3x3(m);
    T inv_det = T(1) / det;
    
    matrix<T, 3, 3> result;
    
    // Compute cofactor matrix (transposed)
    result(0,0) = (m(1,1) * m(2,2) - m(1,2) * m(2,1)) * inv_det;
    result(1,0) = -(m(1,0) * m(2,2) - m(1,2) * m(2,0)) * inv_det;
    result(2,0) = (m(1,0) * m(2,1) - m(1,1) * m(2,0)) * inv_det;
    
    result(0,1) = -(m(0,1) * m(2,2) - m(0,2) * m(2,1)) * inv_det;
    result(1,1) = (m(0,0) * m(2,2) - m(0,2) * m(2,0)) * inv_det;
    result(2,1) = -(m(0,0) * m(2,1) - m(0,1) * m(2,0)) * inv_det;
    
    result(0,2) = (m(0,1) * m(1,2) - m(0,2) * m(1,1)) * inv_det;
    result(1,2) = -(m(0,0) * m(1,2) - m(0,2) * m(1,0)) * inv_det;
    result(2,2) = (m(0,0) * m(1,1) - m(0,1) * m(1,0)) * inv_det;
    
    return result;
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

// Fast transpose for 4x4 matrices using SIMD
template<typename T>
auto fast_transpose_4x4(const matrix<T, 4, 4>& m) 
    -> std::enable_if_t<std::is_same_v<T, float>, matrix<T, 4, 4>> {
    matrix<T, 4, 4> result;
    
#ifdef EULER_HAS_SIMD
    __m128 row0 = _mm_load_ps(&m(0,0));
    __m128 row1 = _mm_load_ps(&m(1,0));
    __m128 row2 = _mm_load_ps(&m(2,0));
    __m128 row3 = _mm_load_ps(&m(3,0));
    
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    
    _mm_store_ps(&result(0,0), row0);
    _mm_store_ps(&result(1,0), row1);
    _mm_store_ps(&result(2,0), row2);
    _mm_store_ps(&result(3,0), row3);
#else
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result(i,j) = m(j,i);
        }
    }
#endif
    
    return result;
}

// Specialization selector based on matrix size
template<typename T, size_t M, size_t N, size_t P, bool RowMajor1, bool RowMajor2>
auto multiply_specialized(const matrix<T, M, N, RowMajor1>& a, const matrix<T, N, P, RowMajor2>& b) {
    if constexpr (M == 2 && N == 2 && P == 2 && std::is_same_v<T, float>) {
        return multiply_2x2_simd(a, b);
    } else if constexpr (M == 3 && N == 3 && P == 3 && std::is_same_v<T, float>) {
        return multiply_3x3_simd(a, b);
    } else if constexpr (M == 4 && N == 4 && P == 4 && std::is_same_v<T, float>) {
        return multiply_4x4_simd(a, b);
    } else {
        // This should never be reached due to if constexpr conditions
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
#ifdef EULER_HAS_SIMD
        if constexpr (std::is_same_v<T, float>) {
            // Process 4 matrices at a time using AVX
            for (size_t i = 0; i < count; ++i) {
                result_array[i] = multiply_4x4_simd(a_array[i], b_array[i]);
            }
        } else
#endif
        {
            // Fallback to standard multiplication
            for (size_t i = 0; i < count; ++i) {
                result_array[i] = a_array[i] * b_array[i];
            }
        }
    }
    
    // Batch transpose operations
    static void transpose_4x4_batch(const matrix<T, 4, 4>* input,
                                   matrix<T, 4, 4>* output,
                                   size_t count) {
#ifdef EULER_HAS_SIMD
        if constexpr (std::is_same_v<T, float>) {
            for (size_t i = 0; i < count; ++i) {
                output[i] = fast_transpose_4x4(input[i]);
            }
        } else
#endif
        {
            for (size_t i = 0; i < count; ++i) {
                output[i] = transpose(input[i]);
            }
        }
    }
};

} // namespace euler