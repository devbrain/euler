/**
 * @example 02_matrix_operations.cc
 * @brief Matrix operations and transformations
 * 
 * This example demonstrates:
 * - All matrix initialization methods (row-major, column-major, from_rows, from_cols)
 * - Basic matrix arithmetic
 * - Matrix multiplication
 * - Transpose, inverse, and determinant
 * - Matrix views and submatrices
 * - Special matrices (identity, rotation, etc.)
 */

#include <euler/euler.hh>
#include <euler/io/io.hh>
#include <iostream>
#include <iomanip>

using namespace euler;

// Helper function to print matrices nicely
template<typename MatrixExpr>
void print_matrix(const std::string& name, const MatrixExpr& m) {
    constexpr size_t R = MatrixExpr::rows;
    constexpr size_t C = MatrixExpr::cols;
    std::cout << name << " (" << R << "x" << C << "):\n";
    for (size_t i = 0; i < R; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < C; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << m(i, j);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Overload for matrix views (dynamic dimensions)
template<typename T>
void print_matrix(const std::string& name, const matrix_view<T>& m) {
    std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    std::cout << m << "\n\n";
}

template<typename T>
void print_matrix(const std::string& name, const const_matrix_view<T>& m) {
    std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    std::cout << m << "\n\n";
}

int main() {
    std::cout << "=== Euler Library: Matrix Operations Example ===\n\n";
    
    // 1. All matrix initialization methods
    std::cout << "1. Matrix initialization methods:\n\n";
    
    // a) Using nested initializer lists (row-by-row by default)
    std::cout << "a) Nested initializer lists:\n";
    matrix<float, 3, 3> m1({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });
    print_matrix("m1 (nested init)", m1);
    
    // b) Using from_rows (explicit row-major)
    std::cout << "b) from_rows (row-major order):\n";
    auto m2 = matrix<float, 3, 3>::from_rows({
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });
    print_matrix("m2 (from_rows)", m2);
    std::cout << "m1 == m2: " << (m1 == m2 ? "true" : "false") << "\n\n";
    
    // c) Using from_cols (column-major order)
    std::cout << "c) from_cols (column-major order):\n";
    auto m3 = matrix<float, 3, 3>::from_cols({
        {1, 4, 7},  // First column
        {2, 5, 8},  // Second column
        {3, 6, 9}   // Third column
    });
    print_matrix("m3 (from_cols)", m3);
    std::cout << "m1 == m3: " << (m1 == m3 ? "true" : "false") << " (same data, different input order)\n\n";
    
    // d) Using from_row_major (flat array, row-major)
    std::cout << "d) from_row_major (flat array in row-major order):\n";
    auto m4 = matrix<float, 3, 3>::from_row_major({
        1, 2, 3,    // Row 0
        4, 5, 6,    // Row 1
        7, 8, 9     // Row 2
    });
    print_matrix("m4 (from_row_major)", m4);
    
    // e) Using from_col_major (flat array, column-major)
    std::cout << "e) from_col_major (flat array in column-major order):\n";
    auto m5 = matrix<float, 3, 3>::from_col_major({
        1, 4, 7,    // Column 0
        2, 5, 8,    // Column 1  
        3, 6, 9     // Column 2
    });
    print_matrix("m5 (from_col_major)", m5);
    std::cout << "m4 == m5: " << (m4 == m5 ? "true" : "false") << " (same matrix, different input order)\n\n";
    
    // f) Special matrices
    std::cout << "f) Special matrix constructors:\n";
    auto identity = matrix<float, 3, 3>::identity();
    print_matrix("identity", identity);
    
    auto zeros = matrix<float, 2, 3>::zero();
    print_matrix("zeros", zeros);
    
    matrix<float, 2, 3> filled(2.5f);  // All elements = 2.5
    print_matrix("filled with 2.5", filled);
    
    // g) Different storage layouts
    std::cout << "g) Different storage layouts (row-major vs column-major):\n";
    matrix<float, 3, 3, false> row_major = matrix<float, 3, 3, false>::from_row_major({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    matrix<float, 3, 3, true> col_major = matrix<float, 3, 3, true>::from_row_major({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    std::cout << "Row-major storage: " << row_major(0,0) << ", " << row_major(0,1) << ", " << row_major(0,2) << "...\n";
    std::cout << "Col-major storage: " << col_major(0,0) << ", " << col_major(0,1) << ", " << col_major(0,2) << "...\n";
    std::cout << "Both represent the same matrix: " << (row_major == col_major ? "true" : "false") << "\n\n";
    
    // 2. Element access
    std::cout << "2. Element access:\n";
    std::cout << "m1(0,0) = " << m1(0, 0) << "\n";
    std::cout << "m1(1,2) = " << m1(1, 2) << "\n";
    
    // Modify elements
    m1(0, 0) = 10;
    std::cout << "After m1(0,0) = 10:\n";
    print_matrix("m1", m1);
    
    // 3. Basic arithmetic
    std::cout << "3. Basic arithmetic:\n";
    auto m_diag = matrix<float, 3, 3>::from_rows({
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}
    });
    
    auto sum = m1 + m_diag;
    auto diff = m1 - m_diag;
    auto scaled = 2.0f * m_diag;
    
    print_matrix("diagonal matrix", m_diag);
    print_matrix("m1 + diagonal", sum);
    print_matrix("m1 - diagonal", diff);
    print_matrix("2 * diagonal", scaled);
    
    // 4. Matrix multiplication
    std::cout << "4. Matrix multiplication:\n";
    auto a = matrix<float, 2, 3>::from_rows({
        {1, 2, 3},
        {4, 5, 6}
    });
    
    auto b = matrix<float, 3, 2>::from_rows({
        {7, 8},
        {9, 10},
        {11, 12}
    });
    
    auto mat_c = a * b;  // Results in 2x2 matrix
    print_matrix("a", a);
    print_matrix("b", b);
    print_matrix("a * b", mat_c);
    
    // 5. Transpose
    std::cout << "5. Transpose:\n";
    auto a_transpose = transpose(a);
    print_matrix("transpose(a)", a_transpose);
    
    // 6. Square matrix operations
    std::cout << "6. Square matrix operations:\n";
    auto m_square = matrix<float, 3, 3>::from_rows({
        {2, 1, 0},
        {1, 3, 1},
        {0, 1, 2}
    });
    
    float det = determinant(m_square);
    auto m_square_inv = inverse(m_square);
    float tr = trace(m_square);
    
    print_matrix("m_square", m_square);
    std::cout << "determinant(m_square) = " << det << "\n";
    std::cout << "trace(m_square) = " << tr << "\n";
    print_matrix("inverse(m_square)", m_square_inv);
    
    // Verify inverse
    auto should_be_identity = m_square * m_square_inv;
    print_matrix("m_square * inverse(m_square)", should_be_identity);
    
    // 7. Matrix-vector multiplication
    std::cout << "7. Matrix-vector multiplication:\n";
    vector<float, 3> v(1, 2, 3);
    auto result = m_square * v;
    
    std::cout << "v = " << v << "\n";
    std::cout << "m_square * v = " << result << "\n\n";
    
    // 8. 2D rotation matrix
    std::cout << "8. 2D rotation matrix:\n";
    auto angle = degree<float>(45);
    // Create rotation matrix manually
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    matrix<float, 2, 2> rot2d({
        {cos_angle, -sin_angle},
        {sin_angle,  cos_angle}
    });
    print_matrix("45° rotation", rot2d);
    
    vector<float, 2> v2d(1, 0);
    auto v2d_rotated = rot2d * v2d;
    std::cout << "Rotating (1,0) by 45°: " << v2d_rotated << "\n\n";
    
    // 9. 3D transformation matrices
    std::cout << "9. 3D transformation matrices:\n";
    
    // Rotation around Z axis
    float angle_z = radian<float>(degree<float>(30));
    float cz = cos(angle_z);
    float sz = sin(angle_z);
    matrix<float, 3, 3> rot_z = {
        { cz, -sz, 0},
        { sz,  cz, 0},
        { 0,   0,  1}
    };
    print_matrix("30° rotation around Z", rot_z);
    
    // Scale matrix
    matrix<float, 3, 3> scale = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 1}
    };
    print_matrix("Scale (2, 3, 1)", scale);
    
    // 10. Matrix views (submatrices)
    std::cout << "10. Matrix views:\n";
    matrix<float, 4, 4> big = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    
    print_matrix("Original 4x4", big);
    
    // Get a 2x2 submatrix starting at (1,1)
    auto sub_view = big.submatrix(1, 1, 2, 2);
    std::cout << "2x2 submatrix at (1,1) view: " << sub_view << "\n\n";
    
    // Get row and column views
    auto row_view = big.row(2);
    auto col_view = big.col(3);
    
    std::cout << "Row 2 view: " << row_view << "\n";
    std::cout << "Column 3 view: " << col_view << "\n\n";
    
    // Modifying through views
    std::cout << "Modifying through views:\n";
    sub_view(0, 0) = 100;  // This modifies the original matrix!
    std::cout << "After sub_view(0,0) = 100:\n";
    print_matrix("Modified big matrix", big);
    std::cout << "Notice element (1,1) changed to 100\n\n";
    
    // 11. Expression templates efficiency
    std::cout << "11. Expression templates (efficient evaluation):\n";
    auto m_expr1 = matrix<float, 3, 3>::from_rows({
        {1, 2, 0},
        {2, 1, 0},
        {0, 0, 1}
    });
    
    auto m_expr2 = matrix<float, 3, 3>::from_rows({
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1}
    });
    
    // This complex expression is evaluated in a single pass
    auto result_expr = 2.0f * m_expr1 + 3.0f * m_expr2 - transpose(m_expr1);
    print_matrix("2*m_expr1 + 3*m_expr2 - transpose(m_expr1)", result_expr);
    
    // 12. Practical implications of storage order
    std::cout << "12. Storage order implications:\n";
    
    // Create matrices with different storage orders
    matrix<float, 3, 3, false> rm = matrix<float, 3, 3, false>::from_row_major({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    
    matrix<float, 3, 3, true> cm = matrix<float, 3, 3, true>::from_row_major({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    
    std::cout << "Row-major vs Column-major storage:\n";
    std::cout << "  Both matrices contain the same logical data\n";
    std::cout << "  rm == cm: " << (rm == cm ? "true" : "false") << "\n\n";
    
    // Show how to efficiently iterate based on storage order
    std::cout << "Efficient iteration patterns:\n";
    std::cout << "  Row-major: iterate rows then columns (cache-friendly)\n";
    std::cout << "  Column-major: iterate columns then rows (cache-friendly)\n\n";
    
    // Demonstrate initialization from external data
    std::cout << "Interfacing with external libraries:\n";
    float opengl_data[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};  // Column-major (OpenGL style)
    float directx_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Row-major (DirectX style)
    
    auto opengl_matrix = matrix<float, 3, 3, true>::from_col_major({
        opengl_data[0], opengl_data[1], opengl_data[2],
        opengl_data[3], opengl_data[4], opengl_data[5],
        opengl_data[6], opengl_data[7], opengl_data[8]
    });
    
    auto directx_matrix = matrix<float, 3, 3, false>::from_row_major({
        directx_data[0], directx_data[1], directx_data[2],
        directx_data[3], directx_data[4], directx_data[5],
        directx_data[6], directx_data[7], directx_data[8]
    });
    
    std::cout << "  OpenGL matrix (column-major): " << opengl_matrix(0,1) << " at (0,1)\n";
    std::cout << "  DirectX matrix (row-major): " << directx_matrix(0,1) << " at (0,1)\n";
    std::cout << "  Both show: " << (opengl_matrix == directx_matrix ? "same" : "different") << " logical matrix\n";
    
    return 0;
}