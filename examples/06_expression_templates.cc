/**
 * @example 06_expression_templates.cc
 * @brief Expression templates and performance
 * 
 * This example demonstrates:
 * - How expression templates eliminate temporaries
 * - Performance benefits of lazy evaluation
 * - Complex expressions evaluated in single pass
 * - SIMD optimization with expression templates
 */

#include <euler/euler.hh>
#include <iostream>
#include <chrono>
#include <vector>

using namespace euler;
using namespace std::chrono;

// Timer helper
class Timer {
    high_resolution_clock::time_point start;
public:
    Timer() : start(high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count() / 1000.0;
    }
};

int main() {
    std::cout << "=== Euler Library: Expression Templates Example ===\n\n";
    
    // 1. Basic concept
    std::cout << "1. Expression templates eliminate temporaries:\n";
    
    vector<float, 3> a(1, 2, 3);
    vector<float, 3> b(4, 5, 6);
    vector<float, 3> c(7, 8, 9);
    
    // This expression creates NO temporary vectors!
    // It's evaluated element-wise in a single pass
    auto result = 2.0f * a + 3.0f * b - c;
    
    std::cout << "Expression: 2*a + 3*b - c\n";
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "c = " << c << "\n";
    std::cout << "Result = " << result << "\n\n";
    
    // 2. Complex vector expressions
    std::cout << "2. Complex vector expressions:\n";
    
    vector<float, 4> v1(1, 2, 3, 4);
    vector<float, 4> v2(5, 6, 7, 8);
    vector<float, 4> v3(9, 10, 11, 12);
    vector<float, 4> v4(13, 14, 15, 16);
    
    // This would normally create many temporaries
    // But expression templates evaluate it in one pass
    auto complex_expr = (v1 + v2) * 2.0f - (v3 - v4) / 3.0f + v1 * v2;
    
    std::cout << "Complex expression: (v1+v2)*2 - (v3-v4)/3 + v1*v2\n";
    std::cout << "Result = " << complex_expr << "\n\n";
    
    // 3. Matrix expression templates
    std::cout << "3. Matrix expression templates:\n";
    
    matrix<float, 3, 3> m1 = matrix<float, 3, 3>::from_row_major({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    
    matrix<float, 3, 3> m2 = matrix<float, 3, 3>::identity();
    matrix<float, 3, 3> m3 = matrix<float, 3, 3>::from_row_major({
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    });
    
    // Complex matrix expression - no temporaries!
    auto mat_expr = 2.0f * m1 + 3.0f * m2 - transpose(m3);
    
    std::cout << "Matrix expression: 2*m1 + 3*I - transpose(m3)\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(6) << mat_expr(i, j);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // 4. Performance comparison
    std::cout << "4. Performance comparison:\n";
    
    const size_t N = 1000000;
    std::vector<vector<float, 3>> data1(N), data2(N), data3(N), results(N);
    
    // Initialize with random data
    for (size_t i = 0; i < N; ++i) {
        data1[i] = vector<float, 3>(i * 0.1f, i * 0.2f, i * 0.3f);
        data2[i] = vector<float, 3>(i * 0.4f, i * 0.5f, i * 0.6f);
        data3[i] = vector<float, 3>(i * 0.7f, i * 0.8f, i * 0.9f);
    }
    
    // Expression template version
    {
        Timer timer;
        for (size_t i = 0; i < N; ++i) {
            results[i] = 2.0f * data1[i] + 3.0f * data2[i] - data3[i];
        }
        std::cout << "Expression templates: " << timer.elapsed_ms() << " ms\n";
    }
    
    // Manual component-wise (what expression templates do internally)
    {
        Timer timer;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                results[i][j] = 2.0f * data1[i][j] + 3.0f * data2[i][j] - data3[i][j];
            }
        }
        std::cout << "Manual component-wise: " << timer.elapsed_ms() << " ms\n";
    }
    
    std::cout << "(Expression templates should be similar to manual)\n\n";
    
    // 5. Lazy evaluation
    std::cout << "5. Lazy evaluation demonstration:\n";
    
    vector<float, 3> x(1, 2, 3);
    vector<float, 3> y(4, 5, 6);
    
    // This doesn't compute anything yet!
    auto lazy_expr = x + y;
    
    std::cout << "Created expression: x + y\n";
    std::cout << "Type of expression: " << typeid(lazy_expr).name() << "\n";
    std::cout << "(This is an expression type, not a vector!)\n\n";
    
    // Computation happens when we access elements
    std::cout << "Accessing elements triggers computation:\n";
    std::cout << "lazy_expr[0] = " << lazy_expr[0] << "\n";
    std::cout << "lazy_expr[1] = " << lazy_expr[1] << "\n";
    std::cout << "lazy_expr[2] = " << lazy_expr[2] << "\n\n";
    
    // Or when we assign to a vector
    vector<float, 3> computed = lazy_expr;  // Now it computes
    std::cout << "Assigned to vector: " << computed << "\n\n";
    
    // 6. Chaining operations
    std::cout << "6. Chaining operations efficiently:\n";
    
    vector<double, 4> p1(1, 0, 0, 0);
    vector<double, 4> p2(0, 1, 0, 0);
    vector<double, 4> p3(0, 0, 1, 0);
    vector<double, 4> p4(0, 0, 0, 1);
    
    // This complex chain is still evaluated in one pass
    auto chain = normalize(p1 + p2) + normalize(p3 + p4);
    
    std::cout << "normalize(p1 + p2) + normalize(p3 + p4) = " << chain << "\n";
    std::cout << "Length should be ~2: " << length(chain) << "\n\n";
    
    // 7. Expression templates with functions
    std::cout << "7. Expression templates with mathematical functions:\n";
    
    vector<float, 3> angles(0, constants<float>::pi/4, constants<float>::pi/2);
    
    // Even complex expressions with functions work
    auto trig_expr = 2.0f * sin(angles) + cos(angles);
    
    std::cout << "angles = " << angles << "\n";
    std::cout << "2*sin(angles) + cos(angles) = " << trig_expr << "\n\n";
    
    // 8. Matrix multiplication chains
    std::cout << "8. Efficient matrix multiplication chains:\n";
    
    matrix<float, 2, 3> A = matrix<float, 2, 3>::from_row_major({
        1, 2, 3,
        4, 5, 6
    });
    
    matrix<float, 3, 4> B = matrix<float, 3, 4>::from_row_major({
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18
    });
    
    matrix<float, 4, 2> C = matrix<float, 4, 2>::from_row_major({
        19, 20,
        21, 22,
        23, 24,
        25, 26
    });
    
    // Matrix multiplication is optimally ordered
    auto chain_result = A * B * C;  // (2x3) * (3x4) * (4x2) = (2x2)
    
    std::cout << "Result of A * B * C:\n";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 2; ++j) {
            std::cout << std::setw(8) << chain_result(i, j);
        }
        std::cout << "\n";
    }
    
    std::cout << "\nExpression templates ensure optimal evaluation order!\n";
    
    return 0;
}