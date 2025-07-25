/**
 * @example 01_vector_basics.cc
 * @brief Basic vector operations demonstration
 * 
 * This example shows how to:
 * - Create vectors
 * - Perform basic arithmetic operations
 * - Calculate dot and cross products
 * - Normalize vectors
 * - Use vector expressions
 */

#include <euler/euler.hh>
#include <euler/io/io.hh>
#include <iostream>
#include <iomanip>

using namespace euler;

int main() {
    std::cout << "=== Euler Library: Vector Operations Example ===\n\n";
    
    // 1. Creating vectors
    std::cout << "1. Creating vectors:\n";
    vector<float, 3> v1(1.0f, 2.0f, 3.0f);
    vector<float, 3> v2{4.0f, 5.0f, 6.0f};  // Alternative syntax
    vector<float, 3> v3(7.0f);              // Fill constructor (all components = 7.0)
    
    std::cout << "v1 = " << v1 << "\n";
    std::cout << "v2 = " << v2 << "\n";
    std::cout << "v3 = " << v3 << "\n\n";
    
    // 2. Basic arithmetic
    std::cout << "2. Basic arithmetic:\n";
    auto sum = v1 + v2;
    auto diff = v2 - v1;
    auto scaled = 2.0f * v1;
    
    std::cout << "v1 + v2 = " << sum << "\n";
    std::cout << "v2 - v1 = " << diff << "\n";
    std::cout << "2 * v1 = " << scaled << "\n\n";
    
    // 3. Dot product
    std::cout << "3. Dot product:\n";
    float dot_prod = dot(v1, v2);
    std::cout << "dot(v1, v2) = " << dot_prod << "\n";
    std::cout << "This represents: " << v1[0] << "*" << v2[0] 
              << " + " << v1[1] << "*" << v2[1] 
              << " + " << v1[2] << "*" << v2[2] << " = " << dot_prod << "\n\n";
    
    // 4. Cross product (3D only)
    std::cout << "4. Cross product:\n";
    auto cross_prod = cross(v1, v2);
    std::cout << "cross(v1, v2) = " << cross_prod << "\n";
    
    // Verify cross product is perpendicular to both vectors
    std::cout << "Verification (should be ~0):\n";
    std::cout << "  dot(cross_prod, v1) = " << dot(cross_prod, v1) << "\n";
    std::cout << "  dot(cross_prod, v2) = " << dot(cross_prod, v2) << "\n\n";
    
    // 5. Vector length and normalization
    std::cout << "5. Length and normalization:\n";
    float len = v1.length();
    std::cout << "length(v1) = " << len << "\n";
    std::cout << "length_squared(v1) = " << v1.length_squared() << "\n";
    
    auto v1_normalized = normalize(v1);
    std::cout << "normalize(v1) = " << v1_normalized << "\n";
    vector<float, 3> v1_norm_vec = v1_normalized;  // Convert expression to vector
    std::cout << "length(normalize(v1)) = " << v1_norm_vec.length() << "\n\n";
    
    // 6. Component access
    std::cout << "6. Component access:\n";
    std::cout << "v1[0] = " << v1[0] << ", v1[1] = " << v1[1] << ", v1[2] = " << v1[2] << "\n";
    
    // Modify components
    v1[1] = 10.0f;
    std::cout << "After v1[1] = 10: " << v1 << "\n\n";
    
    // 7. Expression templates in action
    std::cout << "7. Expression templates (efficient compound operations):\n";
    vector<float, 3> a(1, 0, 0);
    vector<float, 3> b(0, 1, 0);
    vector<float, 3> c(0, 0, 1);
    
    // This creates a single loop, not multiple temporary vectors
    auto result = 2.0f * a + 3.0f * b - c;
    std::cout << "2*a + 3*b - c = " << result << "\n\n";
    
    // 8. Working with different dimensions
    std::cout << "8. Vectors of different dimensions:\n";
    vector<double, 2> v2d(3.0, 4.0);
    std::cout << "2D vector: " << v2d << ", length = " << v2d.length() << "\n";
    
    vector<float, 4> v4d(1, 2, 3, 4);
    std::cout << "4D vector: " << v4d << ", length = " << v4d.length() << "\n\n";
    
    // 9. Special vectors
    std::cout << "9. Special vectors:\n";
    auto unit_x = vector<float, 3>::unit_x();
    auto unit_y = vector<float, 3>::unit_y();
    auto unit_z = vector<float, 3>::unit_z();
    auto zero = vector<float, 3>::zero();
    
    std::cout << "unit_x = " << unit_x << "\n";
    std::cout << "unit_y = " << unit_y << "\n";
    std::cout << "unit_z = " << unit_z << "\n";
    std::cout << "zero = " << zero << "\n\n";
    
    // 10. Angle between vectors
    std::cout << "10. Angle between vectors:\n";
    vector<float, 3> va(1, 0, 0);
    vector<float, 3> vb(1, 1, 0);
    
    float cos_angle = dot(va, vb) / (va.length() * vb.length());
    auto angle_rad = std::acos(cos_angle);
    auto angle_deg = angle_rad * 180.0f / constants<float>::pi;
    
    std::cout << "Angle between " << va << " and " << vb << ":\n";
    std::cout << "  " << angle_rad << " radians\n";
    std::cout << "  " << angle_deg << " degrees\n";
    
    return 0;
}