/**
 * @example 04_angle_types.cc
 * @brief Type-safe angle handling
 * 
 * This example demonstrates:
 * - Creating angles in degrees and radians
 * - Automatic conversion between units
 * - Type safety preventing unit errors
 * - Angle arithmetic and comparisons
 * - Using angles with trigonometric functions
 */

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace euler;

int main() {
    std::cout << "=== Euler Library: Type-Safe Angles Example ===\n\n";
    
    // 1. Creating angles
    std::cout << "1. Creating angles:\n";
    
    // Using degree and radian types
    auto angle1 = degree<float>(45.0f);
    auto angle2 = radian<float>(constants<float>::pi / 4);
    
    std::cout << "angle1 = " << angle1 << "\n";
    std::cout << "angle2 = " << angle2 << "\n";
    
    // These represent the same angle!
    std::cout << "Are they equal? " << (angle1 == angle2 ? "YES" : "NO") << "\n\n";
    
    // 2. Automatic conversion
    std::cout << "2. Automatic conversion:\n";
    
    // Convert degree to radian
    radian<float> angle1_rad = angle1;  // Automatic conversion
    std::cout << "45° in radians: " << angle1_rad << "\n";
    
    // Convert radian to degree
    degree<float> angle2_deg = angle2;  // Automatic conversion
    std::cout << "π/4 rad in degrees: " << angle2_deg << "\n\n";
    
    // 3. Angle arithmetic
    std::cout << "3. Angle arithmetic:\n";
    
    auto deg30 = degree<float>(30);
    auto deg60 = degree<float>(60);
    auto rad_pi6 = radian<float>(constants<float>::pi / 6);
    
    // Addition (mixed units work!)
    auto sum = deg30 + deg60;
    std::cout << "30° + 60° = " << sum << "\n";
    
    auto mixed_sum = deg30 + rad_pi6;  // π/6 rad = 30°
    std::cout << "30° + π/6 rad = " << degree<float>(mixed_sum) << "\n";
    
    // Subtraction
    auto diff = deg60 - deg30;
    std::cout << "60° - 30° = " << diff << "\n";
    
    // Multiplication by scalar
    auto doubled = 2.0f * deg30;
    std::cout << "2 × 30° = " << doubled << "\n";
    
    // Division by scalar
    auto halved = deg60 / 2.0f;
    std::cout << "60° / 2 = " << halved << "\n\n";
    
    // 4. Angle comparisons
    std::cout << "4. Angle comparisons:\n";
    
    auto a1 = degree<float>(45);
    auto a2 = degree<float>(90);
    auto a3 = radian<float>(constants<float>::pi / 4);  // = 45°
    
    std::cout << "45° < 90°? " << (a1 < a2 ? "YES" : "NO") << "\n";
    std::cout << "45° == π/4 rad? " << (a1 == a3 ? "YES" : "NO") << "\n";
    std::cout << "90° > π/4 rad? " << (a2 > a3 ? "YES" : "NO") << "\n\n";
    
    // 5. Trigonometric functions
    std::cout << "5. Trigonometric functions:\n";
    
    // Works with both degree and radian
    auto deg45 = degree<float>(45);
    auto rad45 = radian<float>(constants<float>::pi / 4);
    
    std::cout << "sin(45°) = " << sin(deg45) << "\n";
    std::cout << "cos(45°) = " << cos(deg45) << "\n";
    std::cout << "tan(45°) = " << tan(deg45) << "\n";
    
    std::cout << "sin(π/4) = " << sin(rad45) << "\n";
    std::cout << "cos(π/4) = " << cos(rad45) << "\n\n";
    
    // 6. Inverse trigonometric functions
    std::cout << "6. Inverse trigonometric functions:\n";
    
    float value = 0.5f;
    auto asin_rad = asin(value);  // Returns radian by default
    auto acos_rad = acos(value);
    
    std::cout << "asin(0.5) = " << asin_rad << " = " << degree<float>(asin_rad) << "\n";
    std::cout << "acos(0.5) = " << acos_rad << " = " << degree<float>(acos_rad) << "\n\n";
    
    // 7. Angle wrapping/normalization
    std::cout << "7. Angle wrapping:\n";
    
    auto big_angle = degree<float>(450);  // More than 360°
    auto negative_angle = degree<float>(-30);
    
    std::cout << "450° = " << big_angle << "\n";
    std::cout << "-30° = " << negative_angle << "\n";
    
    // Normalize to [0, 360) or [-180, 180]
    auto normalized = degree<float>(fmod(big_angle.value(), 360.0f));
    std::cout << "450° normalized to [0, 360): " << normalized << "\n\n";
    
    // 8. Using angles in practical calculations
    std::cout << "8. Practical example - circular motion:\n";
    
    const int num_points = 8;
    const float radius = 1.0f;
    
    std::cout << "Points on a unit circle:\n";
    for (int i = 0; i < num_points; ++i) {
        auto angle = degree<float>(360.0f * i / num_points);
        float x = radius * cos(angle);
        float y = radius * sin(angle);
        
        std::cout << "  " << std::setw(4) << angle << ": "
                  << "(" << std::fixed << std::setprecision(3) 
                  << x << ", " << y << ")\n";
    }
    std::cout << "\n";
    
    // 9. Angle literals (if using C++14 or later)
    std::cout << "9. Convenient angle creation:\n";
    
    // Using explicit constructors
    auto a90 = degree<float>(90.0f);
    auto api = radian<float>(constants<float>::pi);
    
    std::cout << "degree<float>(90) = " << a90 << "\n";
    std::cout << "radian<float>(π) = " << api << " = " << degree<float>(api) << "\n\n";
    
    // 10. Type safety demonstration
    std::cout << "10. Type safety prevents errors:\n";
    
    // This would cause a compile error if uncommented:
    // float raw_angle = 45.0f;
    // float result = sin(raw_angle);  // Error! sin expects angle<T, Unit>, not float
    
    // Correct usage:
    float result = sin(degree<float>(45.0f));
    std::cout << "sin(degree<float>(45)) = " << result << "\n";
    
    // This prevents common bugs like:
    // - Passing degrees to a function expecting radians
    // - Forgetting to convert units
    // - Mixing up angle representations
    
    std::cout << "\nType safety ensures you never accidentally mix degrees and radians!\n";
    
    return 0;
}