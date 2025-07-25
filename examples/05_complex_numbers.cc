/**
 * @example 05_complex_numbers.cc
 * @brief Complex number operations
 * 
 * This example demonstrates:
 * - Creating complex numbers
 * - Complex arithmetic
 * - Polar and Cartesian forms
 * - Complex functions (exp, log, pow, trig)
 * - Using complex numbers with angles
 */

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace euler;

// Helper to print complex numbers nicely
template<typename T>
void print_complex(const std::string& name, const complex<T>& c) {
    std::cout << name << " = " << c.real() << " + " << c.imag() << "i";
    std::cout << " = " << abs(c) << " ∠ " << degree<T>(arg(c)) << "\n";
}

int main() {
    std::cout << "=== Euler Library: Complex Numbers Example ===\n\n";
    
    // 1. Creating complex numbers
    std::cout << "1. Creating complex numbers:\n";
    
    complex<float> c1(3, 4);        // 3 + 4i
    complex<float> c2(5, -2);       // 5 - 2i
    complex<float> c3(2.0f);        // 2 + 0i (real number)
    complex<float> c4(0, 1);        // 0 + 1i (imaginary unit)
    
    print_complex("c1", c1);
    print_complex("c2", c2);
    print_complex("c3", c3);
    print_complex("i", c4);
    std::cout << "\n";
    
    // 2. Basic arithmetic
    std::cout << "2. Basic arithmetic:\n";
    
    auto sum = c1 + c2;
    auto diff = c1 - c2;
    auto prod = c1 * c2;
    auto quot = c1 / c2;
    
    print_complex("c1 + c2", sum);
    print_complex("c1 - c2", diff);
    print_complex("c1 * c2", prod);
    print_complex("c1 / c2", quot);
    std::cout << "\n";
    
    // 3. Complex properties
    std::cout << "3. Complex properties:\n";
    
    std::cout << "c1 = " << c1 << "\n";
    std::cout << "  Real part: " << c1.real() << "\n";
    std::cout << "  Imaginary part: " << c1.imag() << "\n";
    std::cout << "  Magnitude: " << abs(c1) << "\n";
    std::cout << "  Phase angle: " << arg(c1) << " rad = " << degree<float>(arg(c1)) << "\n";
    std::cout << "  Conjugate: " << conj(c1) << "\n";
    std::cout << "  Norm squared: " << norm(c1) << "\n\n";
    
    // 4. Polar form
    std::cout << "4. Polar form:\n";
    
    // Create from polar coordinates
    auto magnitude = 2.0f;
    auto phase = degree<float>(45);
    auto c_polar = complex<float>::polar(magnitude, phase);
    
    print_complex("2 ∠ 45°", c_polar);
    std::cout << "  Cartesian: (" << c_polar.real() << ", " << c_polar.imag() << ")\n\n";
    
    // 5. Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
    std::cout << "5. Euler's formula demonstration:\n";
    
    auto theta = degree<float>(90);
    auto euler_result = exp(complex<float>(0, 1) * theta.value());
    
    std::cout << "e^(i*90°) = " << euler_result << "\n";
    std::cout << "cos(90°) + i*sin(90°) = " << cos(theta) << " + " << sin(theta) << "i\n";
    std::cout << "These should be equal!\n\n";
    
    // 6. Complex exponential and logarithm
    std::cout << "6. Complex exponential and logarithm:\n";
    
    complex<float> z(1, 1);
    auto exp_z = exp(z);
    auto log_z = log(z);
    
    print_complex("z", z);
    print_complex("exp(z)", exp_z);
    print_complex("log(z)", log_z);
    
    // Verify: exp(log(z)) = z
    auto back = exp(log_z);
    print_complex("exp(log(z))", back);
    std::cout << "\n";
    
    // 7. Complex powers
    std::cout << "7. Complex powers:\n";
    
    auto base = complex<float>(2, 1);
    auto squared = pow(base, 2.0f);
    auto sqrt_val = sqrt(base);
    auto cubed = pow(base, complex<float>(3, 0));
    
    print_complex("base", base);
    print_complex("base^2", squared);
    print_complex("√base", sqrt_val);
    print_complex("base^3", cubed);
    
    // Verify: (√base)^2 = base
    auto sqrt_squared = pow(sqrt_val, 2.0f);
    print_complex("(√base)^2", sqrt_squared);
    std::cout << "\n";
    
    // 8. Complex trigonometry
    std::cout << "8. Complex trigonometry:\n";
    
    complex<float> z_trig(0.5f, 0.5f);
    
    auto sin_z = sin(z_trig);
    auto cos_z = cos(z_trig);
    auto tan_z = tan(z_trig);
    
    print_complex("z", z_trig);
    print_complex("sin(z)", sin_z);
    print_complex("cos(z)", cos_z);
    print_complex("tan(z)", tan_z);
    
    // Verify: sin²(z) + cos²(z) = 1
    auto identity = pow(sin_z, 2.0f) + pow(cos_z, 2.0f);
    print_complex("sin²(z) + cos²(z)", identity);
    std::cout << "\n";
    
    // 9. Roots of unity
    std::cout << "9. nth roots of unity:\n";
    
    const int n = 6;
    std::cout << "6th roots of unity:\n";
    
    for (int k = 0; k < n; ++k) {
        auto angle = radian<float>(2.0f * constants<float>::pi * k / n);
        auto root = complex<float>::polar(1.0f, angle);
        
        std::cout << "  k=" << k << ": ";
        print_complex("", root);
        
        // Verify it's a 6th root
        auto powered = pow(root, static_cast<float>(n));
        std::cout << "    Verification: root^6 = " << powered << "\n";
    }
    std::cout << "\n";
    
    // 10. Practical example: FFT butterfly operation
    std::cout << "10. FFT butterfly operation example:\n";
    
    // Two input values
    complex<float> a(1, 0);
    complex<float> b(0, 1);
    
    // Twiddle factor for N=8, k=1
    auto W = complex<float>::polar(1.0f, radian<float>(-2.0f * constants<float>::pi / 8));
    
    // Butterfly operation
    auto top = a + W * b;
    auto bottom = a - W * b;
    
    std::cout << "Input:\n";
    print_complex("  a", a);
    print_complex("  b", b);
    print_complex("  W", W);
    
    std::cout << "Output:\n";
    print_complex("  top = a + W*b", top);
    print_complex("  bottom = a - W*b", bottom);
    
    return 0;
}