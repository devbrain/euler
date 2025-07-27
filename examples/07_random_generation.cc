/**
 * @example 07_random_generation.cc
 * @brief Random generation for geometric objects
 * 
 * This example demonstrates:
 * - Random number generation basics
 * - Random vectors (unit, in sphere, on sphere)
 * - Random rotations and quaternions
 * - Random angles with distributions
 * - Random matrices
 * - Monte Carlo simulations
 */

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

using namespace euler;

int main() {
    std::cout << "=== Euler Library: Random Generation Example ===\n\n";
    
    // 1. Basic random number generation
    std::cout << "1. Basic random number generation:\n";
    
    random_generator rng(42);  // Seed for reproducibility
    
    // Uniform distribution
    std::cout << "Uniform float [0,1): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << rng.uniform<float>() << " ";
    }
    std::cout << "\n";
    
    // Uniform integers
    std::cout << "Uniform int [1,6]: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << rng.uniform(1, 6) << " ";
    }
    std::cout << " (dice rolls)\n";
    
    // Normal distribution
    std::cout << "Normal(0,1): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(3) 
                  << rng.normal<float>(0.0f, 1.0f) << " ";
    }
    std::cout << "\n\n";
    
    // 2. Random vectors
    std::cout << "2. Random vectors:\n";
    
    // Random vector in box
    for (int i = 0; i < 3; ++i) {
        auto v = random_vector<float, 3>(rng, -1.0f, 1.0f);
        std::cout << "Random in [-1,1]³: " << v << "\n";
    }
    std::cout << "\n";
    
    // Random unit vectors (on sphere)
    std::cout << "Random unit vectors:\n";
    for (int i = 0; i < 3; ++i) {
        auto v = random_unit_vector<float, 3>(rng);
        std::cout << "  " << v << " (length = " << v.length() << ")\n";
    }
    std::cout << "\n";
    
    // Random vectors in sphere
    std::cout << "Random vectors in unit sphere:\n";
    for (int i = 0; i < 3; ++i) {
        auto v = random_in_sphere<float, 3>(rng);
        std::cout << "  " << v << " (length = " << v.length() << ")\n";
    }
    std::cout << "\n";
    
    // 3. Random rotations
    std::cout << "3. Random rotations:\n";
    
    // Random quaternions (uniform on SO(3))
    for (int i = 0; i < 3; ++i) {
        auto q = random_quaternion<float>(rng);
        auto angle = degree<float>(q.angle());
        auto axis = q.axis();
        
        std::cout << "Random rotation: " << angle 
                  << " around (" << axis[0] << ", " << axis[1] << ", " << axis[2] << ")\n";
    }
    std::cout << "\n";
    
    // Random small rotations
    std::cout << "Small rotations (max 30°):\n";
    auto max_angle = degree<float>(30);
    for (int i = 0; i < 3; ++i) {
        auto q = random_quaternion<float>(rng, max_angle);
        std::cout << "  Angle: " << degree<float>(q.angle()) << "\n";
    }
    std::cout << "\n";
    
    // 4. Random angles
    std::cout << "4. Random angles:\n";
    
    // Uniform angles
    std::cout << "Uniform angles [0°, 360°):\n";
    for (int i = 0; i < 5; ++i) {
        auto angle = random_angle<float, degree_tag>(rng, 
            degree<float>(0), degree<float>(360));
        std::cout << "  " << angle << "\n";
    }
    std::cout << "\n";
    
    // Von Mises distribution (concentrated around mean)
    std::cout << "Von Mises angles (mean=90°, κ=5):\n";
    auto mean_angle = degree<float>(90);
    float kappa = 5.0f;  // Concentration parameter
    for (int i = 0; i < 5; ++i) {
        auto angle = random_angle_von_mises(rng, mean_angle, kappa);
        std::cout << "  " << angle << "\n";
    }
    std::cout << "\n";
    
    // 5. Random matrices
    std::cout << "5. Random matrices:\n";
    
    // Random matrix with elements in range
    auto m = random_matrix<float, 2, 3>(rng, -1.0f, 1.0f);
    std::cout << "Random 2x3 matrix [-1,1]:\n";
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(3) << m(i, j);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Random rotation matrix
    auto rot_mat = random_rotation_matrix<float, 3>(rng);
    std::cout << "Random 3D rotation matrix:\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(3) 
                      << rot_mat(i, j);
        }
        std::cout << "\n";
    }
    auto det = determinant(rot_mat);
    std::cout << "Determinant = " << det << " (should be 1)\n\n";
    
    // 6. Monte Carlo example: Estimating π
    std::cout << "6. Monte Carlo estimation of π:\n";
    
    const int num_samples = 100000;
    int inside_circle = 0;
    
    for (int i = 0; i < num_samples; ++i) {
        auto point = random_vector<float, 2>(rng, -1.0f, 1.0f);
        if (point.length_squared() <= 1.0f) {
            inside_circle++;
        }
    }
    
    float pi_estimate = 4.0f * static_cast<float>(inside_circle) / static_cast<float>(num_samples);
    std::cout << "Samples: " << num_samples << "\n";
    std::cout << "Points in circle: " << inside_circle << "\n";
    std::cout << "π estimate: " << pi_estimate << "\n";
    std::cout << "Actual π: " << constants<float>::pi << "\n";
    std::cout << "Error: " << std::abs(pi_estimate - constants<float>::pi) << "\n\n";
    
    // 7. Random points on geometric shapes
    std::cout << "7. Random points on geometric shapes:\n";
    
    // Random points on circle (2D)
    std::cout << "Points on unit circle:\n";
    for (int i = 0; i < 5; ++i) {
        auto angle = random_angle<float, radian_tag>(rng, 
            radian<float>(0), radian<float>(2 * constants<float>::pi));
        float x = cos(angle);
        float y = sin(angle);
        std::cout << "  (" << x << ", " << y << ")\n";
    }
    std::cout << "\n";
    
    // Random barycentric coordinates (for triangles)
    std::cout << "Random barycentric coordinates:\n";
    for (int i = 0; i < 3; ++i) {
        auto bary = random_on_simplex<float, 3>(rng);
        std::cout << "  (" << bary[0] << ", " << bary[1] << ", " << bary[2] 
                  << ") sum=" << bary[0] + bary[1] + bary[2] << "\n";
    }
    std::cout << "\n";
    
    // 8. Distribution testing
    std::cout << "8. Testing quaternion distribution:\n";
    
    // Generate many quaternions and check angle distribution
    const int n_samples = 10000;
    std::map<int, int> angle_histogram;
    
    for (int i = 0; i < n_samples; ++i) {
        auto q = random_quaternion<float>(rng);
        auto angle_deg = static_cast<int>(degree<float>(q.angle()).value());
        angle_histogram[angle_deg / 10]++;  // 10-degree bins
    }
    
    std::cout << "Angle distribution (should be roughly sin(θ/2)):\n";
    for (int bin = 0; bin <= 18; ++bin) {
        std::cout << std::setw(3) << (bin * 10) << "°-" 
                  << std::setw(3) << ((bin + 1) * 10) << "°: ";
        
        int count = angle_histogram[bin];
        int bar_length = count * 50 / n_samples;
        for (int j = 0; j < bar_length; ++j) {
            std::cout << "*";
        }
        std::cout << " " << count << "\n";
    }
    
    return 0;
}