/**
 * @file coordinates_io_example.cc
 * @brief Example demonstrating coordinate system I/O operators
 */

#include <euler/coordinates/io.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/projective3.hh>
#include <euler/dda/io.hh>
#include <euler/dda/dda_traits.hh>
#include <iostream>
#include <iomanip>

using namespace euler;
using namespace euler::dda;

int main() {
    std::cout << "=== Coordinate System I/O Examples ===\n\n";
    
    // 2D points
    std::cout << "2D Points:\n";
    point2f p2f{1.5f, 2.5f};
    point2i p2i{10, 20};
    std::cout << "  Float: " << p2f << "\n";
    std::cout << "  Integer: " << p2i << "\n\n";
    
    // 3D points
    std::cout << "3D Points:\n";
    point3f p3f{1.0f, 2.0f, 3.0f};
    point3d p3d{1.234567, 2.345678, 3.456789};
    std::cout << "  Float: " << p3f << "\n";
    std::cout << "  Double: " << p3d << "\n";
    std::cout << "  With precision: " << std::fixed << std::setprecision(3) << p3d << "\n\n";
    
    // Reset formatting
    std::cout.unsetf(std::ios::fixed);
    
    // Projective coordinates
    std::cout << "Projective Coordinates:\n";
    projective2<float> proj2{100.0f, 200.0f, 2.0f};
    projective3<double> proj3{100.0, 200.0, 300.0, 2.0};
    std::cout << "  2D: " << proj2 << "\n";
    std::cout << "  3D: " << proj3 << "\n\n";
    
    // DDA-specific types
    std::cout << "DDA Types:\n";
    pixel<int> pix{{15, 25}};
    aa_pixel<float> aa_pix{{15.5f, 25.5f}, 0.75f, 0.125f};
    span s{100, 10, 90};
    rectangle<int> rect{{0, 0}, {640, 480}};
    
    std::cout << "  Pixel: " << pix << "\n";
    std::cout << "  AA Pixel: " << aa_pix << "\n";
    std::cout << "  Span: " << s << " (width: " << s.width() << ")\n";
    std::cout << "  Rectangle: " << rect << "\n\n";
    
    // Enums
    std::cout << "DDA Enums:\n";
    std::cout << "  Curve types: " << curve_type::parametric << ", " 
              << curve_type::cartesian << ", " << curve_type::polar << "\n";
    std::cout << "  Cap styles: " << cap_style::butt << ", " 
              << cap_style::round << ", " << cap_style::square << "\n";
    std::cout << "  AA algorithms: " << aa_algorithm::wu << ", " 
              << aa_algorithm::gupta_sproull << ", " << aa_algorithm::supersampling << "\n\n";
    
    // Formatting examples
    std::cout << "Formatting Examples:\n";
    point3d p{1.23456789, 2.34567890, 3.45678901};
    
    std::cout << "  Default: " << p << "\n";
    std::cout << "  Fixed 2: " << std::fixed << std::setprecision(2) << p << "\n";
    std::cout << "  Fixed 5: " << std::fixed << std::setprecision(5) << p << "\n";
    std::cout << "  Width 10: " << std::setw(10) << p << "\n";
    
    return 0;
}