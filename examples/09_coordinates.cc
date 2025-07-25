#include <euler/euler.hh>
#include <iostream>
#include <iomanip>

using namespace euler;

int main() {
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "=== Euler Coordinates Module Demo ===\n\n";
    
    // 2D Points
    std::cout << "2D Points:\n";
    point2f p1(3.0f, 4.0f);
    point2f p2(6.0f, 8.0f);
    std::cout << "  p1 = " << p1 << "\n";
    std::cout << "  p2 = " << p2 << "\n";
    
    // Point operations
    auto v = p2 - p1;  // Displacement vector
    std::cout << "  p2 - p1 = " << v << " (vector)\n";
    
    auto p3 = p1 + v;  // Translate point by vector
    std::cout << "  p1 + v = " << p3 << "\n";
    
    std::cout << "  distance(p1, p2) = " << distance(p1, p2) << "\n";
    std::cout << "  midpoint(p1, p2) = " << midpoint(p1, p2) << "\n";
    
    // Named constructors
    std::cout << "\nPolar coordinates:\n";
    auto p_polar = point2f::polar(5.0f, degree<float>(45));
    std::cout << "  polar(5, 45°) = " << p_polar << "\n";
    
    // Swizzling
    std::cout << "\nSwizzling:\n";
    std::cout << "  p1.xy() = " << p1.xy() << "\n";
    std::cout << "  p1.yx() = " << p1.yx() << "\n";
    
    // 3D Points
    std::cout << "\n3D Points:\n";
    point3f q1(1.0f, 2.0f, 3.0f);
    point3f q2(4.0f, 5.0f, 6.0f);
    std::cout << "  q1 = " << q1 << "\n";
    std::cout << "  q2 = " << q2 << "\n";
    
    // Spherical coordinates
    auto q_sph = point3f::spherical(10.0f, degree<float>(30), degree<float>(60));
    std::cout << "  spherical(10, 30°, 60°) = " << q_sph << "\n";
    
    // 2D Transformations
    std::cout << "\n2D Transformations:\n";
    point2f p(1.0f, 0.0f);
    
    // Rotation
    auto rot = rotation_matrix2(degree<float>(90));
    auto p_rot = rot * p;
    std::cout << "  rotate([1,0], 90°) = " << p_rot << "\n";
    
    // Scale
    auto scale = scale_matrix2(2.0f, 3.0f);
    auto p_scale = scale * p;
    std::cout << "  scale([1,0], 2, 3) = " << p_scale << "\n";
    
    // Translation
    auto trans = translation_matrix2(5.0f, 5.0f);
    auto p_trans = trans * p;
    std::cout << "  translate([1,0], 5, 5) = " << p_trans << "\n";
    
    // Combined transformation
    matrix<float, 3, 3> combined = trans * rot * scale;
    auto p_combined = combined * p;
    std::cout << "  combined transform = " << p_combined << "\n";
    
    // Projective coordinates
    std::cout << "\nProjective Coordinates:\n";
    projective2<float> proj1(6.0f, 8.0f, 2.0f);
    std::cout << "  proj1 = [" << proj1.x << ", " << proj1.y << ", " << proj1.w << "]\n";
    std::cout << "  proj1.point() = " << proj1.point() << "\n";
    
    // Point at infinity
    auto proj_inf = projective2<float>::at_infinity(1.0f, 0.0f);
    std::cout << "  at_infinity(1, 0) = [" << proj_inf.x << ", " << proj_inf.y 
              << ", " << proj_inf.w << "]\n";
    std::cout << "  is_infinite = " << (proj_inf.is_infinite() ? "true" : "false") << "\n";
    
    // Screen coordinate conversions
    std::cout << "\nScreen Coordinates:\n";
    point2f screen(400.0f, 300.0f);
    auto ndc = screen_to_ndc(screen, 800.0f, 600.0f);
    auto uv = screen_to_uv(screen, 800.0f, 600.0f);
    std::cout << "  screen(400, 300) -> NDC = " << ndc << "\n";
    std::cout << "  screen(400, 300) -> UV = " << uv << "\n";
    
    // 3D Graphics pipeline
    std::cout << "\n3D Graphics Pipeline:\n";
    
    // Define camera
    point3f eye(5.0f, 5.0f, 5.0f);
    point3f center(0.0f, 0.0f, 0.0f);
    vector<float, 3> up(0.0f, 1.0f, 0.0f);
    
    // View matrix
    auto view = look_at(eye, center, up);
    
    // Projection matrix
    auto proj = perspective(degree<float>(45), 1.33f, 0.1f, 100.0f);
    
    // Test point
    point3f world_point(1.0f, 0.0f, 0.0f);
    
    // Transform through pipeline
    auto view_point = view * world_point;
    std::cout << "  world " << world_point << " -> view " << view_point << "\n";
    
    // Lerp between points
    std::cout << "\nInterpolation:\n";
    auto p_lerp = lerp(p1, p2, 0.5f);
    std::cout << "  lerp(p1, p2, 0.5) = " << p_lerp << "\n";
    
    // Barycentric coordinates
    point2f a(0.0f, 0.0f);
    point2f b(10.0f, 0.0f);
    point2f c(0.0f, 10.0f);
    auto p_bary = barycentric(a, b, c, 0.33f, 0.33f, 0.34f);
    std::cout << "  barycentric center = " << p_bary << "\n";
    
    return 0;
}