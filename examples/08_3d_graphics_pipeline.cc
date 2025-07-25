/**
 * @example 08_3d_graphics_pipeline.cc
 * @brief Complete 3D graphics transformation pipeline
 * 
 * This example demonstrates a complete 3D graphics pipeline:
 * - Model, view, and projection matrices
 * - Camera controls with quaternions
 * - Object transformations
 * - Frustum calculations
 * - Screen space conversion
 * - Practical rendering setup
 */

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace euler;

// Simple 3D object - a cube
struct Mesh {
    std::vector<vector<float, 3>> vertices;
    std::vector<std::array<int, 3>> triangles;
    
    static Mesh create_cube(float size = 1.0f) {
        Mesh cube;
        float h = size / 2;
        
        // 8 vertices of a cube
        cube.vertices = {
            {-h, -h, -h}, {+h, -h, -h}, {+h, +h, -h}, {-h, +h, -h},  // Front
            {-h, -h, +h}, {+h, -h, +h}, {+h, +h, +h}, {-h, +h, +h}   // Back
        };
        
        // 12 triangles (2 per face)
        cube.triangles = {
            {0,1,2}, {0,2,3},  // Front
            {5,4,7}, {5,7,6},  // Back
            {4,0,3}, {4,3,7},  // Left
            {1,5,6}, {1,6,2},  // Right
            {3,2,6}, {3,6,7},  // Top
            {4,5,1}, {4,1,0}   // Bottom
        };
        
        return cube;
    }
};

// Camera class using quaternions
class Camera {
public:
    vector<float, 3> position;
    quaternion<float> orientation;
    float fov;
    float aspect_ratio;
    float near_plane;
    float far_plane;
    
    Camera() 
        : position(0, 0, 5)
        , orientation(quaternion<float>::identity())
        , fov(60.0f)
        , aspect_ratio(16.0f/9.0f)
        , near_plane(0.1f)
        , far_plane(100.0f) {}
    
    // Look at a target
    void look_at(const vector<float, 3>& target, const vector<float, 3>& up = {0, 1, 0}) {
        vector<float, 3> forward = normalize(target - position);
        vector<float, 3> right = normalize(cross(forward, up));
        vector<float, 3> real_up = cross(right, forward);
        
        // Create rotation matrix from basis vectors
        matrix<float, 3, 3> rot_matrix;
        rot_matrix(0, 0) = right[0];    rot_matrix(0, 1) = real_up[0];    rot_matrix(0, 2) = -forward[0];
        rot_matrix(1, 0) = right[1];    rot_matrix(1, 1) = real_up[1];    rot_matrix(1, 2) = -forward[1];
        rot_matrix(2, 0) = right[2];    rot_matrix(2, 1) = real_up[2];    rot_matrix(2, 2) = -forward[2];
        
        orientation = quaternion<float>::from_matrix(rot_matrix);
    }
    
    // Get view matrix
    matrix<float, 4, 4> get_view_matrix() const {
        // Convert quaternion to rotation matrix
        auto rot3x3 = conjugate(orientation).to_matrix3();
        
        // Expand to 4x4
        matrix<float, 4, 4> view = matrix<float, 4, 4>::identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                view(i, j) = rot3x3(i, j);
            }
        }
        
        // Apply translation
        auto rotated_pos = rotate(position, conjugate(orientation));
        view(0, 3) = -rotated_pos[0];
        view(1, 3) = -rotated_pos[1];
        view(2, 3) = -rotated_pos[2];
        
        return view;
    }
    
    // Get projection matrix
    matrix<float, 4, 4> get_projection_matrix() const {
        matrix<float, 4, 4> proj(0.0f);
        
        float fov_rad = radian<float>(degree<float>(fov)).value();
        float f = 1.0f / std::tan(fov_rad / 2.0f);
        float range_inv = 1.0f / (near_plane - far_plane);
        
        proj(0, 0) = f / aspect_ratio;
        proj(1, 1) = f;
        proj(2, 2) = (near_plane + far_plane) * range_inv;
        proj(2, 3) = 2.0f * near_plane * far_plane * range_inv;
        proj(3, 2) = -1.0f;
        
        return proj;
    }
};

// Transform a point through the graphics pipeline
vector<float, 4> transform_point(const vector<float, 3>& point,
                                const matrix<float, 4, 4>& model,
                                const matrix<float, 4, 4>& view,
                                const matrix<float, 4, 4>& projection) {
    // Convert to homogeneous coordinates
    vector<float, 4> p(point[0], point[1], point[2], 1.0f);
    
    // Apply transformations
    p = model * p;      // Model space to world space
    p = view * p;       // World space to camera space
    p = projection * p; // Camera space to clip space
    
    return p;
}

// Convert from clip space to screen space
vector<float, 2> to_screen_space(const vector<float, 4>& clip_pos,
                                float screen_width, float screen_height) {
    // Perspective divide
    float x_ndc = clip_pos[0] / clip_pos[3];
    float y_ndc = clip_pos[1] / clip_pos[3];
    
    // Convert from [-1,1] to screen coordinates
    float x_screen = (x_ndc + 1.0f) * 0.5f * screen_width;
    float y_screen = (1.0f - y_ndc) * 0.5f * screen_height;  // Flip Y
    
    return vector<float, 2>(x_screen, y_screen);
}

int main() {
    std::cout << "=== Euler Library: 3D Graphics Pipeline Example ===\n\n";
    
    // 1. Setup scene
    std::cout << "1. Scene setup:\n";
    
    // Create objects
    auto cube = Mesh::create_cube(1.0f);
    std::cout << "Created cube with " << cube.vertices.size() << " vertices\n";
    
    // Camera setup
    Camera camera;
    camera.position = vector<float, 3>(3, 4, 5);
    camera.look_at(vector<float, 3>(0, 0, 0));
    
    std::cout << "Camera position: " << camera.position << "\n";
    std::cout << "Camera orientation: " << degree<float>(camera.orientation.angle()) 
              << " around " << camera.orientation.axis() << "\n\n";
    
    // 2. Object transformations
    std::cout << "2. Object transformations:\n";
    
    // Create model matrix with rotation and scale
    auto rotation = quaternion<float>::from_euler(
        degree<float>(30), degree<float>(45), degree<float>(0)
    );
    auto scale = vector<float, 3>(1.5f, 1.0f, 0.8f);
    auto translation = vector<float, 3>(0.5f, 0, 0);
    
    // Build model matrix
    matrix<float, 4, 4> model = matrix<float, 4, 4>::identity();
    
    // Apply scale
    model(0, 0) = scale[0];
    model(1, 1) = scale[1];
    model(2, 2) = scale[2];
    
    // Apply rotation
    auto rot_mat = rotation.to_matrix3();
    matrix<float, 4, 4> rot4x4 = matrix<float, 4, 4>::identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rot4x4(i, j) = rot_mat(i, j);
        }
    }
    model = rot4x4 * model;
    
    // Apply translation
    model(0, 3) = translation[0];
    model(1, 3) = translation[1];
    model(2, 3) = translation[2];
    
    std::cout << "Model transformation:\n";
    std::cout << "  Scale: " << scale << "\n";
    std::cout << "  Rotation: " << degree<float>(rotation.angle()) << "\n";
    std::cout << "  Translation: " << translation << "\n\n";
    
    // 3. Transform vertices through pipeline
    std::cout << "3. Vertex transformation:\n";
    
    auto view = camera.get_view_matrix();
    auto projection = camera.get_projection_matrix();
    
    // Transform first few vertices
    std::cout << "Transforming cube vertices:\n";
    std::cout << std::setw(20) << "World" << std::setw(20) << "Camera" 
              << std::setw(20) << "Clip" << std::setw(20) << "Screen\n";
    
    const float screen_width = 1920;
    const float screen_height = 1080;
    
    for (size_t i = 0; i < 4; ++i) {
        auto& vertex = cube.vertices[i];
        
        // World space
        vector<float, 4> world(vertex[0], vertex[1], vertex[2], 1.0f);
        world = model * world;
        
        // Camera space
        auto camera_space = view * world;
        
        // Clip space
        auto clip_space = projection * camera_space;
        
        // Screen space
        auto screen = to_screen_space(clip_space, screen_width, screen_height);
        
        std::cout << "V" << i << ": ";
        std::cout << std::setw(15) << "(" << world[0] << "," << world[1] << "," << world[2] << ")";
        std::cout << std::setw(15) << "(" << camera_space[0] << "," << camera_space[1] << "," << camera_space[2] << ")";
        std::cout << std::setw(15) << "(" << clip_space[0] << "," << clip_space[1] << "," << clip_space[2] << ")";
        std::cout << std::setw(15) << "(" << screen[0] << "," << screen[1] << ")\n";
    }
    std::cout << "\n";
    
    // 4. Camera movement
    std::cout << "4. Camera movement with quaternions:\n";
    
    // Orbit camera around origin
    const int steps = 8;
    const float radius = 5.0f;
    
    std::cout << "Orbiting camera around origin:\n";
    for (int i = 0; i <= steps; ++i) {
        float angle = 2.0f * constants<float>::pi * i / steps;
        
        // Update camera position
        camera.position = vector<float, 3>(
            radius * std::cos(angle),
            3.0f,
            radius * std::sin(angle)
        );
        camera.look_at(vector<float, 3>(0, 0, 0));
        
        std::cout << "  Step " << i << ": pos=" << camera.position 
                  << ", angle=" << degree<float>(radian<float>(angle)) << "\n";
    }
    std::cout << "\n";
    
    // 5. Frustum calculations
    std::cout << "5. Frustum calculations:\n";
    
    float fov_rad = radian<float>(degree<float>(camera.fov)).value();
    float half_height = camera.near_plane * std::tan(fov_rad / 2.0f);
    float half_width = half_height * camera.aspect_ratio;
    
    std::cout << "Near plane dimensions:\n";
    std::cout << "  Width: " << 2 * half_width << "\n";
    std::cout << "  Height: " << 2 * half_height << "\n";
    
    // Frustum corners in camera space
    std::vector<vector<float, 3>> frustum_corners = {
        // Near plane
        {-half_width, -half_height, -camera.near_plane},
        { half_width, -half_height, -camera.near_plane},
        { half_width,  half_height, -camera.near_plane},
        {-half_width,  half_height, -camera.near_plane},
        // Far plane
        {-half_width * camera.far_plane / camera.near_plane, 
         -half_height * camera.far_plane / camera.near_plane, -camera.far_plane},
        { half_width * camera.far_plane / camera.near_plane, 
         -half_height * camera.far_plane / camera.near_plane, -camera.far_plane},
        { half_width * camera.far_plane / camera.near_plane,  
          half_height * camera.far_plane / camera.near_plane, -camera.far_plane},
        {-half_width * camera.far_plane / camera.near_plane,  
          half_height * camera.far_plane / camera.near_plane, -camera.far_plane}
    };
    
    std::cout << "\nFrustum corners in world space:\n";
    for (size_t i = 0; i < 4; ++i) {
        auto world_pos = rotate(frustum_corners[i], camera.orientation) + camera.position;
        std::cout << "  Near " << i << ": " << world_pos << "\n";
    }
    std::cout << "\n";
    
    // 6. Animation with interpolation
    std::cout << "6. Smooth animation with quaternion SLERP:\n";
    
    auto start_rot = quaternion<float>::identity();
    auto end_rot = quaternion<float>::from_axis_angle(
        vector<float, 3>::unit_y(), degree<float>(180)
    );
    
    std::cout << "Interpolating 180° rotation around Y:\n";
    for (float t = 0; t <= 1.0f; t += 0.25f) {
        auto current_rot = slerp(start_rot, end_rot, t);
        auto angle = degree<float>(current_rot.angle());
        
        std::cout << "  t=" << std::fixed << std::setprecision(2) << t 
                  << ": " << angle << "\n";
    }
    
    return 0;
}