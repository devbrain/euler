/**
 * @example 03_quaternion_rotations.cc
 * @brief Quaternion rotations and orientation
 * 
 * This example demonstrates:
 * - Creating quaternions from various representations
 * - Quaternion multiplication and interpolation
 * - Converting between quaternions and other rotation representations
 * - Rotating vectors with quaternions
 * - Avoiding gimbal lock
 */

#include <euler/euler.hh>
#include <iostream>
#include <iomanip>

using namespace euler;

// Helper to print quaternion info
template<typename T>
void print_quaternion(const std::string& name, const quaternion<T>& q) {
    std::cout << name << ": ";
    std::cout << "(" << q.w() << ", " << q.x() << "i, " << q.y() << "j, " << q.z() << "k)\n";
    std::cout << "  Norm: " << q.norm() << "\n";
    std::cout << "  Angle: " << degree<T>(q.angle()) << "\n";
    auto axis = q.axis();
    std::cout << "  Axis: (" << axis[0] << ", " << axis[1] << ", " << axis[2] << ")\n\n";
}

int main() {
    std::cout << "=== Euler Library: Quaternion Rotations Example ===\n\n";
    
    // 1. Creating quaternions
    std::cout << "1. Creating quaternions:\n";
    
    // Identity quaternion (no rotation)
    auto q_identity = quaternion<float>::identity();
    print_quaternion("Identity", q_identity);
    
    // From axis-angle
    vector<float, 3> axis = normalize(vector<float, 3>(1, 1, 1));
    auto angle = degree<float>(90);
    auto q1 = quaternion<float>::from_axis_angle(axis, angle);
    print_quaternion("90° around (1,1,1)", q1);
    
    // From Euler angles (XYZ order)
    auto roll = degree<float>(30);   // X rotation
    auto pitch = degree<float>(45);  // Y rotation
    auto yaw = degree<float>(60);    // Z rotation
    auto q2 = quaternion<float>::from_euler(roll, pitch, yaw, euler_order::XYZ);
    print_quaternion("From Euler (30°, 45°, 60°)", q2);
    
    // 2. Quaternion multiplication (composition of rotations)
    std::cout << "2. Composing rotations:\n";
    
    // Create two 90° rotations
    auto q_x = quaternion<float>::from_axis_angle(vector<float, 3>::unit_x(), degree<float>(90));
    auto q_y = quaternion<float>::from_axis_angle(vector<float, 3>::unit_y(), degree<float>(90));
    
    // Compose them (rotate around X, then Y)
    auto q_composed = q_y * q_x;  // Note: quaternion multiplication is right-to-left
    print_quaternion("90° X then 90° Y", q_composed);
    
    // 3. Rotating vectors
    std::cout << "3. Rotating vectors:\n";
    vector<float, 3> v(1, 0, 0);
    
    // Rotate using quaternion
    auto v_rotated = rotate(v, q_x);
    std::cout << "Original vector: " << v << "\n";
    std::cout << "After 90° rotation around X: " << v_rotated << "\n\n";
    
    // Multiple rotations
    auto v_double = rotate(rotate(v, q_x), q_y);
    auto v_composed = rotate(v, q_composed);
    std::cout << "Two separate rotations: " << v_double << "\n";
    std::cout << "Composed rotation: " << v_composed << "\n";
    std::cout << "(These should be equal)\n\n";
    
    // 4. Converting to rotation matrix
    std::cout << "4. Converting to rotation matrix:\n";
    auto rot_matrix = q1.to_matrix3();
    std::cout << "Quaternion rotation as 3x3 matrix:\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << rot_matrix(i, j);
        }
        std::cout << "\n";
    }
    
    // Verify it's the same rotation
    auto v_by_quat = rotate(v, q1);
    auto v_by_matrix = rot_matrix * v;
    std::cout << "\nRotation by quaternion: " << v_by_quat << "\n";
    std::cout << "Rotation by matrix: " << v_by_matrix << "\n\n";
    
    // 5. Extracting Euler angles
    std::cout << "5. Extracting Euler angles:\n";
    auto euler_angles = q2.to_euler(euler_order::XYZ);
    std::cout << "Original Euler angles: (" << roll << ", " << pitch << ", " << yaw << ")\n";
    std::cout << "Extracted angles: (" 
              << degree<float>(euler_angles[0]) << ", "
              << degree<float>(euler_angles[1]) << ", "
              << degree<float>(euler_angles[2]) << ")\n\n";
    
    // 6. Quaternion interpolation (SLERP)
    std::cout << "6. Spherical linear interpolation (SLERP):\n";
    auto q_start = quaternion<float>::identity();
    auto q_end = quaternion<float>::from_axis_angle(vector<float, 3>::unit_z(), degree<float>(180));
    
    std::cout << "Interpolating from identity to 180° Z rotation:\n";
    for (float t = 0; t <= 1.0f; t += 0.25f) {
        auto q_interp = slerp(q_start, q_end, t);
        auto angle_ = degree<float>(q_interp.angle());
        std::cout << "  t=" << t << ": angle=" << angle_ << "\n";
    }
    std::cout << "\n";
    
    // 7. Avoiding gimbal lock
    std::cout << "7. Gimbal lock demonstration:\n";
    
    // Create a gimbal lock situation with Euler angles
    auto gimbal_pitch = degree<float>(90);  // This causes gimbal lock
    auto q_gimbal = quaternion<float>::from_euler(
        degree<float>(30), gimbal_pitch, degree<float>(45), euler_order::XYZ
    );
    
    // Try to extract Euler angles back
    auto extracted = q_gimbal.to_euler(euler_order::XYZ);
    std::cout << "Input Euler angles: (30°, 90°, 45°)\n";
    std::cout << "Extracted angles: (" 
              << degree<float>(extracted[0]) << ", "
              << degree<float>(extracted[1]) << ", "
              << degree<float>(extracted[2]) << ")\n";
    std::cout << "Note: With 90° pitch, roll and yaw are coupled (gimbal lock)\n\n";
    
    // 8. Quaternion from two vectors
    std::cout << "8. Quaternion from two vectors:\n";
    vector<float, 3> from_vec = normalize(vector<float, 3>(1, 0, 0));
    vector<float, 3> to_vec = normalize(vector<float, 3>(0, 1, 1));
    
    auto q_align = quaternion<float>::from_vectors(from_vec, to_vec);
    print_quaternion("Rotation from (1,0,0) to (0,1,1)", q_align);
    
    // Verify it works
    auto aligned = rotate(from_vec, q_align);
    std::cout << "Original: " << from_vec << "\n";
    std::cout << "Target: " << to_vec << "\n";
    std::cout << "Rotated: " << aligned << "\n\n";
    
    // 9. Conjugate and inverse
    std::cout << "9. Conjugate and inverse:\n";
    auto q_conj = conjugate(q1);
    print_quaternion("Original", q1);
    print_quaternion("Conjugate", q_conj);
    
    // For unit quaternions, conjugate = inverse
    auto v_forward = rotate(v, q1);
    auto v_backward = rotate(v_forward, q_conj);
    std::cout << "Original vector: " << v << "\n";
    std::cout << "After rotation: " << v_forward << "\n";
    std::cout << "After inverse rotation: " << v_backward << "\n\n";
    
    // 10. Testing the fixed XZY order
    std::cout << "10. Testing XZY Euler order (bug fix verification):\n";
    auto q_xzy = quaternion<float>::from_euler(
        degree<float>(45), degree<float>(30), degree<float>(60), euler_order::XZY
    );
    
    std::cout << "Quaternion is normalized: " << (q_xzy.is_normalized() ? "YES" : "NO") << "\n";
    std::cout << "Norm: " << q_xzy.norm() << " (should be 1.0)\n";
    
    return 0;
}