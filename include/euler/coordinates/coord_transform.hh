/**
 * @file coord_transform.hh
 * @brief Coordinate system transformation utilities
 * @ingroup CoordinatesModule
 */
#pragma once

#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/matrix/matrix.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/angles/angle_traits.hh>
#include <euler/angles/angle_ops.hh>

namespace euler {

// 2D transformation matrix builders

/**
 * @brief Create 2D translation matrix
 * @param tx X translation
 * @param ty Y translation
 * @return 3x3 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 3, 3> translation_matrix2(T tx, T ty) {
    return {
        {1, 0, tx},
        {0, 1, ty},
        {0, 0,  1}
    };
}

/**
 * @brief Create 2D translation matrix from vector
 * @param t Translation vector
 * @return 3x3 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 3, 3> translation_matrix2(const vector<T, 2>& t) {
    return translation_matrix2(t[0], t[1]);
}

/**
 * @brief Create 2D rotation matrix
 * @param angle Rotation angle (accepts degree or radian)
 * @return 3x3 homogeneous transformation matrix
 */
template<typename Angle>
matrix<typename Angle::value_type, 3, 3> rotation_matrix2(const Angle& angle) {
    static_assert(is_angle_v<Angle>, "rotation_matrix2 requires angle type");
    using T = typename Angle::value_type;
    auto theta = to_radians(angle);
    T c = cos(theta);
    T s = sin(theta);
    return {
        {c, -s, 0},
        {s,  c, 0},
        {0,  0, 1}
    };
}

/**
 * @brief Create 2D scale matrix
 * @param sx X scale factor
 * @param sy Y scale factor
 * @return 3x3 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 3, 3> scale_matrix2(T sx, T sy) {
    return {
        {sx,  0, 0},
        { 0, sy, 0},
        { 0,  0, 1}
    };
}

/**
 * @brief Create uniform 2D scale matrix
 * @param s Scale factor for both axes
 * @return 3x3 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 3, 3> scale_matrix2(T s) {
    return scale_matrix2(s, s);
}

// 3D transformation matrix builders

/**
 * @brief Create 3D translation matrix
 * @param tx X translation
 * @param ty Y translation
 * @param tz Z translation
 * @return 4x4 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 4, 4> translation_matrix3(T tx, T ty, T tz) {
    return {
        {1, 0, 0, tx},
        {0, 1, 0, ty},
        {0, 0, 1, tz},
        {0, 0, 0,  1}
    };
}

/**
 * @brief Create 3D translation matrix from vector
 * @param t Translation vector
 * @return 4x4 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 4, 4> translation_matrix3(const vector<T, 3>& t) {
    return translation_matrix3(t[0], t[1], t[2]);
}

/**
 * @brief Create 3D rotation matrix around X axis
 * @param angle Rotation angle (accepts degree or radian)
 * @return 4x4 homogeneous transformation matrix
 */
template<typename Angle>
matrix<typename Angle::value_type, 4, 4> rotation_matrix3_x(const Angle& angle) {
    static_assert(is_angle_v<Angle>, "rotation_matrix3_x requires angle type");
    using T = typename Angle::value_type;
    auto theta = to_radians(angle);
    T c = cos(theta);
    T s = sin(theta);
    return {
        {1,  0,  0, 0},
        {0,  c, -s, 0},
        {0,  s,  c, 0},
        {0,  0,  0, 1}
    };
}

/**
 * @brief Create 3D rotation matrix around Y axis
 * @param angle Rotation angle (accepts degree or radian)
 * @return 4x4 homogeneous transformation matrix
 */
template<typename Angle>
matrix<typename Angle::value_type, 4, 4> rotation_matrix3_y(const Angle& angle) {
    static_assert(is_angle_v<Angle>, "rotation_matrix3_y requires angle type");
    using T = typename Angle::value_type;
    auto theta = to_radians(angle);
    T c = cos(theta);
    T s = sin(theta);
    return {
        { c,  0,  s, 0},
        { 0,  1,  0, 0},
        {-s,  0,  c, 0},
        { 0,  0,  0, 1}
    };
}

/**
 * @brief Create 3D rotation matrix around Z axis
 * @param angle Rotation angle (accepts degree or radian)
 * @return 4x4 homogeneous transformation matrix
 */
template<typename Angle>
matrix<typename Angle::value_type, 4, 4> rotation_matrix3_z(const Angle& angle) {
    static_assert(is_angle_v<Angle>, "rotation_matrix3_z requires angle type");
    using T = typename Angle::value_type;
    auto theta = to_radians(angle);
    T c = cos(theta);
    T s = sin(theta);
    return {
        {c, -s, 0, 0},
        {s,  c, 0, 0},
        {0,  0, 1, 0},
        {0,  0, 0, 1}
    };
}

/**
 * @brief Create 3D scale matrix
 * @param sx X scale factor
 * @param sy Y scale factor
 * @param sz Z scale factor
 * @return 4x4 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 4, 4> scale_matrix3(T sx, T sy, T sz) {
    return {
        {sx,  0,  0, 0},
        { 0, sy,  0, 0},
        { 0,  0, sz, 0},
        { 0,  0,  0, 1}
    };
}

/**
 * @brief Create uniform 3D scale matrix
 * @param s Scale factor for all axes
 * @return 4x4 homogeneous transformation matrix
 */
template<typename T>
constexpr matrix<T, 4, 4> scale_matrix3(T s) {
    return scale_matrix3(s, s, s);
}

// Coordinate system conversions

/**
 * @brief Convert screen coordinates to normalized device coordinates (NDC)
 * @param screen Screen coordinates (origin at top-left)
 * @param width Screen width
 * @param height Screen height
 * @return NDC coordinates in range [-1, 1] (origin at center)
 */
template<typename T>
point2<T> screen_to_ndc(const point2<T>& screen, T width, T height) {
    return {
        (screen.x / width) * 2 - 1,
        1 - (screen.y / height) * 2  // Flip Y axis
    };
}

/**
 * @brief Convert normalized device coordinates (NDC) to screen coordinates
 * @param ndc NDC coordinates in range [-1, 1]
 * @param width Screen width
 * @param height Screen height
 * @return Screen coordinates (origin at top-left)
 */
template<typename T>
point2<T> ndc_to_screen(const point2<T>& ndc, T width, T height) {
    return {
        (ndc.x + 1) * width / 2,
        (1 - ndc.y) * height / 2  // Flip Y axis
    };
}

/**
 * @brief Convert screen coordinates to UV texture coordinates
 * @param screen Screen coordinates
 * @param width Screen/texture width
 * @param height Screen/texture height
 * @return UV coordinates in range [0, 1]
 */
template<typename T>
point2<T> screen_to_uv(const point2<T>& screen, T width, T height) {
    return {
        screen.x / width,
        screen.y / height
    };
}

/**
 * @brief Convert UV texture coordinates to screen coordinates
 * @param uv UV coordinates in range [0, 1]
 * @param width Screen/texture width
 * @param height Screen/texture height
 * @return Screen coordinates
 */
template<typename T>
point2<T> uv_to_screen(const point2<T>& uv, T width, T height) {
    return {
        uv.x * width,
        uv.y * height
    };
}

/**
 * @brief Create view transformation matrix (look-at)
 * @param eye Eye position
 * @param center Target position to look at
 * @param up Up direction vector
 * @return 4x4 view transformation matrix
 */
template<typename T>
matrix<T, 4, 4> look_at(const point3<T>& eye, const point3<T>& center, 
                       const vector<T, 3>& up) {
    vector<T, 3> f = normalize(center - eye);  // Forward
    vector<T, 3> s = normalize(cross(f, up));  // Right
    vector<T, 3> u = cross(s, f);              // Up
    
    return {
        { s[0],  s[1],  s[2], -dot(s, eye.vec())},
        { u[0],  u[1],  u[2], -dot(u, eye.vec())},
        {-f[0], -f[1], -f[2],  dot(f, eye.vec())},
        {    0,     0,     0,                  1}
    };
}

/**
 * @brief Create perspective projection matrix
 * @param fovy Vertical field of view angle
 * @param aspect Aspect ratio (width/height)
 * @param z_near Near clipping plane distance
 * @param z_far Far clipping plane distance
 * @return 4x4 perspective projection matrix
 */
template<typename Angle, typename T = typename Angle::value_type>
matrix<T, 4, 4> perspective(const Angle& fovy, T aspect, T z_near, T z_far) {
    static_assert(is_angle_v<Angle>, "perspective requires angle type");

    EULER_CHECK(aspect > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "perspective: aspect ratio must be positive");
    EULER_CHECK(std::abs(z_near - z_far) > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "perspective: z_near and z_far must be different");

    auto fovy_rad = to_radians(fovy);
    T half_fovy = fovy_rad / T(2);
    T tan_half_fovy = tan(half_fovy);

    EULER_CHECK(std::abs(tan_half_fovy) > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "perspective: fovy must not be 0 or pi");

    T f = T(1) / tan_half_fovy;
    T nf = T(1) / (z_near - z_far);

    return {
        {f/aspect,  0,                           0,  0},
        {       0,  f,                           0,  0},
        {       0,  0,     (z_far + z_near) * nf, 2 * z_far * z_near * nf},
        {       0,  0,                          -1,  0}
    };
}

/**
 * @brief Create orthographic projection matrix
 * @param left Left clipping plane
 * @param right Right clipping plane
 * @param bottom Bottom clipping plane
 * @param top Top clipping plane
 * @param z_near Near clipping plane
 * @param z_far Far clipping plane
 * @return 4x4 orthographic projection matrix
 */
template<typename T>
matrix<T, 4, 4> ortho(T left, T right, T bottom, T top, T z_near, T z_far) {
    EULER_CHECK(std::abs(right - left) > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "ortho: right and left must be different");
    EULER_CHECK(std::abs(top - bottom) > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "ortho: top and bottom must be different");
    EULER_CHECK(std::abs(z_far - z_near) > std::numeric_limits<T>::epsilon(),
                error_code::invalid_argument, "ortho: z_far and z_near must be different");

    T rl = T(1) / (right - left);
    T tb = T(1) / (top - bottom);
    T fn = T(1) / (z_far - z_near);

    return {
        {2*rl,     0,     0, -(right + left) * rl},
        {   0,  2*tb,     0, -(top + bottom) * tb},
        {   0,     0, -2*fn,   -(z_far + z_near) * fn},
        {   0,     0,     0,                         1}
    };
}

} // namespace euler