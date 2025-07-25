#include <doctest/doctest.h>
#include <euler/coordinates/coord_transform.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/matrix/matrix_ops.hh>
#include <euler/vector/vector_ops.hh>

using namespace euler;

TEST_CASE("2D transformation matrices") {
    SUBCASE("translation matrix") {
        auto m = translation_matrix2(3.0f, 4.0f);
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 0.0f);
        CHECK(m(0, 2) == 3.0f);
        CHECK(m(1, 0) == 0.0f);
        CHECK(m(1, 1) == 1.0f);
        CHECK(m(1, 2) == 4.0f);
        CHECK(m(2, 0) == 0.0f);
        CHECK(m(2, 1) == 0.0f);
        CHECK(m(2, 2) == 1.0f);
        
        // Test with vector
        vector<float, 2> t(5.0f, 6.0f);
        auto m2 = translation_matrix2(t);
        CHECK(m2(0, 2) == 5.0f);
        CHECK(m2(1, 2) == 6.0f);
    }
    
    SUBCASE("rotation matrix with degrees") {
        auto m = rotation_matrix2(degree<float>(90));
        CHECK(m(0, 0) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(0, 1) == doctest::Approx(-1.0f));
        CHECK(m(1, 0) == doctest::Approx(1.0f));
        CHECK(m(1, 1) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(2, 2) == 1.0f);
    }
    
    SUBCASE("rotation matrix with radians") {
        auto m = rotation_matrix2(radian<float>(pi/2));
        CHECK(m(0, 0) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(0, 1) == doctest::Approx(-1.0f));
        CHECK(m(1, 0) == doctest::Approx(1.0f));
        CHECK(m(1, 1) == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    
    SUBCASE("scale matrix") {
        auto m = scale_matrix2(2.0f, 3.0f);
        CHECK(m(0, 0) == 2.0f);
        CHECK(m(0, 1) == 0.0f);
        CHECK(m(1, 0) == 0.0f);
        CHECK(m(1, 1) == 3.0f);
        CHECK(m(2, 2) == 1.0f);
        
        // Uniform scale
        auto m2 = scale_matrix2(4.0f);
        CHECK(m2(0, 0) == 4.0f);
        CHECK(m2(1, 1) == 4.0f);
    }
}

TEST_CASE("3D transformation matrices") {
    SUBCASE("translation matrix") {
        auto m = translation_matrix3(3.0f, 4.0f, 5.0f);
        CHECK(m(0, 3) == 3.0f);
        CHECK(m(1, 3) == 4.0f);
        CHECK(m(2, 3) == 5.0f);
        CHECK(m(3, 3) == 1.0f);
        
        // Test with vector
        vector<float, 3> t(6.0f, 7.0f, 8.0f);
        auto m2 = translation_matrix3(t);
        CHECK(m2(0, 3) == 6.0f);
        CHECK(m2(1, 3) == 7.0f);
        CHECK(m2(2, 3) == 8.0f);
    }
    
    SUBCASE("rotation matrix X") {
        auto m = rotation_matrix3_x(degree<float>(90));
        // X axis unchanged
        CHECK(m(0, 0) == 1.0f);
        CHECK(m(0, 1) == 0.0f);
        CHECK(m(0, 2) == 0.0f);
        // Y -> Z rotation
        CHECK(m(1, 1) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(1, 2) == doctest::Approx(-1.0f));
        CHECK(m(2, 1) == doctest::Approx(1.0f));
        CHECK(m(2, 2) == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    
    SUBCASE("rotation matrix Y") {
        auto m = rotation_matrix3_y(degree<float>(90));
        // Y axis unchanged
        CHECK(m(1, 0) == 0.0f);
        CHECK(m(1, 1) == 1.0f);
        CHECK(m(1, 2) == 0.0f);
        // Z -> X rotation
        CHECK(m(0, 0) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(0, 2) == doctest::Approx(1.0f));
        CHECK(m(2, 0) == doctest::Approx(-1.0f));
        CHECK(m(2, 2) == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    
    SUBCASE("rotation matrix Z") {
        auto m = rotation_matrix3_z(degree<float>(90));
        // Z axis unchanged
        CHECK(m(2, 0) == 0.0f);
        CHECK(m(2, 1) == 0.0f);
        CHECK(m(2, 2) == 1.0f);
        // X -> Y rotation
        CHECK(m(0, 0) == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(m(0, 1) == doctest::Approx(-1.0f));
        CHECK(m(1, 0) == doctest::Approx(1.0f));
        CHECK(m(1, 1) == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    
    SUBCASE("scale matrix") {
        auto m = scale_matrix3(2.0f, 3.0f, 4.0f);
        CHECK(m(0, 0) == 2.0f);
        CHECK(m(1, 1) == 3.0f);
        CHECK(m(2, 2) == 4.0f);
        CHECK(m(3, 3) == 1.0f);
        
        // Uniform scale
        auto m2 = scale_matrix3(5.0f);
        CHECK(m2(0, 0) == 5.0f);
        CHECK(m2(1, 1) == 5.0f);
        CHECK(m2(2, 2) == 5.0f);
    }
}

TEST_CASE("coordinate system conversions") {
    SUBCASE("screen to NDC") {
        point2f screen(800.0f, 600.0f);
        auto ndc = screen_to_ndc(screen, 1600.0f, 1200.0f);
        CHECK(ndc.x == doctest::Approx(0.0f));
        CHECK(ndc.y == doctest::Approx(0.0f));
        
        // Top-left corner
        screen = point2f(0.0f, 0.0f);
        ndc = screen_to_ndc(screen, 1600.0f, 1200.0f);
        CHECK(ndc.x == doctest::Approx(-1.0f));
        CHECK(ndc.y == doctest::Approx(1.0f));
        
        // Bottom-right corner
        screen = point2f(1600.0f, 1200.0f);
        ndc = screen_to_ndc(screen, 1600.0f, 1200.0f);
        CHECK(ndc.x == doctest::Approx(1.0f));
        CHECK(ndc.y == doctest::Approx(-1.0f));
    }
    
    SUBCASE("NDC to screen") {
        point2f ndc(0.0f, 0.0f);
        auto screen = ndc_to_screen(ndc, 1600.0f, 1200.0f);
        CHECK(screen.x == doctest::Approx(800.0f));
        CHECK(screen.y == doctest::Approx(600.0f));
        
        // Top-left
        ndc = point2f(-1.0f, 1.0f);
        screen = ndc_to_screen(ndc, 1600.0f, 1200.0f);
        CHECK(screen.x == doctest::Approx(0.0f));
        CHECK(screen.y == doctest::Approx(0.0f));
        
        // Bottom-right
        ndc = point2f(1.0f, -1.0f);
        screen = ndc_to_screen(ndc, 1600.0f, 1200.0f);
        CHECK(screen.x == doctest::Approx(1600.0f));
        CHECK(screen.y == doctest::Approx(1200.0f));
    }
    
    SUBCASE("screen to UV") {
        point2f screen(400.0f, 300.0f);
        auto uv = screen_to_uv(screen, 800.0f, 600.0f);
        CHECK(uv.x == doctest::Approx(0.5f));
        CHECK(uv.y == doctest::Approx(0.5f));
        
        // Top-left
        screen = point2f(0.0f, 0.0f);
        uv = screen_to_uv(screen, 800.0f, 600.0f);
        CHECK(uv.x == doctest::Approx(0.0f));
        CHECK(uv.y == doctest::Approx(0.0f));
        
        // Bottom-right
        screen = point2f(800.0f, 600.0f);
        uv = screen_to_uv(screen, 800.0f, 600.0f);
        CHECK(uv.x == doctest::Approx(1.0f));
        CHECK(uv.y == doctest::Approx(1.0f));
    }
    
    SUBCASE("UV to screen") {
        point2f uv(0.5f, 0.5f);
        auto screen = uv_to_screen(uv, 800.0f, 600.0f);
        CHECK(screen.x == doctest::Approx(400.0f));
        CHECK(screen.y == doctest::Approx(300.0f));
        
        // Origin
        uv = point2f(0.0f, 0.0f);
        screen = uv_to_screen(uv, 800.0f, 600.0f);
        CHECK(screen.x == doctest::Approx(0.0f));
        CHECK(screen.y == doctest::Approx(0.0f));
        
        // Full extent
        uv = point2f(1.0f, 1.0f);
        screen = uv_to_screen(uv, 800.0f, 600.0f);
        CHECK(screen.x == doctest::Approx(800.0f));
        CHECK(screen.y == doctest::Approx(600.0f));
    }
}

TEST_CASE("point transformations") {
    SUBCASE("2D point translation") {
        point2f p(1.0f, 2.0f);
        auto m = translation_matrix2(3.0f, 4.0f);
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(4.0f));
        CHECK(p2.y == doctest::Approx(6.0f));
        
        // Using operator
        auto p3 = m * p;
        CHECK(p3.x == doctest::Approx(4.0f));
        CHECK(p3.y == doctest::Approx(6.0f));
    }
    
    SUBCASE("2D point rotation") {
        point2f p(1.0f, 0.0f);
        auto m = rotation_matrix2(degree<float>(90));
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p2.y == doctest::Approx(1.0f));
    }
    
    SUBCASE("2D point scale") {
        point2f p(2.0f, 3.0f);
        auto m = scale_matrix2(2.0f, 3.0f);
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(4.0f));
        CHECK(p2.y == doctest::Approx(9.0f));
    }
    
    SUBCASE("2D combined transformation") {
        point2f p(1.0f, 0.0f);
        // Scale, then rotate, then translate
        matrix<float, 3, 3> m = translation_matrix2(5.0f, 5.0f) * 
                                rotation_matrix2(degree<float>(90)) * 
                                scale_matrix2(2.0f);
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(5.0f).epsilon(1e-6f));
        CHECK(p2.y == doctest::Approx(7.0f));
    }
    
    SUBCASE("3D point translation") {
        point3f p(1.0f, 2.0f, 3.0f);
        auto m = translation_matrix3(4.0f, 5.0f, 6.0f);
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(5.0f));
        CHECK(p2.y == doctest::Approx(7.0f));
        CHECK(p2.z == doctest::Approx(9.0f));
        
        // Using operator
        auto p3 = m * p;
        CHECK(p3.x == doctest::Approx(5.0f));
        CHECK(p3.y == doctest::Approx(7.0f));
        CHECK(p3.z == doctest::Approx(9.0f));
    }
    
    SUBCASE("3D point rotation") {
        point3f p(1.0f, 0.0f, 0.0f);
        auto m = rotation_matrix3_z(degree<float>(90));
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p2.y == doctest::Approx(1.0f));
        CHECK(p2.z == doctest::Approx(0.0f));
    }
    
    SUBCASE("3D point scale") {
        point3f p(2.0f, 3.0f, 4.0f);
        auto m = scale_matrix3(2.0f, 3.0f, 4.0f);
        auto p2 = transform(m, p);
        CHECK(p2.x == doctest::Approx(4.0f));
        CHECK(p2.y == doctest::Approx(9.0f));
        CHECK(p2.z == doctest::Approx(16.0f));
    }
}

TEST_CASE("view and projection matrices") {
    SUBCASE("look at matrix") {
        point3f eye(0.0f, 0.0f, 5.0f);
        point3f center(0.0f, 0.0f, 0.0f);
        vector<float, 3> up(0.0f, 1.0f, 0.0f);
        
        auto m = look_at(eye, center, up);
        
        // Eye should transform to origin
        auto eye_transformed = m * eye;
        CHECK(eye_transformed.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(eye_transformed.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(eye_transformed.z == doctest::Approx(0.0f).epsilon(1e-6f));
        
        // Center should be on negative Z axis
        auto center_transformed = m * center;
        CHECK(center_transformed.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(center_transformed.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(center_transformed.z == doctest::Approx(-5.0f));
    }
    
    SUBCASE("perspective matrix") {
        auto m = perspective(degree<float>(45), 1.0f, 0.1f, 100.0f);
        
        // Near plane point should map to -1
        projective3<float> near_point(0.0f, 0.0f, -0.1f, 1.0f);
        vector<float, 4> v = m * near_point.vec();
        projective3<float> result(v[0], v[1], v[2], v[3]);
        result = result.normalized();
        CHECK(result.z == doctest::Approx(-1.0f).epsilon(1e-5f));
        
        // Far plane point should map to 1
        projective3<float> far_point(0.0f, 0.0f, -100.0f, 1.0f);
        vector<float, 4> v2 = m * far_point.vec();
        projective3<float> result2(v2[0], v2[1], v2[2], v2[3]);
        result2 = result2.normalized();
        CHECK(result2.z == doctest::Approx(1.0f).epsilon(1e-3f));
    }
    
    SUBCASE("orthographic matrix") {
        auto m = ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 100.0f);
        
        // Center point should remain at center
        point3f center(0.0f, 0.0f, -50.0f);
        auto transformed = m * center;
        CHECK(transformed.x == doctest::Approx(0.0f));
        CHECK(transformed.y == doctest::Approx(0.0f));
        
        // Edge points
        point3f left(-10.0f, 0.0f, -50.0f);
        transformed = m * left;
        CHECK(transformed.x == doctest::Approx(-1.0f));
        
        point3f right(10.0f, 0.0f, -50.0f);
        transformed = m * right;
        CHECK(transformed.x == doctest::Approx(1.0f));
    }
}