#include <doctest/doctest.h>
#include <euler/coordinates/projective3.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <type_traits>
#include <limits>

using namespace euler;

TEST_CASE("projective3 type traits") {
    SUBCASE("is standard layout") {
        CHECK(std::is_standard_layout_v<projective3<float>>);
        CHECK(std::is_standard_layout_v<projective3<double>>);
    }
    
    SUBCASE("is trivially copyable") {
        CHECK(std::is_trivially_copyable_v<projective3<float>>);
        CHECK(std::is_trivially_copyable_v<projective3<double>>);
    }
    
    SUBCASE("size and alignment") {
        CHECK(sizeof(projective3<float>) == 4 * sizeof(float));
        CHECK(sizeof(projective3<double>) == 4 * sizeof(double));
    }
}

TEST_CASE("projective3 constructors") {
    SUBCASE("default constructor") {
        projective3<float> p;
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
        CHECK(p.z == 0.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("value constructor") {
        projective3<float> p(2.0f, 3.0f, 4.0f, 0.5f);
        CHECK(p.x == 2.0f);
        CHECK(p.y == 3.0f);
        CHECK(p.z == 4.0f);
        CHECK(p.w == 0.5f);
    }
    
    SUBCASE("from point3 (implicit)") {
        point3f pt(4.0f, 5.0f, 6.0f);
        projective3<float> p = pt;  // Implicit conversion
        CHECK(p.x == 4.0f);
        CHECK(p.y == 5.0f);
        CHECK(p.z == 6.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("type conversion") {
        projective3<double> pd(3.5, 4.7, 5.9, 2.0);
        projective3<float> pf(pd);
        CHECK(pf.x == doctest::Approx(3.5f));
        CHECK(pf.y == doctest::Approx(4.7f));
        CHECK(pf.z == doctest::Approx(5.9f));
        CHECK(pf.w == doctest::Approx(2.0f));
    }
}

TEST_CASE("projective3 point conversion") {
    SUBCASE("normal point (w != 0)") {
        projective3<float> p(6.0f, 8.0f, 10.0f, 2.0f);
        point3f pt = p.point();
        CHECK(pt.x == doctest::Approx(3.0f));
        CHECK(pt.y == doctest::Approx(4.0f));
        CHECK(pt.z == doctest::Approx(5.0f));
    }
    
    SUBCASE("point at w = 1") {
        projective3<float> p(5.0f, 7.0f, 9.0f, 1.0f);
        point3f pt = p.point();
        CHECK(pt.x == 5.0f);
        CHECK(pt.y == 7.0f);
        CHECK(pt.z == 9.0f);
    }
    
    SUBCASE("point at infinity (w = 0)") {
        projective3<float> p(1.0f, 2.0f, 3.0f, 0.0f);
        point3f pt = p.point();
        CHECK(std::isinf(pt.x));
        CHECK(std::isinf(pt.y));
        CHECK(std::isinf(pt.z));
    }
    
    SUBCASE("explicit operator") {
        projective3<float> p(10.0f, 15.0f, 20.0f, 5.0f);
        point3f pt = static_cast<point3f>(p);
        CHECK(pt.x == doctest::Approx(2.0f));
        CHECK(pt.y == doctest::Approx(3.0f));
        CHECK(pt.z == doctest::Approx(4.0f));
    }
}

TEST_CASE("projective3 is_finite") {
    SUBCASE("finite point") {
        projective3<float> p(3.0f, 4.0f, 5.0f, 1.0f);
        CHECK(!p.is_infinite());
    }
    
    SUBCASE("point at infinity") {
        projective3<float> p(1.0f, 0.0f, 0.0f, 0.0f);
        CHECK(p.is_infinite());
    }
    
    SUBCASE("very small w") {
        projective3<float> p(1.0f, 2.0f, 3.0f, 1e-10f);
        CHECK(!p.is_infinite());  // Still finite, just very large when converted
    }
}

TEST_CASE("projective3 normalize") {
    SUBCASE("normal normalization") {
        projective3<float> p(6.0f, 8.0f, 10.0f, 2.0f);
        p = p.normalized();
        CHECK(p.x == doctest::Approx(3.0f));
        CHECK(p.y == doctest::Approx(4.0f));
        CHECK(p.z == doctest::Approx(5.0f));
        CHECK(p.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("already normalized") {
        projective3<float> p(3.0f, 4.0f, 5.0f, 1.0f);
        p = p.normalized();
        CHECK(p.x == 3.0f);
        CHECK(p.y == 4.0f);
        CHECK(p.z == 5.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("negative w") {
        projective3<float> p(6.0f, 8.0f, 10.0f, -2.0f);
        p = p.normalized();
        CHECK(p.x == doctest::Approx(-3.0f));
        CHECK(p.y == doctest::Approx(-4.0f));
        CHECK(p.z == doctest::Approx(-5.0f));
        CHECK(p.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("cannot normalize infinity") {
        projective3<float> p(1.0f, 2.0f, 3.0f, 0.0f);
        p = p.normalized();
        // Should remain unchanged
        CHECK(p.x == 1.0f);
        CHECK(p.y == 2.0f);
        CHECK(p.z == 3.0f);
        CHECK(p.w == 0.0f);
    }
}

TEST_CASE("projective3 vector conversion") {
    SUBCASE("to vector") {
        projective3<float> p(3.0f, 4.0f, 5.0f, 2.0f);
        auto v = p.vec();
        CHECK(v[0] == 3.0f);
        CHECK(v[1] == 4.0f);
        CHECK(v[2] == 5.0f);
        CHECK(v[3] == 2.0f);
    }
    
    SUBCASE("const to vector") {
        const projective3<float> p(5.0f, 6.0f, 7.0f, 1.0f);
        auto v = p.vec();
        CHECK(v[0] == 5.0f);
        CHECK(v[1] == 6.0f);
        CHECK(v[2] == 7.0f);
        CHECK(v[3] == 1.0f);
    }
}

TEST_CASE("projective3 element access") {
    projective3<float> p(2.0f, 3.0f, 4.0f, 0.5f);
    
    SUBCASE("array subscript access") {
        CHECK(p[0] == 2.0f);
        CHECK(p[1] == 3.0f);
        CHECK(p[2] == 4.0f);
        CHECK(p[3] == 0.5f);
        
        p[0] = 5.0f;
        p[1] = 6.0f;
        p[2] = 7.0f;
        p[3] = 1.0f;
        
        CHECK(p.x == 5.0f);
        CHECK(p.y == 6.0f);
        CHECK(p.z == 7.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("const array access") {
        const projective3<float> cp(7.0f, 8.0f, 9.0f, 2.0f);
        CHECK(cp[0] == 7.0f);
        CHECK(cp[1] == 8.0f);
        CHECK(cp[2] == 9.0f);
        CHECK(cp[3] == 2.0f);
    }
}

TEST_CASE("projective3 round trip conversion") {
    SUBCASE("point to projective and back") {
        point3f p1(3.5f, 4.7f, 5.9f);
        projective3<float> proj = p1;
        point3f p2 = proj.point();
        
        CHECK(p2.x == doctest::Approx(p1.x));
        CHECK(p2.y == doctest::Approx(p1.y));
        CHECK(p2.z == doctest::Approx(p1.z));
    }
    
    SUBCASE("multiple conversions") {
        point3f p1(1.0f, 2.0f, 3.0f);
        projective3<float> proj1 = p1;
        projective3<float> proj2(proj1.x * 3, proj1.y * 3, proj1.z * 3, proj1.w * 3);
        point3f p2 = proj2.point();
        
        CHECK(p2.x == doctest::Approx(p1.x));
        CHECK(p2.y == doctest::Approx(p1.y));
        CHECK(p2.z == doctest::Approx(p1.z));
    }
}

TEST_CASE("projective3 with matrix transformations") {
    SUBCASE("identity transformation") {
        matrix<float, 4, 4> m = matrix<float, 4, 4>::identity();
        projective3<float> p(3.0f, 4.0f, 5.0f, 1.0f);
        vector<float, 4> v = p.vec();
        auto v_result = m * v;
        projective3<float> result(v_result[0], v_result[1], v_result[2], v_result[3]);
        
        CHECK(result.x == p.x);
        CHECK(result.y == p.y);
        CHECK(result.z == p.z);
        CHECK(result.w == p.w);
    }
    
    SUBCASE("translation transformation") {
        matrix<float, 4, 4> m = {
            {1, 0, 0, 5},
            {0, 1, 0, 3},
            {0, 0, 1, 2},
            {0, 0, 0, 1}
        };
        projective3<float> p(2.0f, 4.0f, 6.0f, 1.0f);
        vector<float, 4> v = p.vec();
        auto v_result = m * v;
        projective3<float> result(v_result[0], v_result[1], v_result[2], v_result[3]);
        
        CHECK(result.x == doctest::Approx(7.0f));
        CHECK(result.y == doctest::Approx(7.0f));
        CHECK(result.z == doctest::Approx(8.0f));
        CHECK(result.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("scale transformation") {
        matrix<float, 4, 4> m = {
            {2, 0, 0, 0},
            {0, 3, 0, 0},
            {0, 0, 4, 0},
            {0, 0, 0, 1}
        };
        projective3<float> p(3.0f, 2.0f, 1.0f, 1.0f);
        vector<float, 4> v = p.vec();
        auto v_result = m * v;
        projective3<float> result(v_result[0], v_result[1], v_result[2], v_result[3]);
        
        CHECK(result.x == doctest::Approx(6.0f));
        CHECK(result.y == doctest::Approx(6.0f));
        CHECK(result.z == doctest::Approx(4.0f));
        CHECK(result.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("perspective transformation") {
        // Simple perspective that affects w
        matrix<float, 4, 4> m = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0.1f, 0.2f, 0.3f, 1}
        };
        projective3<float> p(10.0f, 5.0f, 2.0f, 1.0f);
        vector<float, 4> v = p.vec();
        auto v_result = m * v;
        projective3<float> result(v_result[0], v_result[1], v_result[2], v_result[3]);
        
        CHECK(result.x == doctest::Approx(10.0f));
        CHECK(result.y == doctest::Approx(5.0f));
        CHECK(result.z == doctest::Approx(2.0f));
        CHECK(result.w == doctest::Approx(3.6f));  // 0.1*10 + 0.2*5 + 0.3*2 + 1*1
        
        // When converted to point3
        point3f pt = result.point();
        CHECK(pt.x == doctest::Approx(10.0f / 3.6f));
        CHECK(pt.y == doctest::Approx(5.0f / 3.6f));
        CHECK(pt.z == doctest::Approx(2.0f / 3.6f));
    }
}

TEST_CASE("projective3 special cases") {
    SUBCASE("very large coordinates") {
        projective3<float> p(1e20f, 2e20f, 3e20f, 1.0f);
        CHECK(!p.is_infinite());
        point3f pt = p.point();
        CHECK(pt.x == doctest::Approx(1e20f));
        CHECK(pt.y == doctest::Approx(2e20f));
        CHECK(pt.z == doctest::Approx(3e20f));
    }
    
    SUBCASE("very small w") {
        projective3<float> p(1.0f, 2.0f, 3.0f, 1e-20f);
        CHECK(!p.is_infinite());
        point3f pt = p.point();
        CHECK(pt.x == doctest::Approx(1e20f));
        CHECK(pt.y == doctest::Approx(2e20f));
        CHECK(pt.z == doctest::Approx(3e20f));
    }
    
    SUBCASE("negative coordinates") {
        projective3<float> p(-6.0f, -8.0f, -10.0f, 2.0f);
        point3f pt = p.point();
        CHECK(pt.x == doctest::Approx(-3.0f));
        CHECK(pt.y == doctest::Approx(-4.0f));
        CHECK(pt.z == doctest::Approx(-5.0f));
    }
    
    SUBCASE("mixed signs") {
        projective3<float> p(6.0f, -8.0f, 10.0f, -2.0f);
        point3f pt = p.point();
        CHECK(pt.x == doctest::Approx(-3.0f));
        CHECK(pt.y == doctest::Approx(4.0f));
        CHECK(pt.z == doctest::Approx(-5.0f));
    }
}

TEST_CASE("projective3 from 2D projective") {
    SUBCASE("2D to 3D conversion") {
        projective2<float> p2(3.0f, 4.0f, 2.0f);
        projective3<float> p3(p2.x, p2.y, 0.0f, p2.w);
        
        CHECK(p3.x == 3.0f);
        CHECK(p3.y == 4.0f);
        CHECK(p3.z == 0.0f);
        CHECK(p3.w == 2.0f);
        
        // Check point conversion matches
        point2f pt2 = p2.point();
        point3f pt3 = p3.point();
        CHECK(pt3.x == doctest::Approx(pt2.x));
        CHECK(pt3.y == doctest::Approx(pt2.y));
        CHECK(pt3.z == 0.0f);
    }
}