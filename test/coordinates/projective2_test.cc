#include <doctest/doctest.h>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <type_traits>
#include <limits>

using namespace euler;

TEST_CASE("projective2 type traits") {
    SUBCASE("is standard layout") {
        CHECK(std::is_standard_layout_v<projective2<float>>);
        CHECK(std::is_standard_layout_v<projective2<double>>);
    }
    
    SUBCASE("is trivially copyable") {
        CHECK(std::is_trivially_copyable_v<projective2<float>>);
        CHECK(std::is_trivially_copyable_v<projective2<double>>);
    }
    
    SUBCASE("size and alignment") {
        CHECK(sizeof(projective2<float>) == 3 * sizeof(float));
        CHECK(sizeof(projective2<double>) == 3 * sizeof(double));
    }
}

TEST_CASE("projective2 constructors") {
    SUBCASE("default constructor") {
        projective2<float> p;
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("value constructor") {
        projective2<float> p(2.0f, 3.0f, 0.5f);
        CHECK(p.x == 2.0f);
        CHECK(p.y == 3.0f);
        CHECK(p.w == 0.5f);
    }
    
    SUBCASE("from point2 (implicit)") {
        point2f pt(4.0f, 5.0f);
        projective2<float> p = pt;  // Implicit conversion
        CHECK(p.x == 4.0f);
        CHECK(p.y == 5.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("type conversion") {
        projective2<double> pd(3.5, 4.7, 2.0);
        projective2<float> pf(pd);
        CHECK(pf.x == doctest::Approx(3.5f));
        CHECK(pf.y == doctest::Approx(4.7f));
        CHECK(pf.w == doctest::Approx(2.0f));
    }
}

TEST_CASE("projective2 point conversion") {
    SUBCASE("normal point (w != 0)") {
        projective2<float> p(6.0f, 8.0f, 2.0f);
        point2f pt = p.point();
        CHECK(pt.x == doctest::Approx(3.0f));
        CHECK(pt.y == doctest::Approx(4.0f));
    }
    
    SUBCASE("point at w = 1") {
        projective2<float> p(5.0f, 7.0f, 1.0f);
        point2f pt = p.point();
        CHECK(pt.x == 5.0f);
        CHECK(pt.y == 7.0f);
    }
    
    SUBCASE("point at infinity (w = 0)") {
        projective2<float> p(1.0f, 2.0f, 0.0f);
        point2f pt = p.point();
        CHECK(std::isinf(pt.x));
        CHECK(std::isinf(pt.y));
    }
    
    SUBCASE("explicit operator") {
        projective2<float> p(10.0f, 15.0f, 5.0f);
        point2f pt = static_cast<point2f>(p);
        CHECK(pt.x == doctest::Approx(2.0f));
        CHECK(pt.y == doctest::Approx(3.0f));
    }
}

TEST_CASE("projective2 is_finite") {
    SUBCASE("finite point") {
        projective2<float> p(3.0f, 4.0f, 1.0f);
        CHECK(!p.is_infinite());
    }
    
    SUBCASE("point at infinity") {
        projective2<float> p(1.0f, 0.0f, 0.0f);
        CHECK(p.is_infinite());
    }
    
    SUBCASE("very small w") {
        projective2<float> p(1.0f, 2.0f, 1e-10f);
        CHECK(!p.is_infinite());  // Still finite, just very large when converted
    }
}

TEST_CASE("projective2 normalize") {
    SUBCASE("normal normalization") {
        projective2<float> p(6.0f, 8.0f, 2.0f);
        p = p.normalized();
        CHECK(p.x == doctest::Approx(3.0f));
        CHECK(p.y == doctest::Approx(4.0f));
        CHECK(p.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("already normalized") {
        projective2<float> p(3.0f, 4.0f, 1.0f);
        p = p.normalized();
        CHECK(p.x == 3.0f);
        CHECK(p.y == 4.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("negative w") {
        projective2<float> p(6.0f, 8.0f, -2.0f);
        p = p.normalized();
        CHECK(p.x == doctest::Approx(-3.0f));
        CHECK(p.y == doctest::Approx(-4.0f));
        CHECK(p.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("cannot normalize infinity") {
        projective2<float> p(1.0f, 2.0f, 0.0f);
        p = p.normalized();
        // Should remain unchanged
        CHECK(p.x == 1.0f);
        CHECK(p.y == 2.0f);
        CHECK(p.w == 0.0f);
    }
}

TEST_CASE("projective2 vector conversion") {
    SUBCASE("to vector") {
        projective2<float> p(3.0f, 4.0f, 2.0f);
        auto v = p.vec();
        CHECK(v[0] == 3.0f);
        CHECK(v[1] == 4.0f);
        CHECK(v[2] == 2.0f);
    }
    
    SUBCASE("const to vector") {
        const projective2<float> p(5.0f, 6.0f, 1.0f);
        auto v = p.vec();
        CHECK(v[0] == 5.0f);
        CHECK(v[1] == 6.0f);
        CHECK(v[2] == 1.0f);
    }
}

TEST_CASE("projective2 element access") {
    projective2<float> p(2.0f, 3.0f, 0.5f);
    
    SUBCASE("array subscript access") {
        CHECK(p[0] == 2.0f);
        CHECK(p[1] == 3.0f);
        CHECK(p[2] == 0.5f);
        
        p[0] = 4.0f;
        p[1] = 5.0f;
        p[2] = 1.0f;
        
        CHECK(p.x == 4.0f);
        CHECK(p.y == 5.0f);
        CHECK(p.w == 1.0f);
    }
    
    SUBCASE("const array access") {
        const projective2<float> cp(7.0f, 8.0f, 2.0f);
        CHECK(cp[0] == 7.0f);
        CHECK(cp[1] == 8.0f);
        CHECK(cp[2] == 2.0f);
    }
}

TEST_CASE("projective2 round trip conversion") {
    SUBCASE("point to projective and back") {
        point2f p1(3.5f, 4.7f);
        projective2<float> proj = p1;
        point2f p2 = proj.point();
        
        CHECK(p2.x == doctest::Approx(p1.x));
        CHECK(p2.y == doctest::Approx(p1.y));
    }
    
    SUBCASE("multiple conversions") {
        point2f p1(1.0f, 2.0f);
        projective2<float> proj1 = p1;
        projective2<float> proj2(proj1.x * 3, proj1.y * 3, proj1.w * 3);
        point2f p2 = proj2.point();
        
        CHECK(p2.x == doctest::Approx(p1.x));
        CHECK(p2.y == doctest::Approx(p1.y));
    }
}

TEST_CASE("projective2 with matrix transformations") {
    SUBCASE("identity transformation") {
        matrix<float, 3, 3> m = matrix<float, 3, 3>::identity();
        projective2<float> p(3.0f, 4.0f, 1.0f);
        vector<float, 3> v = p.vec();
        auto v_result = m * v;
        projective2<float> result(v_result[0], v_result[1], v_result[2]);
        
        CHECK(result.x == p.x);
        CHECK(result.y == p.y);
        CHECK(result.w == p.w);
    }
    
    SUBCASE("translation transformation") {
        matrix<float, 3, 3> m = {
            {1, 0, 5},
            {0, 1, 3},
            {0, 0, 1}
        };
        projective2<float> p(2.0f, 4.0f, 1.0f);
        vector<float, 3> v = p.vec();
        auto v_result = m * v;
        projective2<float> result(v_result[0], v_result[1], v_result[2]);
        
        CHECK(result.x == doctest::Approx(7.0f));
        CHECK(result.y == doctest::Approx(7.0f));
        CHECK(result.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("scale transformation") {
        matrix<float, 3, 3> m = {
            {2, 0, 0},
            {0, 3, 0},
            {0, 0, 1}
        };
        projective2<float> p(3.0f, 2.0f, 1.0f);
        vector<float, 3> v = p.vec();
        auto v_result = m * v;
        projective2<float> result(v_result[0], v_result[1], v_result[2]);
        
        CHECK(result.x == doctest::Approx(6.0f));
        CHECK(result.y == doctest::Approx(6.0f));
        CHECK(result.w == doctest::Approx(1.0f));
    }
    
    SUBCASE("perspective transformation") {
        // Simple perspective that affects w
        matrix<float, 3, 3> m = {
            {1, 0, 0},
            {0, 1, 0},
            {0.1f, 0.2f, 1}
        };
        projective2<float> p(10.0f, 5.0f, 1.0f);
        vector<float, 3> v = p.vec();
        auto v_result = m * v;
        projective2<float> result(v_result[0], v_result[1], v_result[2]);
        
        CHECK(result.x == doctest::Approx(10.0f));
        CHECK(result.y == doctest::Approx(5.0f));
        CHECK(result.w == doctest::Approx(3.0f));  // 0.1*10 + 0.2*5 + 1*1
        
        // When converted to point2
        point2f pt = result.point();
        CHECK(pt.x == doctest::Approx(10.0f / 3.0f));
        CHECK(pt.y == doctest::Approx(5.0f / 3.0f));
    }
}

TEST_CASE("projective2 special cases") {
    SUBCASE("very large coordinates") {
        projective2<float> p(1e20f, 2e20f, 1.0f);
        CHECK(!p.is_infinite());
        point2f pt = p.point();
        CHECK(pt.x == doctest::Approx(1e20f));
        CHECK(pt.y == doctest::Approx(2e20f));
    }
    
    SUBCASE("very small w") {
        projective2<float> p(1.0f, 2.0f, 1e-20f);
        CHECK(!p.is_infinite());
        point2f pt = p.point();
        CHECK(pt.x == doctest::Approx(1e20f));
        CHECK(pt.y == doctest::Approx(2e20f));
    }
    
    SUBCASE("negative coordinates") {
        projective2<float> p(-6.0f, -8.0f, 2.0f);
        point2f pt = p.point();
        CHECK(pt.x == doctest::Approx(-3.0f));
        CHECK(pt.y == doctest::Approx(-4.0f));
    }
    
    SUBCASE("mixed signs") {
        projective2<float> p(6.0f, -8.0f, -2.0f);
        point2f pt = p.point();
        CHECK(pt.x == doctest::Approx(-3.0f));
        CHECK(pt.y == doctest::Approx(4.0f));
    }
}