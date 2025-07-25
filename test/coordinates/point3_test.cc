#include <doctest/doctest.h>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/projective3.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <type_traits>

using namespace euler;

TEST_CASE("point3 type traits") {
    SUBCASE("is standard layout") {
        CHECK(std::is_standard_layout_v<point3f>);
        CHECK(std::is_standard_layout_v<point3d>);
        CHECK(std::is_standard_layout_v<point3i>);
    }
    
    SUBCASE("is trivially copyable") {
        CHECK(std::is_trivially_copyable_v<point3f>);
        CHECK(std::is_trivially_copyable_v<point3d>);
        CHECK(std::is_trivially_copyable_v<point3i>);
    }
    
    SUBCASE("size and alignment") {
        CHECK(sizeof(point3f) == 3 * sizeof(float));
        CHECK(sizeof(point3d) == 3 * sizeof(double));
        CHECK(sizeof(point3i) == 3 * sizeof(int));
    }
}

TEST_CASE("point3 constructors") {
    SUBCASE("default constructor") {
        point3f p;
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
        CHECK(p.z == 0.0f);
    }
    
    SUBCASE("value constructor") {
        point3f p(1.0f, 2.0f, 3.0f);
        CHECK(p.x == 1.0f);
        CHECK(p.y == 2.0f);
        CHECK(p.z == 3.0f);
    }
    
    SUBCASE("from 2D point") {
        point2f p2(3.0f, 4.0f);
        point3f p3(p2);
        CHECK(p3.x == 3.0f);
        CHECK(p3.y == 4.0f);
        CHECK(p3.z == 0.0f);
        
        point3f p3_with_z(p2, 5.0f);
        CHECK(p3_with_z.x == 3.0f);
        CHECK(p3_with_z.y == 4.0f);
        CHECK(p3_with_z.z == 5.0f);
    }
    
    SUBCASE("type conversion") {
        point3d pd(3.5, 4.7, 5.9);
        point3f pf(pd);
        CHECK(pf.x == doctest::Approx(3.5f));
        CHECK(pf.y == doctest::Approx(4.7f));
        CHECK(pf.z == doctest::Approx(5.9f));
        
        point3i pi(pd);
        CHECK(pi.x == 3);
        CHECK(pi.y == 4);
        CHECK(pi.z == 5);
    }
}

TEST_CASE("point3 named constructors") {
    SUBCASE("zero") {
        auto p = point3f::zero();
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
        CHECK(p.z == 0.0f);
    }
    
    SUBCASE("spherical with degrees") {
        // Point on positive X axis
        auto p = point3f::spherical(10.0f, degree<float>(0), degree<float>(90));
        CHECK(p.x == doctest::Approx(10.0f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.z == doctest::Approx(0.0f).epsilon(1e-6f));
        
        // Point on positive Y axis
        p = point3f::spherical(10.0f, degree<float>(90), degree<float>(90));
        CHECK(p.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.y == doctest::Approx(10.0f));
        CHECK(p.z == doctest::Approx(0.0f).epsilon(1e-6f));
        
        // Point on positive Z axis
        p = point3f::spherical(10.0f, degree<float>(0), degree<float>(0));
        CHECK(p.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.z == doctest::Approx(10.0f));
    }
    
    SUBCASE("spherical with radians") {
        auto p = point3f::spherical(5.0f, radian<float>(0), radian<float>(pi/2));
        CHECK(p.x == doctest::Approx(5.0f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.z == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    
    SUBCASE("cylindrical with degrees") {
        // Point on positive X axis at height 3
        auto p = point3f::cylindrical(5.0f, degree<float>(0), 3.0f);
        CHECK(p.x == doctest::Approx(5.0f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.z == 3.0f);
        
        // Point on positive Y axis at height -2
        p = point3f::cylindrical(7.0f, degree<float>(90), -2.0f);
        CHECK(p.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.y == doctest::Approx(7.0f));
        CHECK(p.z == -2.0f);
    }
}

TEST_CASE("point3 element access") {
    SUBCASE("member access") {
        point3f p(1.0f, 2.0f, 3.0f);
        CHECK(p.x == 1.0f);
        CHECK(p.y == 2.0f);
        CHECK(p.z == 3.0f);
        
        p.x = 4.0f;
        p.y = 5.0f;
        p.z = 6.0f;
        CHECK(p.x == 4.0f);
        CHECK(p.y == 5.0f);
        CHECK(p.z == 6.0f);
    }
    
    SUBCASE("array access") {
        point3f p(1.0f, 2.0f, 3.0f);
        CHECK(p[0] == 1.0f);
        CHECK(p[1] == 2.0f);
        CHECK(p[2] == 3.0f);
        
        p[0] = 4.0f;
        p[1] = 5.0f;
        p[2] = 6.0f;
        CHECK(p[0] == 4.0f);
        CHECK(p[1] == 5.0f);
        CHECK(p[2] == 6.0f);
    }
    
    SUBCASE("color aliases") {
        point3f p(0.2f, 0.4f, 0.6f);
        CHECK(p.r() == 0.2f);
        CHECK(p.g() == 0.4f);
        CHECK(p.b() == 0.6f);
        
        p.r() = 0.8f;
        p.g() = 0.5f;
        p.b() = 0.3f;
        CHECK(p.x == 0.8f);
        CHECK(p.y == 0.5f);
        CHECK(p.z == 0.3f);
    }
}

TEST_CASE("point3 swizzling") {
    point3f p(1.0f, 2.0f, 3.0f);
    
    SUBCASE("2D projections") {
        auto xy = p.xy();
        CHECK(xy.x == 1.0f);
        CHECK(xy.y == 2.0f);
        
        auto xz = p.xz();
        CHECK(xz.x == 1.0f);
        CHECK(xz.y == 3.0f);
        
        auto yx = p.yx();
        CHECK(yx.x == 2.0f);
        CHECK(yx.y == 1.0f);
        
        auto yz = p.yz();
        CHECK(yz.x == 2.0f);
        CHECK(yz.y == 3.0f);
        
        auto zx = p.zx();
        CHECK(zx.x == 3.0f);
        CHECK(zx.y == 1.0f);
        
        auto zy = p.zy();
        CHECK(zy.x == 3.0f);
        CHECK(zy.y == 2.0f);
    }
    
    SUBCASE("3D swizzles") {
        auto xyz = p.xyz();
        CHECK(xyz.x == 1.0f);
        CHECK(xyz.y == 2.0f);
        CHECK(xyz.z == 3.0f);
        
        auto zyx = p.zyx();
        CHECK(zyx.x == 3.0f);
        CHECK(zyx.y == 2.0f);
        CHECK(zyx.z == 1.0f);
        
        auto xzy = p.xzy();
        CHECK(xzy.x == 1.0f);
        CHECK(xzy.y == 3.0f);
        CHECK(xzy.z == 2.0f);
        
        auto yxz = p.yxz();
        CHECK(yxz.x == 2.0f);
        CHECK(yxz.y == 1.0f);
        CHECK(yxz.z == 3.0f);
        
        auto yzx = p.yzx();
        CHECK(yzx.x == 2.0f);
        CHECK(yzx.y == 3.0f);
        CHECK(yzx.z == 1.0f);
        
        auto zxy = p.zxy();
        CHECK(zxy.x == 3.0f);
        CHECK(zxy.y == 1.0f);
        CHECK(zxy.z == 2.0f);
    }
}

TEST_CASE("point3 arithmetic operations") {
    point3f p1(1.0f, 2.0f, 3.0f);
    point3f p2(4.0f, 5.0f, 6.0f);
    vector<float, 3> v(7.0f, 8.0f, 9.0f);
    
    SUBCASE("point + vector") {
        auto p3 = p1 + v;
        CHECK(p3.x == 8.0f);
        CHECK(p3.y == 10.0f);
        CHECK(p3.z == 12.0f);
        
        // Commutative
        auto p4 = v + p1;
        CHECK(p4.x == 8.0f);
        CHECK(p4.y == 10.0f);
        CHECK(p4.z == 12.0f);
    }
    
    SUBCASE("point - vector") {
        auto p3 = p1 - v;
        CHECK(p3.x == -6.0f);
        CHECK(p3.y == -6.0f);
        CHECK(p3.z == -6.0f);
    }
    
    SUBCASE("point - point") {
        auto v_result = p2 - p1;
        CHECK(v_result[0] == 3.0f);
        CHECK(v_result[1] == 3.0f);
        CHECK(v_result[2] == 3.0f);
    }
    
    SUBCASE("scalar * point") {
        auto p3 = 2.0f * p1;
        CHECK(p3.x == 2.0f);
        CHECK(p3.y == 4.0f);
        CHECK(p3.z == 6.0f);
        
        // Commutative
        auto p4 = p1 * 2.0f;
        CHECK(p4.x == 2.0f);
        CHECK(p4.y == 4.0f);
        CHECK(p4.z == 6.0f);
    }
    
    SUBCASE("point / scalar") {
        auto p3 = point3f(2.0f, 4.0f, 6.0f) / 2.0f;
        CHECK(p3.x == 1.0f);
        CHECK(p3.y == 2.0f);
        CHECK(p3.z == 3.0f);
    }
}

TEST_CASE("point3 comparison") {
    SUBCASE("equality") {
        point3f p1(1.0f, 2.0f, 3.0f);
        point3f p2(1.0f, 2.0f, 3.0f);
        point3f p3(1.0f, 2.0f, 4.0f);
        
        CHECK(p1 == p2);
        CHECK(!(p1 == p3));
        CHECK(p1 != p3);
        CHECK(!(p1 != p2));
    }
    
    SUBCASE("approximate equality") {
        point3f p1(1.0f, 2.0f, 3.0f);
        point3f p2(1.0f + 1e-7f, 2.0f - 1e-7f, 3.0f + 1e-7f);
        
        CHECK(approx_equal(p1, p2));
        CHECK(approx_equal(p1, p2, 1e-6f));
        CHECK(!approx_equal(p1, p2, 1e-8f));
    }
}

TEST_CASE("point3 geometric operations") {
    SUBCASE("distance") {
        point3f p1(0.0f, 0.0f, 0.0f);
        point3f p2(2.0f, 3.0f, 6.0f);
        
        CHECK(distance(p1, p2) == doctest::Approx(7.0f));
        CHECK(distance_squared(p1, p2) == doctest::Approx(49.0f));
    }
    
    SUBCASE("midpoint") {
        point3f p1(2.0f, 4.0f, 6.0f);
        point3f p2(6.0f, 8.0f, 10.0f);
        
        auto mid = midpoint(p1, p2);
        CHECK(mid.x == 4.0f);
        CHECK(mid.y == 6.0f);
        CHECK(mid.z == 8.0f);
    }
    
    SUBCASE("lerp") {
        point3f p1(0.0f, 0.0f, 0.0f);
        point3f p2(10.0f, 20.0f, 30.0f);
        
        auto p = lerp(p1, p2, 0.0f);
        CHECK(p == p1);
        
        p = lerp(p1, p2, 1.0f);
        CHECK(p == p2);
        
        p = lerp(p1, p2, 0.5f);
        CHECK(p.x == 5.0f);
        CHECK(p.y == 10.0f);
        CHECK(p.z == 15.0f);
    }
    
    SUBCASE("barycentric") {
        point3f a(0.0f, 0.0f, 0.0f);
        point3f b(10.0f, 0.0f, 0.0f);
        point3f c(0.0f, 10.0f, 0.0f);
        
        auto p = barycentric(a, b, c, 1.0f, 0.0f, 0.0f);
        CHECK(p == a);
        
        p = barycentric(a, b, c, 0.0f, 1.0f, 0.0f);
        CHECK(p == b);
        
        p = barycentric(a, b, c, 0.0f, 0.0f, 1.0f);
        CHECK(p == c);
        
        p = barycentric(a, b, c, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
        CHECK(p.x == doctest::Approx(10.0f/3.0f));
        CHECK(p.y == doctest::Approx(10.0f/3.0f));
        CHECK(p.z == 0.0f);
    }
}

TEST_CASE("point3 rounding operations") {
    SUBCASE("round") {
        point3f pf(3.7f, 4.2f, 5.5f);
        auto pi = round(pf);
        CHECK(pi.x == 4);
        CHECK(pi.y == 4);
        CHECK(pi.z == 6);
        
        pf = point3f(-3.7f, -4.2f, -5.5f);
        pi = round(pf);
        CHECK(pi.x == -4);
        CHECK(pi.y == -4);
        CHECK(pi.z == -6);
    }
    
    SUBCASE("floor") {
        point3f pf(3.7f, 4.2f, 5.9f);
        auto pi = floor(pf);
        CHECK(pi.x == 3);
        CHECK(pi.y == 4);
        CHECK(pi.z == 5);
        
        pf = point3f(-3.7f, -4.2f, -5.9f);
        pi = floor(pf);
        CHECK(pi.x == -4);
        CHECK(pi.y == -5);
        CHECK(pi.z == -6);
    }
    
    SUBCASE("ceil") {
        point3f pf(3.1f, 4.2f, 5.9f);
        auto pi = ceil(pf);
        CHECK(pi.x == 4);
        CHECK(pi.y == 5);
        CHECK(pi.z == 6);
        
        pf = point3f(-3.1f, -4.2f, -5.9f);
        pi = ceil(pf);
        CHECK(pi.x == -3);
        CHECK(pi.y == -4);
        CHECK(pi.z == -5);
    }
}