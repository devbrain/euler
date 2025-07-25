#include <doctest/doctest.h>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/point_ops.hh>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <type_traits>
#include <sstream>

using namespace euler;

TEST_CASE("point2 type traits") {
    SUBCASE("is standard layout") {
        CHECK(std::is_standard_layout_v<point2f>);
        CHECK(std::is_standard_layout_v<point2d>);
        CHECK(std::is_standard_layout_v<point2i>);
    }
    
    SUBCASE("is trivially copyable") {
        CHECK(std::is_trivially_copyable_v<point2f>);
        CHECK(std::is_trivially_copyable_v<point2d>);
        CHECK(std::is_trivially_copyable_v<point2i>);
    }
    
    SUBCASE("size and alignment") {
        CHECK(sizeof(point2f) == 2 * sizeof(float));
        CHECK(sizeof(point2d) == 2 * sizeof(double));
        CHECK(sizeof(point2i) == 2 * sizeof(int));
        
        CHECK(alignof(point2f) == alignof(float));
        CHECK(alignof(point2d) == alignof(double));
        CHECK(alignof(point2i) == alignof(int));
    }
}

TEST_CASE("point2 constructors") {
    SUBCASE("default constructor") {
        point2f p;
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
    }
    
    SUBCASE("value constructor") {
        point2f p(3.0f, 4.0f);
        CHECK(p.x == 3.0f);
        CHECK(p.y == 4.0f);
    }
    
    SUBCASE("copy constructor") {
        point2f p1(3.0f, 4.0f);
        point2f p2(p1);
        CHECK(p2.x == 3.0f);
        CHECK(p2.y == 4.0f);
    }
    
    SUBCASE("type conversion") {
        point2d pd(3.5, 4.7);
        point2f pf(pd);
        CHECK(pf.x == doctest::Approx(3.5f));
        CHECK(pf.y == doctest::Approx(4.7f));
        
        point2i pi(pd);
        CHECK(pi.x == 3);
        CHECK(pi.y == 4);
    }
    
    SUBCASE("from vector") {
        vector<float, 2> v(3.0f, 4.0f);
        point2f p(v);
        CHECK(p.x == 3.0f);
        CHECK(p.y == 4.0f);
    }
}

TEST_CASE("point2 named constructors") {
    SUBCASE("zero") {
        auto p = point2f::zero();
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
    }
    
    SUBCASE("polar with degrees") {
        auto p = point2f::polar(5.0f, degree<float>(0));
        CHECK(p.x == doctest::Approx(5.0f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        
        p = point2f::polar(5.0f, degree<float>(90));
        CHECK(p.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.y == doctest::Approx(5.0f));
        
        p = point2f::polar(5.0f, degree<float>(45));
        CHECK(p.x == doctest::Approx(3.5355339f));
        CHECK(p.y == doctest::Approx(3.5355339f));
    }
    
    SUBCASE("polar with radians") {
        auto p = point2f::polar(10.0f, radian<float>(0));
        CHECK(p.x == doctest::Approx(10.0f));
        CHECK(p.y == doctest::Approx(0.0f).epsilon(1e-6f));
        
        p = point2f::polar(10.0f, radian<float>(pi / 2));
        CHECK(p.x == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(p.y == doctest::Approx(10.0f));
    }
}

TEST_CASE("point2 element access") {
    SUBCASE("member access") {
        point2f p(3.0f, 4.0f);
        CHECK(p.x == 3.0f);
        CHECK(p.y == 4.0f);
        
        p.x = 5.0f;
        p.y = 6.0f;
        CHECK(p.x == 5.0f);
        CHECK(p.y == 6.0f);
    }
    
    SUBCASE("array access") {
        point2f p(3.0f, 4.0f);
        CHECK(p[0] == 3.0f);
        CHECK(p[1] == 4.0f);
        
        p[0] = 5.0f;
        p[1] = 6.0f;
        CHECK(p[0] == 5.0f);
        CHECK(p[1] == 6.0f);
    }
    
    SUBCASE("color aliases") {
        point2f p(0.5f, 0.7f);
        CHECK(p.r() == 0.5f);
        CHECK(p.g() == 0.7f);
        
        p.r() = 0.2f;
        p.g() = 0.9f;
        CHECK(p.x == 0.2f);
        CHECK(p.y == 0.9f);
    }
}

TEST_CASE("point2 swizzling") {
    point2f p(3.0f, 4.0f);
    
    SUBCASE("xy swizzle") {
        auto q = p.xy();
        CHECK(q.x == 3.0f);
        CHECK(q.y == 4.0f);
    }
    
    SUBCASE("yx swizzle") {
        auto q = p.yx();
        CHECK(q.x == 4.0f);
        CHECK(q.y == 3.0f);
    }
    
    SUBCASE("xx swizzle") {
        auto q = p.xx();
        CHECK(q.x == 3.0f);
        CHECK(q.y == 3.0f);
    }
    
    SUBCASE("yy swizzle") {
        auto q = p.yy();
        CHECK(q.x == 4.0f);
        CHECK(q.y == 4.0f);
    }
}

TEST_CASE("point2 vector conversion") {
    SUBCASE("to vector") {
        point2f p(3.0f, 4.0f);
        auto v = p.vec();
        CHECK(v[0] == 3.0f);
        CHECK(v[1] == 4.0f);
    }
    
    SUBCASE("from vector") {
        vector<float, 2> v(5.0f, 6.0f);
        point2f p(v);
        CHECK(p.x == 5.0f);
        CHECK(p.y == 6.0f);
    }
}

TEST_CASE("point2 projective conversion") {
    SUBCASE("implicit to projective") {
        point2f p(3.0f, 4.0f);
        projective2<float> proj = p;
        CHECK(proj.x == 3.0f);
        CHECK(proj.y == 4.0f);
        CHECK(proj.w == 1.0f);
    }
    
    SUBCASE("round trip conversion") {
        point2f p1(3.0f, 4.0f);
        projective2<float> proj = p1;
        point2f p2 = proj.point();
        CHECK(p2.x == 3.0f);
        CHECK(p2.y == 4.0f);
    }
}

TEST_CASE("point2 arithmetic operations") {
    point2f p1(3.0f, 4.0f);
    point2f p2(1.0f, 2.0f);
    vector<float, 2> v(5.0f, 6.0f);
    
    SUBCASE("point + vector") {
        auto p3 = p1 + v;
        CHECK(p3.x == 8.0f);
        CHECK(p3.y == 10.0f);
        
        // Commutative
        auto p4 = v + p1;
        CHECK(p4.x == 8.0f);
        CHECK(p4.y == 10.0f);
    }
    
    SUBCASE("point - vector") {
        auto p3 = p1 - v;
        CHECK(p3.x == -2.0f);
        CHECK(p3.y == -2.0f);
    }
    
    SUBCASE("point - point") {
        auto v_result = p1 - p2;
        CHECK(v_result[0] == 2.0f);
        CHECK(v_result[1] == 2.0f);
    }
    
    SUBCASE("scalar * point") {
        auto p3 = 2.0f * p1;
        CHECK(p3.x == 6.0f);
        CHECK(p3.y == 8.0f);
        
        // Commutative
        auto p4 = p1 * 2.0f;
        CHECK(p4.x == 6.0f);
        CHECK(p4.y == 8.0f);
    }
    
    SUBCASE("point / scalar") {
        auto p3 = p1 / 2.0f;
        CHECK(p3.x == 1.5f);
        CHECK(p3.y == 2.0f);
    }
}

TEST_CASE("point2 comparison") {
    SUBCASE("equality") {
        point2f p1(3.0f, 4.0f);
        point2f p2(3.0f, 4.0f);
        point2f p3(3.0f, 5.0f);
        
        CHECK(p1 == p2);
        CHECK(!(p1 == p3));
        CHECK(p1 != p3);
        CHECK(!(p1 != p2));
    }
    
    SUBCASE("approximate equality") {
        point2f p1(1.0f, 2.0f);
        point2f p2(1.0f + 1e-7f, 2.0f - 1e-7f);
        
        CHECK(approx_equal(p1, p2));
        CHECK(approx_equal(p1, p2, 1e-6f));
        CHECK(!approx_equal(p1, p2, 1e-8f));
    }
}

TEST_CASE("point2 geometric operations") {
    SUBCASE("distance") {
        point2f p1(0.0f, 0.0f);
        point2f p2(3.0f, 4.0f);
        
        CHECK(distance(p1, p2) == doctest::Approx(5.0f));
        CHECK(distance_squared(p1, p2) == doctest::Approx(25.0f));
    }
    
    SUBCASE("midpoint") {
        point2f p1(2.0f, 4.0f);
        point2f p2(6.0f, 8.0f);
        
        auto mid = midpoint(p1, p2);
        CHECK(mid.x == 4.0f);
        CHECK(mid.y == 6.0f);
    }
    
    SUBCASE("lerp") {
        point2f p1(0.0f, 0.0f);
        point2f p2(10.0f, 20.0f);
        
        auto p = lerp(p1, p2, 0.0f);
        CHECK(p.x == 0.0f);
        CHECK(p.y == 0.0f);
        
        p = lerp(p1, p2, 1.0f);
        CHECK(p.x == 10.0f);
        CHECK(p.y == 20.0f);
        
        p = lerp(p1, p2, 0.5f);
        CHECK(p.x == 5.0f);
        CHECK(p.y == 10.0f);
        
        p = lerp(p1, p2, 0.25f);
        CHECK(p.x == 2.5f);
        CHECK(p.y == 5.0f);
    }
    
    SUBCASE("barycentric") {
        point2f a(0.0f, 0.0f);
        point2f b(10.0f, 0.0f);
        point2f c(0.0f, 10.0f);
        
        // Corners
        auto p = barycentric(a, b, c, 1.0f, 0.0f, 0.0f);
        CHECK(p == a);
        
        p = barycentric(a, b, c, 0.0f, 1.0f, 0.0f);
        CHECK(p == b);
        
        p = barycentric(a, b, c, 0.0f, 0.0f, 1.0f);
        CHECK(p == c);
        
        // Center
        p = barycentric(a, b, c, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
        CHECK(p.x == doctest::Approx(10.0f/3.0f));
        CHECK(p.y == doctest::Approx(10.0f/3.0f));
    }
}

TEST_CASE("point2 rounding operations") {
    SUBCASE("round") {
        point2f pf(3.7f, 4.2f);
        auto pi = round(pf);
        CHECK(pi.x == 4);
        CHECK(pi.y == 4);
        
        pf = point2f(3.5f, 4.5f);
        pi = round(pf);
        CHECK(pi.x == 4);  // Round half to even
        CHECK(pi.y == 5);
        
        pf = point2f(-3.7f, -4.2f);
        pi = round(pf);
        CHECK(pi.x == -4);
        CHECK(pi.y == -4);
    }
    
    SUBCASE("floor") {
        point2f pf(3.7f, 4.2f);
        auto pi = floor(pf);
        CHECK(pi.x == 3);
        CHECK(pi.y == 4);
        
        pf = point2f(-3.7f, -4.2f);
        pi = floor(pf);
        CHECK(pi.x == -4);
        CHECK(pi.y == -5);
    }
    
    SUBCASE("ceil") {
        point2f pf(3.7f, 4.2f);
        auto pi = ceil(pf);
        CHECK(pi.x == 4);
        CHECK(pi.y == 5);
        
        pf = point2f(-3.7f, -4.2f);
        pi = ceil(pf);
        CHECK(pi.x == -3);
        CHECK(pi.y == -4);
    }
}