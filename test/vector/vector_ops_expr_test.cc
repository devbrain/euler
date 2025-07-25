#include <doctest/doctest.h>
#include <euler/vector/vector.hh>
#include <euler/vector/vector_ops.hh>
#include <euler/vector/vector_expr.hh>
#include <cmath>

TEST_CASE("euler::vector operations with expressions") {
    using namespace euler;
    
    SUBCASE("dot product with expressions") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        vec3<float> c(0.5f, 1.0f, 1.5f);
        
        // dot with pure expressions
        float d1 = dot(a + b, c);
        float d2 = dot(a, b + c);
        float d3 = dot(a + b, c - a);
        
        // Verify results
        vec3<float> a_plus_b = a + b;
        vec3<float> b_plus_c = b + c;
        vec3<float> c_minus_a = c - a;
        
        CHECK(d1 == doctest::Approx(dot(a_plus_b, c)));
        CHECK(d2 == doctest::Approx(dot(a, b_plus_c)));
        CHECK(d3 == doctest::Approx(dot(a_plus_b, c_minus_a)));
        
        // dot with mixed expression and concrete
        float d4 = dot(2.0f * a, b);
        CHECK(d4 == doctest::Approx(2.0f * dot(a, b)));
        
        // dot with normalized expressions
        float d5 = dot(normalize(a), normalize(b));
        vec3<float> na = normalize(a);
        vec3<float> nb = normalize(b);
        CHECK(d5 == doctest::Approx(dot(na, nb)));
    }
    
    SUBCASE("length operations with expressions") {
        vec3<float> a(3.0f, 4.0f, 0.0f);
        vec3<float> b(1.0f, 0.0f, 0.0f);
        
        // length_squared with expressions
        float len_sq1 = length_squared(a + b);
        float len_sq2 = length_squared(2.0f * a);
        float len_sq3 = length_squared(a - b);
        
        CHECK(len_sq1 == doctest::Approx(16.0f + 16.0f)); // (4,4,0)
        CHECK(len_sq2 == doctest::Approx(4.0f * 25.0f));  // 4 * (9+16)
        CHECK(len_sq3 == doctest::Approx(4.0f + 16.0f));  // (2,4,0)
        
        // length with expressions
        float len1 = length(a + b);
        float len2 = length(normalize(a) * 5.0f);
        
        CHECK(len1 == doctest::Approx(std::sqrt(32.0f)));
        CHECK(len2 == doctest::Approx(5.0f));
    }
    
    SUBCASE("normalize with expressions") {
        vec3<float> a(3.0f, 4.0f, 0.0f);
        vec3<float> b(0.0f, 0.0f, 5.0f);
        
        // normalize expression
        vec3<float> n1 = normalize(a + b);
        vec3<float> n2 = normalize(2.0f * a);
        
        CHECK(length(n1) == doctest::Approx(1.0f));
        CHECK(length(n2) == doctest::Approx(1.0f));
        
        // Check direction is preserved
        vec3<float> a_plus_b = a + b;
        vec3<float> expected_n1 = a_plus_b / length(a_plus_b);
        CHECK(n1.x() == doctest::Approx(expected_n1.x()));
        CHECK(n1.y() == doctest::Approx(expected_n1.y()));
        CHECK(n1.z() == doctest::Approx(expected_n1.z()));
    }
    
    SUBCASE("cross product with expressions") {
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(0.0f, 1.0f, 0.0f);
        vec3<float> c(0.0f, 0.0f, 1.0f);
        
        // cross with expressions
        vec3<float> c1 = cross(a + b, c);
        vec3<float> c2 = cross(a, b + c);
        vec3<float> c3 = cross(normalize(a), normalize(b));
        
        vec3<float> temp1(1.0f, 1.0f, 0.0f);
        vec3<float> expected1 = cross(temp1, c);
        vec3<float> temp2(0.0f, 1.0f, 1.0f);
        vec3<float> expected2 = cross(a, temp2);
        
        CHECK(c1.x() == doctest::Approx(expected1.x()));
        CHECK(c1.y() == doctest::Approx(expected1.y()));
        CHECK(c1.z() == doctest::Approx(expected1.z()));
        
        CHECK(c2.x() == doctest::Approx(expected2.x()));
        CHECK(c2.y() == doctest::Approx(expected2.y()));
        CHECK(c2.z() == doctest::Approx(expected2.z()));
        
        CHECK(c3.z() == doctest::Approx(1.0f)); // i × j = k
    }
    
    SUBCASE("reflect with expressions") {
        vec3<float> incident(1.0f, -1.0f, 0.0f);
        vec3<float> normal(0.0f, 1.0f, 0.0f);
        
        // reflect with expressions
        vec3<float> r1 = reflect(incident, normal);
        vec3<float> temp(0.1f, 0.0f, 0.0f);
        vec3<float> r2 = reflect(incident + temp, normal);
        vec3<float> r3 = reflect(normalize(incident), normalize(normal));
        
        CHECK(r1.x() == doctest::Approx(1.0f));
        CHECK(r1.y() == doctest::Approx(1.0f));
        CHECK(r1.z() == doctest::Approx(0.0f));
        
        // Check that expressions work properly
        CHECK(r2.x() == doctest::Approx(1.1f));
        CHECK(r2.y() == doctest::Approx(1.0f));
        
        // Check normalized reflection
        CHECK(length(r3) == doctest::Approx(length(normalize(incident))));
        
        // Reflected ray should have same angle with normal as incident ray
        // The angle between incident and normal equals π - angle between reflected and normal
        float angle_in = angle_between(-incident, normal);  // Negate incident to measure from same side
        float angle_out = angle_between(r1, normal);
        CHECK(std::abs(angle_in - angle_out) < 0.001f);
    }
    
    SUBCASE("lerp with expressions") {
        vec3<float> a(0.0f, 0.0f, 0.0f);
        vec3<float> b(10.0f, 10.0f, 10.0f);
        
        // lerp with expressions
        vec3<float> l1 = lerp(a, b, 0.5f);
        vec3<float> temp1(1.0f, 1.0f, 1.0f);
        vec3<float> l2 = lerp(a + temp1, b, 0.5f);
        vec3<float> temp2(2.0f, 2.0f, 2.0f);
        vec3<float> l3 = lerp(a, b - temp2, 0.5f);
        
        CHECK(l1.x() == doctest::Approx(5.0f));
        CHECK(l2.x() == doctest::Approx(5.5f));
        CHECK(l3.x() == doctest::Approx(4.0f));
        
        // lerp between expressions
        vec3<float> l4 = lerp(2.0f * a, 0.5f * b, 0.4f);
        CHECK(l4.x() == doctest::Approx(2.0f)); // 0.6*0 + 0.4*5
    }
    
    SUBCASE("distance operations with expressions") {
        vec3<float> a(0.0f, 0.0f, 0.0f);
        vec3<float> b(3.0f, 4.0f, 0.0f);
        
        // distance with expressions
        float d1 = distance(a, b);
        vec3<float> temp1(1.0f, 0.0f, 0.0f);
        float d2 = distance(a + temp1, b);
        vec3<float> temp2(0.0f, 0.0f, 5.0f);
        float d3 = distance(a, b + temp2);
        
        CHECK(d1 == doctest::Approx(5.0f));
        CHECK(d2 == doctest::Approx(std::sqrt(4.0f + 16.0f)));
        CHECK(d3 == doctest::Approx(std::sqrt(9.0f + 16.0f + 25.0f)));
        
        // distance_squared with expressions
        float ds1 = distance_squared(a, b);
        float ds2 = distance_squared(2.0f * a, 2.0f * b);
        
        CHECK(ds1 == doctest::Approx(25.0f));
        CHECK(ds2 == doctest::Approx(100.0f));
    }
    
    SUBCASE("min/max/abs with expressions") {
        vec3<float> a(1.0f, -2.0f, 3.0f);
        vec3<float> b(-1.0f, 2.0f, 1.0f);
        
        // min with expressions
        vec3<float> m1 = min(a, b);
        vec3<float> temp(1.0f, 1.0f, 1.0f);
        vec3<float> m2 = min(a + temp, b);
        vec3<float> m3 = min(abs(a), abs(b));
        
        CHECK(m1.x() == doctest::Approx(-1.0f));
        CHECK(m1.y() == doctest::Approx(-2.0f));
        CHECK(m1.z() == doctest::Approx(1.0f));
        
        CHECK(m2.x() == doctest::Approx(-1.0f));
        CHECK(m2.y() == doctest::Approx(-1.0f));
        CHECK(m2.z() == doctest::Approx(1.0f));
        
        CHECK(m3.x() == doctest::Approx(1.0f));
        CHECK(m3.y() == doctest::Approx(2.0f));
        CHECK(m3.z() == doctest::Approx(1.0f));
        
        // max with expressions
        vec3<float> mx1 = max(a, b);
        vec3<float> mx2 = max(normalize(a) * 2.0f, normalize(b) * 2.0f);
        
        CHECK(mx1.x() == doctest::Approx(1.0f));
        CHECK(mx1.y() == doctest::Approx(2.0f));
        CHECK(mx1.z() == doctest::Approx(3.0f));
        
        // Check max with normalized vectors - result should be <= 2.0 for each component
        CHECK(mx2.x() <= 2.0f);
        CHECK(mx2.y() <= 2.0f);
        CHECK(mx2.z() <= 2.0f);
        
        // abs with expressions
        vec3<float> ab1 = abs(a);
        vec3<float> ab2 = abs(a - b);
        
        CHECK(ab1.x() == doctest::Approx(1.0f));
        CHECK(ab1.y() == doctest::Approx(2.0f));
        CHECK(ab1.z() == doctest::Approx(3.0f));
        
        CHECK(ab2.x() == doctest::Approx(2.0f));
        CHECK(ab2.y() == doctest::Approx(4.0f));
        CHECK(ab2.z() == doctest::Approx(2.0f));
    }
    
    SUBCASE("clamp with expressions") {
        vec3<float> v(0.5f, 1.5f, -0.5f);
        vec3<float> min_val(0.0f, 0.0f, 0.0f);
        vec3<float> max_val(1.0f, 1.0f, 1.0f);
        
        // clamp with vector bounds
        vec3<float> c1 = clamp(v, min_val, max_val);
        CHECK(c1.x() == doctest::Approx(0.5f));
        CHECK(c1.y() == doctest::Approx(1.0f));
        CHECK(c1.z() == doctest::Approx(0.0f));
        
        // clamp with scalar bounds
        vec3<float> c2 = clamp(v * 2.0f, 0.0f, 1.0f);
        CHECK(c2.x() == doctest::Approx(1.0f));
        CHECK(c2.y() == doctest::Approx(1.0f));
        CHECK(c2.z() == doctest::Approx(0.0f));
        
        // clamp expression
        vec3<float> temp(0.5f, 0.5f, 0.5f);
        vec3<float> c3 = clamp(v + temp, min_val, max_val);
        CHECK(c3.x() == doctest::Approx(1.0f));
        CHECK(c3.y() == doctest::Approx(1.0f));
        CHECK(c3.z() == doctest::Approx(0.0f));
    }
    
    SUBCASE("angle between expressions") {
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(0.0f, 1.0f, 0.0f);
        vec3<float> c(1.0f, 1.0f, 0.0f);
        
        // angle with expressions
        float a1 = angle_between(a, b);
        float a2 = angle_between(a + b, c);
        float a3 = angle_between(normalize(a), normalize(b));
        float a4 = angle_between(2.0f * a, 3.0f * b);
        
        CHECK(a1 == doctest::Approx(constants<float>::half_pi));
        CHECK(a2 == doctest::Approx(0.0f).epsilon(0.001f)); // same direction
        CHECK(a3 == doctest::Approx(constants<float>::half_pi));
        CHECK(a4 == doctest::Approx(constants<float>::half_pi)); // scaling doesn't affect angle
    }
    
    SUBCASE("project and reject with expressions") {
        vec3<float> a(3.0f, 4.0f, 0.0f);
        vec3<float> b(1.0f, 0.0f, 0.0f);
        
        // project onto axis
        vec3<float> p1 = project(a, b);
        CHECK(p1.x() == doctest::Approx(3.0f));
        CHECK(p1.y() == doctest::Approx(0.0f));
        CHECK(p1.z() == doctest::Approx(0.0f));
        
        // project expression
        vec3<float> temp1(0.0f, 0.0f, 5.0f);
        vec3<float> p2 = project(a + temp1, b);
        CHECK(p2.x() == doctest::Approx(3.0f));
        CHECK(p2.y() == doctest::Approx(0.0f));
        CHECK(p2.z() == doctest::Approx(0.0f));
        
        // reject
        vec3<float> r1 = reject(a, b);
        CHECK(r1.x() == doctest::Approx(0.0f));
        CHECK(r1.y() == doctest::Approx(4.0f));
        CHECK(r1.z() == doctest::Approx(0.0f));
        
        // reject expression
        vec3<float> temp2(0.0f, 0.0f, 5.0f);
        vec3<float> r2 = reject(a + temp2, b);
        CHECK(r2.x() == doctest::Approx(0.0f));
        CHECK(r2.y() == doctest::Approx(4.0f));
        CHECK(r2.z() == doctest::Approx(5.0f));
        
        // project + reject should equal original
        vec3<float> sum = project(a, b) + reject(a, b);
        CHECK(sum.x() == doctest::Approx(a.x()));
        CHECK(sum.y() == doctest::Approx(a.y()));
        CHECK(sum.z() == doctest::Approx(a.z()));
    }
    
    SUBCASE("faceforward with expressions") {
        vec3<float> n(0.0f, 1.0f, 0.0f);    // surface normal
        vec3<float> i(1.0f, -1.0f, 0.0f);   // incident ray
        vec3<float> nref(0.0f, 1.0f, 0.0f); // reference normal
        
        // faceforward basic
        vec3<float> f1 = faceforward(n, i, nref);
        CHECK(f1.y() == doctest::Approx(1.0f)); // keeps original direction
        
        // faceforward with opposing incident
        vec3<float> i2(1.0f, 1.0f, 0.0f);
        vec3<float> f2 = faceforward(n, i2, nref);
        CHECK(f2.y() == doctest::Approx(-1.0f)); // flips direction
        
        // faceforward with expression
        vec3<float> f3 = faceforward(normalize(n), i, nref);
        CHECK(f3.y() == doctest::Approx(1.0f));
        
        vec3<float> f4 = faceforward(n, normalize(i), nref);
        CHECK(f4.y() == doctest::Approx(1.0f));
    }
    
    SUBCASE("complex expression chains") {
        vec3<float> a(1.0f, 0.0f, 0.0f);
        vec3<float> b(0.0f, 1.0f, 0.0f);
        vec3<float> c(0.0f, 0.0f, 1.0f);
        
        // Chain multiple operations
        vec3<float> result1 = normalize(cross(a, b) + cross(b, c));
        CHECK(length(result1) == doctest::Approx(1.0f));
        
        // Complex expression with multiple operations
        vec3<float> result2 = clamp(
            lerp(
                normalize(a), 
                normalize(b), 
                0.5f
            ),
            0.0f,
            1.0f
        );
        
        // All components should be between 0 and 1
        CHECK(result2.x() >= 0.0f);
        CHECK(result2.x() <= 1.0f);
        CHECK(result2.y() >= 0.0f);
        CHECK(result2.y() <= 1.0f);
        CHECK(result2.z() >= 0.0f);
        CHECK(result2.z() <= 1.0f);
        
        // Very complex expression
        float complex_result = dot(
            normalize(reflect(a, normalize(b))),
            normalize(project(c, cross(a, b)))
        );
        CHECK(std::isfinite(complex_result));
    }
    
    SUBCASE("special cases and edge conditions") {
        vec3<float> zero(0.0f, 0.0f, 0.0f);
        vec3<float> a(1.0f, 0.0f, 0.0f);
        
        // Operations with zero vector
        vec3<float> n1 = a + zero;
        CHECK(n1.x() == doctest::Approx(1.0f));
        
        // Very small vectors
        vec3<float> tiny(1e-10f, 1e-10f, 1e-10f);
        float len_tiny = length(tiny);
        CHECK(len_tiny > 0.0f);
        CHECK(len_tiny < 1e-5f);
        
        // Large vectors
        vec3<float> large(1e10f, 1e10f, 1e10f);
        vec3<float> norm_large = normalize(large);
        CHECK(length(norm_large) == doctest::Approx(1.0f));
    }
}

TEST_CASE("euler::vector operations expression performance") {
    using namespace euler;
    
    SUBCASE("expression templates avoid temporaries") {
        vec3<float> a(1.0f, 2.0f, 3.0f);
        vec3<float> b(4.0f, 5.0f, 6.0f);
        vec3<float> c(7.0f, 8.0f, 9.0f);
        
        // This should create no temporaries until final assignment
        vec3<float> result = a + b + c;
        
        CHECK(result.x() == 12.0f);
        CHECK(result.y() == 15.0f);
        CHECK(result.z() == 18.0f);
        
        // Complex expression should also avoid temporaries
        vec3<float> complex = normalize(cross(a + b, c - a));
        CHECK(length(complex) == doctest::Approx(1.0f));
    }
}