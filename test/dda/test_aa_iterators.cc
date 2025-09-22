#include <euler/dda/dda.hh>
#include <doctest/doctest.h>
#include <euler/core/compiler.hh>
#include <cmath>
#include <vector>

using namespace euler;
using namespace euler::dda;

// Test that AA curve iterator compiles and generates pixels
EULER_DISABLE_WARNING_PUSH
EULER_DISABLE_WARNING_STRICT_OVERFLOW
TEST_CASE("AA curve iterator") {
    // Test parametric curve - a small arc instead of full circle
    auto circle = [](float t) -> point2<float> {
        return {10.0f * cos(t), 10.0f * sin(t)};
    };
    
    // Test a small arc (0 to π/4) instead of full circle
    auto aa_curve = make_aa_curve_iterator(circle, 0.0f, 3.14159f / 4.0f);
    
    
    int count = 0;
    float total_coverage = 0.0f;
    float min_coverage = 1.0f;
    float max_coverage = 0.0f;
    
    while (aa_curve != decltype(aa_curve)::end()) {
        auto pixel = *aa_curve;
        total_coverage += pixel.coverage;
        min_coverage = std::min(min_coverage, pixel.coverage);
        max_coverage = std::max(max_coverage, pixel.coverage);
        count++;
        ++aa_curve;
        
        // Safety check to prevent runaway loops
        if (count > 1000) {
            FAIL("Iterator generated too many pixels - possible infinite loop");
            break;
        }
    }
    
    // Check results after the loop
    CHECK(count > 10);  // Should generate some pixels for an arc
    CHECK(count < 200); // But not too many for a small arc
    CHECK(total_coverage > 5.0f);  // Should have some coverage
    CHECK(min_coverage >= 0.0f);
    CHECK(min_coverage <= 1.0f);
    CHECK(max_coverage >= 0.0f);
    CHECK(max_coverage <= 1.0f);
}

// Test AA Cartesian curve
TEST_CASE("AA Cartesian curve") {
    // Test parabola y = x^2 / 10
    auto parabola = [](float x) { return x * x / 10.0f; };
    
    // Test smaller range
    auto aa_cart = make_aa_cartesian_curve(parabola, -5.0f, 5.0f);
    
    int count = 0;
    float total_coverage = 0.0f;
    for (; aa_cart != decltype(aa_cart)::end(); ++aa_cart) {
        auto pixel = *aa_cart;
        total_coverage += pixel.coverage;
        count++;
        
        if (count > 500) {
            FAIL("Iterator generated too many pixels");
            break;
        }
    }
    
    CHECK(count > 20);  // Should generate pixels along the curve
    CHECK(count < 300); // But not too many
    CHECK(total_coverage > 10.0f);
}

// Test AA polar curve
TEST_CASE("AA polar curve") {
    // Test spiral r = theta
    auto spiral = [](float theta) { return theta * 2.0f; };
    
    // Test smaller range - one turn instead of four
    auto aa_polar = make_aa_polar_curve(spiral, 0.0f, 2 * 3.14159f, point2<float>{50.0f, 50.0f});
    
    int count = 0;
    float total_coverage = 0.0f;
    for (; aa_polar != decltype(aa_polar)::end(); ++aa_polar) {
        auto pixel = *aa_polar;
        total_coverage += pixel.coverage;
        count++;
        
        if (count > 1000) {
            FAIL("Iterator generated too many pixels");
            break;
        }
    }
    
    CHECK(count > 50);  // Should generate many pixels for a spiral
    CHECK(count < 800); // But not too many
    CHECK(total_coverage > 20.0f);
}

// Test AA B-spline iterator
TEST_CASE("AA B-spline iterator") {
    std::vector<point2<float>> control_points = {
        {0.0f, 0.0f},
        {10.0f, 20.0f},
        {30.0f, 15.0f},
        {40.0f, 5.0f}
    };
    
    auto aa_spline = make_aa_bspline(control_points, 3);
    
    int count = 0;
    float total_coverage = 0.0f;
    for (; aa_spline != decltype(aa_spline)::end(); ++aa_spline) {
        auto pixel = *aa_spline;
        CHECK(pixel.coverage >= 0.0f);
        CHECK(pixel.coverage <= 1.0f);
        total_coverage += pixel.coverage;
        count++;
    }
    
    CHECK(count > 30);  // Should generate pixels along the spline
    CHECK(total_coverage > 10.0f);  // Should have significant coverage
}

// Test AA Catmull-Rom spline
TEST_CASE("AA Catmull-Rom spline") {
    std::vector<point2<float>> points = {
        {0.0f, 0.0f},
        {10.0f, 10.0f},
        {20.0f, 5.0f},
        {30.0f, 15.0f}
    };
    
    auto aa_catmull = make_aa_catmull_rom(points);
    
    int count = 0;
    for (; aa_catmull != decltype(aa_catmull)::end(); ++aa_catmull) {
        auto pixel = *aa_catmull;
        CHECK(pixel.coverage >= 0.0f);
        CHECK(pixel.coverage <= 1.0f);
        count++;
    }
    
    CHECK(count > 20);  // Should generate pixels along the spline
}

// Test edge cases
TEST_CASE("AA edge cases") {
    // Single point B-spline
    std::vector<point2<float>> single_point = {{5.0f, 5.0f}};
    auto aa_single = make_aa_bspline(single_point, 0);
    
    int count = 0;
    for (; aa_single != decltype(aa_single)::end(); ++aa_single) {
        count++;
    }
    CHECK(count > 0);  // Should still generate at least one pixel
    
    // Straight line curve
    auto line = [](float t) -> point2<float> { return {t, t}; };
    auto aa_line = make_aa_curve_iterator(line, 0.0f, 10.0f);
    
    count = 0;
    for (; aa_line != decltype(aa_line)::end(); ++aa_line) {
        auto pixel = *aa_line;
        CHECK(pixel.coverage >= 0.0f);
        CHECK(pixel.coverage <= 1.0f);
        count++;
    }
    CHECK(count > 10);  // Should generate pixels along the line
}
EULER_DISABLE_WARNING_POP