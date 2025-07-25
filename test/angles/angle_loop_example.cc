#include <doctest/doctest.h>
#include <euler/angles/degree.hh>
#include <euler/angles/radian.hh>
#include <euler/angles/angle_ops.hh>
#include <euler/core/types.hh>
#include <vector>
#include <numeric>

using namespace euler;
using namespace euler::literals;

TEST_CASE("Practical angle loop examples") {
    SUBCASE("Generate rotation angles for a circle") {
        std::vector<degree<float>> angles;
        
        // Generate 12 evenly spaced angles around a circle
        for (auto angle = 0.0_deg; angle < 360.0_deg; angle += 30.0_deg) {
            angles.push_back(angle);
        }
        
        CHECK(angles.size() == 12);
        CHECK(angles.front() == 0.0_deg);
        CHECK(angles.back() == 330.0_deg);
    }
    
    SUBCASE("Sweep angle range with fine steps") {
        std::vector<angle_components<float>> points;
        
        // Generate points along an arc from -45° to +45°
        for (auto angle = -45.0_deg; angle <= 45.0_deg; angle += 5.0_deg) {
            points.push_back(angle_to_components(angle));
        }
        
        CHECK(points.size() == 19);  // -45, -40, ..., 0, ..., 40, 45
        
        // Check middle point (0 degrees)
        auto& middle = points[9];
        CHECK(std::abs(middle.cos - 1.0f) < 1e-6f);
        CHECK(std::abs(middle.sin - 0.0f) < 1e-6f);
    }
    
    SUBCASE("Find angle in range") {
        auto target = 165.0_deg;
        degree<float> closest;
        float min_diff = 360.0f;
        
        // Search in 15-degree increments
        for (auto angle = 0.0_deg; angle < 360.0_deg; angle += 15.0_deg) {
            auto diff = std::abs((angle - target).value());
            if (diff < min_diff) {
                min_diff = diff;
                closest = angle;
            }
        }
        
        CHECK(closest == 165.0_deg);  // Exact match exists
    }
    
    SUBCASE("Accumulate angles with wrapping") {
        auto total = 0.0_deg;
        
        // Add angles that will exceed 360 degrees
        for (int i = 0; i < 5; ++i) {
            total += 90.0_deg;
            total = wrap_positive(total);
        }
        
        CHECK(total == 90.0_deg);  // 450° wrapped to 90°
    }
    
    SUBCASE("Generate sine table") {
        std::vector<float> sine_table;
        
        // Generate sine values for every 10 degrees
        for (auto angle = 0.0_deg; angle <= 90.0_deg; angle += 10.0_deg) {
            auto components = angle_to_components(angle);
            sine_table.push_back(components.sin);
        }
        
        CHECK(sine_table.size() == 10);
        CHECK(std::abs(sine_table[0] - 0.0f) < 1e-6f);     // sin(0°)
        CHECK(std::abs(sine_table[9] - 1.0f) < 1e-6f);     // sin(90°)
        CHECK(std::abs(sine_table[3] - 0.5f) < 1e-2f);     // sin(30°) ≈ 0.5
    }
    
    SUBCASE("Iterate with custom step using while loop") {
        auto start = 0.0_rad;
        auto end = constants<float>::pi * 1.0_rad;
        auto step = constants<float>::pi / 6.0f * 1.0_rad;  // π/6 radians = 30°
        
        std::vector<radian<float>> angles;
        auto current = start;
        
        while (current <= end) {
            angles.push_back(current);
            current += step;
        }
        
        CHECK(angles.size() == 7);  // 0, π/6, π/3, π/2, 2π/3, 5π/6, π
    }
}