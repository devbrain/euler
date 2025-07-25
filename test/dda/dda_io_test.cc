/**
 * @file dda_io_test.cc
 * @brief Tests for DDA type I/O operators
 */

#include <euler/dda/io.hh>
#include <euler/dda/dda_traits.hh>
#include <doctest/doctest.h>
#include <sstream>
#include <iomanip>

using namespace euler;
using namespace euler::dda;

TEST_CASE("Pixel output operator") {
    SUBCASE("Integer pixel") {
        pixel<int> p{{10, 20}};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(10, 20)");
    }
    
    SUBCASE("Float pixel") {
        pixel<float> p{{1.5f, 2.5f}};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(1.5, 2.5)");
    }
}

TEST_CASE("Antialiased pixel output operator") {
    SUBCASE("Basic output") {
        aa_pixel<float> p{{10.5f, 20.5f}, 0.75f, 0.0f};
        std::ostringstream oss;
        oss << p;
        std::string result = oss.str();
        CHECK(result.find("(10.5, 20.5)") != std::string::npos);
        CHECK(result.find("[coverage: 0.75") != std::string::npos);
        CHECK(result.find("distance:") == std::string::npos); // No distance since it's 0
    }
    
    SUBCASE("With distance") {
        aa_pixel<float> p{{10.5f, 20.5f}, 0.75f, 0.25f};
        std::ostringstream oss;
        oss << p;
        std::string result = oss.str();
        CHECK(result.find("(10.5, 20.5)") != std::string::npos);
        CHECK(result.find("[coverage: 0.75") != std::string::npos);
        CHECK(result.find("distance: 0.25") != std::string::npos);
    }
    
    SUBCASE("With formatting") {
        aa_pixel<double> p{{10.123456, 20.654321}, 0.123456f, 0.654321f};
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << p;
        // Check that precision applies to all values
        CHECK(oss.str().find("0.123") != std::string::npos);
        CHECK(oss.str().find("0.654") != std::string::npos);
    }
}

TEST_CASE("Span output operator") {
    SUBCASE("Basic span") {
        span s{100, 10, 50};
        std::ostringstream oss;
        oss << s;
        CHECK(oss.str() == "span(y: 100, x: [10, 50])");
    }
    
    SUBCASE("Single pixel span") {
        span s{50, 25, 25};
        std::ostringstream oss;
        oss << s;
        CHECK(oss.str() == "span(y: 50, x: [25, 25])");
        CHECK(s.width() == 1);
    }
}

TEST_CASE("Rectangle output operator") {
    SUBCASE("Integer rectangle") {
        rectangle<int> r{{0, 0}, {100, 100}};
        std::ostringstream oss;
        oss << r;
        CHECK(oss.str() == "[(0, 0) - (100, 100)]");
    }
    
    SUBCASE("Float rectangle") {
        rectangle<float> r{{-10.5f, -20.5f}, {30.5f, 40.5f}};
        std::ostringstream oss;
        oss << r;
        CHECK(oss.str() == "[(-10.5, -20.5) - (30.5, 40.5)]");
    }
}

TEST_CASE("Enum output operators") {
    std::ostringstream oss;
    
    SUBCASE("curve_type") {
        oss << curve_type::parametric;
        CHECK(oss.str() == "parametric");
        
        oss.str("");
        oss << curve_type::cartesian;
        CHECK(oss.str() == "cartesian");
        
        oss.str("");
        oss << curve_type::polar;
        CHECK(oss.str() == "polar");
    }
    
    SUBCASE("cap_style") {
        oss << cap_style::butt;
        CHECK(oss.str() == "butt");
        
        oss.str("");
        oss << cap_style::round;
        CHECK(oss.str() == "round");
        
        oss.str("");
        oss << cap_style::square;
        CHECK(oss.str() == "square");
    }
    
    SUBCASE("aa_algorithm") {
        oss << aa_algorithm::wu;
        CHECK(oss.str() == "wu");
        
        oss.str("");
        oss << aa_algorithm::gupta_sproull;
        CHECK(oss.str() == "gupta_sproull");
        
        oss.str("");
        oss << aa_algorithm::supersampling;
        CHECK(oss.str() == "supersampling");
    }
}

TEST_CASE("Combined output") {
    pixel<int> p{{10, 20}};
    aa_pixel<float> aa{{15.5f, 25.5f}, 0.8f, 0.1f};
    span s{50, 0, 99};
    rectangle<int> r{{0, 0}, {100, 100}};
    
    std::ostringstream oss;
    oss << "Pixel: " << p << ", AA: " << aa << ", Span: " << s << ", Rect: " << r;
    
    // Check that the output contains the expected parts
    std::string result = oss.str();
    CHECK(result.find("Pixel: (10, 20)") != std::string::npos);
    CHECK(result.find("AA: (15.5, 25.5) [coverage: 0.8") != std::string::npos);
    CHECK(result.find("distance: 0.1") != std::string::npos);
    CHECK(result.find("Span: span(y: 50, x: [0, 99])") != std::string::npos);
    CHECK(result.find("Rect: [(0, 0) - (100, 100)]") != std::string::npos);
}