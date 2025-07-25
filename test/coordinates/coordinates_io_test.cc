/**
 * @file coordinates_io_test.cc
 * @brief Tests for coordinate system I/O operators
 */

#include <euler/coordinates/io.hh>
#include <euler/coordinates/point2.hh>
#include <euler/coordinates/point3.hh>
#include <euler/coordinates/projective2.hh>
#include <euler/coordinates/projective3.hh>
#include <doctest/doctest.h>
#include <sstream>
#include <iomanip>

using namespace euler;

// Type aliases
using projective2f = projective2<float>;
using projective2d = projective2<double>;
using projective3f = projective3<float>;
using projective3d = projective3<double>;

TEST_CASE("2D point output operator") {
    SUBCASE("Basic output") {
        point2f p{1.5f, 2.5f};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(1.5, 2.5)");
    }
    
    SUBCASE("Integer points") {
        point2i p{10, 20};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(10, 20)");
    }
    
    SUBCASE("With formatting") {
        point2d p{1.234567, 2.345678};
        std::ostringstream oss;
        
        // Test precision
        oss << std::fixed << std::setprecision(3) << p;
        CHECK(oss.str() == "(1.235, 2.346)");
        
        // Test width
        oss.str("");
        oss << std::setw(10) << p;
        // Width applies to each component
        CHECK(oss.str().find("     1.235") != std::string::npos);
    }
}

TEST_CASE("3D point output operator") {
    SUBCASE("Basic output") {
        point3f p{1.0f, 2.0f, 3.0f};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(1, 2, 3)");
    }
    
    SUBCASE("Double precision") {
        point3d p{1.5, 2.5, 3.5};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "(1.5, 2.5, 3.5)");
    }
}

TEST_CASE("2D projective output operator") {
    SUBCASE("Basic output") {
        projective2f p{100.0f, 200.0f, 2.0f};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "[100, 200, 2]");
    }
    
    SUBCASE("With formatting") {
        projective2d p{1.234567, 2.345678, 0.5};
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << p;
        CHECK(oss.str() == "[1.23, 2.35, 0.50]");
    }
}

TEST_CASE("3D projective output operator") {
    SUBCASE("Basic output") {
        projective3f p{100.0f, 200.0f, 300.0f, 2.0f};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "[100, 200, 300, 2]");
    }
    
    SUBCASE("Double projective") {
        projective3d p{10.0, 20.0, 30.0, 1.0};
        std::ostringstream oss;
        oss << p;
        CHECK(oss.str() == "[10, 20, 30, 1]");
    }
}

TEST_CASE("Multiple coordinate outputs") {
    point2f p2{1.5f, 2.5f};
    point3f p3{1.0f, 2.0f, 3.0f};
    projective2f proj2{100.0f, 200.0f, 2.0f};
    projective3f proj3{100.0f, 200.0f, 300.0f, 2.0f};
    
    std::ostringstream oss;
    oss << "2D point: " << p2 << ", 3D point: " << p3
        << ", 2D proj: " << proj2 << ", 3D proj: " << proj3;
    
    CHECK(oss.str() == "2D point: (1.5, 2.5), 3D point: (1, 2, 3), "
                       "2D proj: [100, 200, 2], 3D proj: [100, 200, 300, 2]");
}