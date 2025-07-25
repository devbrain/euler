#include <doctest/doctest.h>
#include <euler/core/types.hh>
#include <type_traits>

TEST_CASE("euler::types basic type definitions") {
    using namespace euler;
    
    SUBCASE("scalar types") {
        CHECK(std::is_same_v<float32, float>);
        CHECK(std::is_same_v<float64, double>);
        CHECK(sizeof(float32) == 4);
        CHECK(sizeof(float64) == 8);
    }
    
    SUBCASE("size types") {
        CHECK(std::is_same_v<euler::size_t, std::size_t>);
        CHECK(std::is_same_v<index_t, std::size_t>);
    }
    
    SUBCASE("default precision") {
        #ifdef EULER_DEFAULT_PRECISION_DOUBLE
        CHECK(std::is_same_v<scalar, float64>);
        #else
        CHECK(std::is_same_v<scalar, float32>);
        #endif
    }
}

TEST_CASE("euler::constants") {
    using namespace euler;
    
    SUBCASE("mathematical constants for float") {
        using C = constants<float>;
        CHECK(C::pi == doctest::Approx(3.14159265f));
        CHECK(C::e == doctest::Approx(2.71828182f));
        CHECK(C::sqrt2 == doctest::Approx(1.41421356f));
        CHECK(C::deg_to_rad == doctest::Approx(0.01745329f));
        CHECK(C::rad_to_deg == doctest::Approx(57.2957795f));
    }
    
    SUBCASE("mathematical constants for double") {
        using C = constants<double>;
        CHECK(C::pi == doctest::Approx(3.14159265358979));
        CHECK(C::e == doctest::Approx(2.71828182845904));
        CHECK(C::sqrt2 == doctest::Approx(1.41421356237309));
    }
    
    SUBCASE("convenience aliases") {
        CHECK(pi > 3.14f);
        CHECK(pi < 3.15f);
        CHECK(e > 2.71f);
        CHECK(e < 2.72f);
        CHECK(epsilon > 0.0f);
    }
}

TEST_CASE("euler::type traits helpers") {
    using namespace euler;
    
    SUBCASE("is_floating_point_v") {
        CHECK(is_floating_point_v<float>);
        CHECK(is_floating_point_v<double>);
        CHECK(!is_floating_point_v<int>);
        CHECK(!is_floating_point_v<unsigned>);
    }
    
    SUBCASE("is_arithmetic_v") {
        CHECK(is_arithmetic_v<float>);
        CHECK(is_arithmetic_v<double>);
        CHECK(is_arithmetic_v<int>);
        CHECK(is_arithmetic_v<unsigned>);
        CHECK(!is_arithmetic_v<void*>);
    }
}