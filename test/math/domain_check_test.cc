#include <euler/math/basic.hh>
#include <euler/math/trigonometry.hh>
#include <euler/vector/vector.hh>
#include <euler/matrix/matrix.hh>
#include <doctest/doctest.h>

using namespace euler;

// Helper to check if an expression throws
template<typename Func>
bool throws_exception(Func&& f) {
    try {
        f();
        return false;
    } catch (...) {
        return true;
    }
}

TEST_CASE("Domain checks for mathematical functions") {
    SUBCASE("Square root domain checks") {
        // Scalar negative should throw
        CHECK(throws_exception([]() { euler::sqrt(-1.0f); }));

        // Vector with negative element should throw
        vec3f v(1.0f, -4.0f, 9.0f);
        CHECK(throws_exception([&v]() { euler::sqrt(v); }));

        // Positive values should work
        CHECK_NOTHROW(euler::sqrt(4.0f));
        vec3f v_pos(1.0f, 4.0f, 9.0f);
        CHECK_NOTHROW(euler::sqrt(v_pos));
    }

    SUBCASE("Logarithm domain checks") {
        // Zero and negative should throw
        CHECK(throws_exception([]() { euler::log(0.0f); }));
        CHECK(throws_exception([]() { euler::log(-1.0f); }));
        CHECK(throws_exception([]() { euler::log2(0.0f); }));
        CHECK(throws_exception([]() { euler::log10(-5.0f); }));

        // Vector with non-positive element
        vec3f v(1.0f, 0.0f, 2.0f);
        CHECK(throws_exception([&v]() { euler::log(v); }));

        // Positive values should work
        CHECK_NOTHROW(euler::log(1.0f));
        CHECK_NOTHROW(euler::log2(8.0f));
        CHECK_NOTHROW(euler::log10(100.0f));
    }

    SUBCASE("Power function domain checks") {
        // Negative base with non-integer exponent should throw
        CHECK(throws_exception([]() { euler::pow(-2.0f, 0.5f); }));
        CHECK(throws_exception([]() { euler::pow(-2.0f, 1.5f); }));

        // Vector case
        vec3f bases(-2.0f, 3.0f, 4.0f);
        vec3f exponents(0.5f, 2.0f, 3.0f);
        CHECK(throws_exception([&bases, &exponents]() { euler::pow(bases, exponents); }));

        // Valid cases should work
        CHECK_NOTHROW(euler::pow(2.0f, 0.5f));
        CHECK_NOTHROW(euler::pow(-2.0f, 2.0f));  // Integer exponent is OK
    }

    SUBCASE("Division by zero checks") {
        // Reciprocal
        CHECK(throws_exception([]() { euler::rcp(0.0f); }));
        vec3f v_zero(1.0f, 0.0f, 2.0f);
        CHECK(throws_exception([&v_zero]() { euler::rcp(v_zero); }));

        // Modulo
        CHECK(throws_exception([]() { euler::mod(5.0f, 0.0f); }));
        CHECK(throws_exception([]() { euler::fmod(5.0f, 0.0f); }));

        // Valid cases
        CHECK_NOTHROW(euler::rcp(2.0f));
        CHECK_NOTHROW(euler::mod(5.0f, 2.0f));
    }

    SUBCASE("Inverse trigonometric domain checks") {
        // asin/acos domain is [-1, 1]
        CHECK(throws_exception([]() { euler::asin(1.5f); }));
        CHECK(throws_exception([]() { euler::asin(-1.5f); }));
        CHECK(throws_exception([]() { euler::acos(2.0f); }));
        CHECK(throws_exception([]() { euler::acos(-2.0f); }));

        // Vector case
        vec3f v(0.5f, 1.5f, -0.5f);
        CHECK(throws_exception([&v]() { euler::asin(v); }));

        // Valid cases
        CHECK_NOTHROW(euler::asin(0.5f));
        CHECK_NOTHROW(euler::acos(-0.5f));
        CHECK_NOTHROW(euler::asin(1.0f));   // Boundary values
        CHECK_NOTHROW(euler::acos(-1.0f));
    }

    SUBCASE("Inverse hyperbolic domain checks") {
        // acosh domain is [1, ∞)
        CHECK(throws_exception([]() { euler::acosh(0.5f); }));
        CHECK(throws_exception([]() { euler::acosh(-1.0f); }));

        // atanh domain is (-1, 1)
        CHECK(throws_exception([]() { euler::atanh(1.0f); }));
        CHECK(throws_exception([]() { euler::atanh(-1.0f); }));
        CHECK(throws_exception([]() { euler::atanh(2.0f); }));

        // Valid cases
        CHECK_NOTHROW(euler::acosh(1.0f));   // Boundary value
        CHECK_NOTHROW(euler::acosh(2.0f));
        CHECK_NOTHROW(euler::atanh(0.5f));
        CHECK_NOTHROW(euler::atanh(-0.5f));
    }
}

#ifdef EULER_DEBUG
TEST_CASE("Domain check error messages") {
    SUBCASE("Check error messages contain function names") {
        // This test only makes sense in debug mode where we get detailed messages
        try {
            euler::sqrt(-1.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("sqrt") != std::string::npos);
        }

        try {
            euler::log(0.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("log") != std::string::npos);
        }

        try {
            euler::asin(2.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("asin") != std::string::npos);
        }
    }
}
#endif
