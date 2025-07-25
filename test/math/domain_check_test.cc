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
        CHECK(throws_exception([]() { sqrt(-1.0f); }));
        
        // Vector with negative element should throw
        vec3f v(1.0f, -4.0f, 9.0f);
        CHECK(throws_exception([&v]() { sqrt(v); }));
        
        // Positive values should work
        CHECK_NOTHROW(sqrt(4.0f));
        vec3f v_pos(1.0f, 4.0f, 9.0f);
        CHECK_NOTHROW(sqrt(v_pos));
    }
    
    SUBCASE("Logarithm domain checks") {
        // Zero and negative should throw
        CHECK(throws_exception([]() { log(0.0f); }));
        CHECK(throws_exception([]() { log(-1.0f); }));
        CHECK(throws_exception([]() { log2(0.0f); }));
        CHECK(throws_exception([]() { log10(-5.0f); }));
        
        // Vector with non-positive element
        vec3f v(1.0f, 0.0f, 2.0f);
        CHECK(throws_exception([&v]() { log(v); }));
        
        // Positive values should work
        CHECK_NOTHROW(log(1.0f));
        CHECK_NOTHROW(log2(8.0f));
        CHECK_NOTHROW(log10(100.0f));
    }
    
    SUBCASE("Power function domain checks") {
        // Negative base with non-integer exponent should throw
        CHECK(throws_exception([]() { pow(-2.0f, 0.5f); }));
        CHECK(throws_exception([]() { pow(-2.0f, 1.5f); }));
        
        // Vector case
        vec3f bases(-2.0f, 3.0f, 4.0f);
        vec3f exponents(0.5f, 2.0f, 3.0f);
        CHECK(throws_exception([&bases, &exponents]() { pow(bases, exponents); }));
        
        // Valid cases should work
        CHECK_NOTHROW(pow(2.0f, 0.5f));
        CHECK_NOTHROW(pow(-2.0f, 2.0f));  // Integer exponent is OK
    }
    
    SUBCASE("Division by zero checks") {
        // Reciprocal
        CHECK(throws_exception([]() { rcp(0.0f); }));
        vec3f v_zero(1.0f, 0.0f, 2.0f);
        CHECK(throws_exception([&v_zero]() { rcp(v_zero); }));
        
        // Modulo
        CHECK(throws_exception([]() { mod(5.0f, 0.0f); }));
        CHECK(throws_exception([]() { fmod(5.0f, 0.0f); }));
        
        // Valid cases
        CHECK_NOTHROW(rcp(2.0f));
        CHECK_NOTHROW(mod(5.0f, 2.0f));
    }
    
    SUBCASE("Inverse trigonometric domain checks") {
        // asin/acos domain is [-1, 1]
        CHECK(throws_exception([]() { asin(1.5f); }));
        CHECK(throws_exception([]() { asin(-1.5f); }));
        CHECK(throws_exception([]() { acos(2.0f); }));
        CHECK(throws_exception([]() { acos(-2.0f); }));
        
        // Vector case
        vec3f v(0.5f, 1.5f, -0.5f);
        CHECK(throws_exception([&v]() { asin(v); }));
        
        // Valid cases
        CHECK_NOTHROW(asin(0.5f));
        CHECK_NOTHROW(acos(-0.5f));
        CHECK_NOTHROW(asin(1.0f));   // Boundary values
        CHECK_NOTHROW(acos(-1.0f));
    }
    
    SUBCASE("Inverse hyperbolic domain checks") {
        // acosh domain is [1, ∞)
        CHECK(throws_exception([]() { acosh(0.5f); }));
        CHECK(throws_exception([]() { acosh(-1.0f); }));
        
        // atanh domain is (-1, 1)
        CHECK(throws_exception([]() { atanh(1.0f); }));
        CHECK(throws_exception([]() { atanh(-1.0f); }));
        CHECK(throws_exception([]() { atanh(2.0f); }));
        
        // Valid cases
        CHECK_NOTHROW(acosh(1.0f));   // Boundary value
        CHECK_NOTHROW(acosh(2.0f));
        CHECK_NOTHROW(atanh(0.5f));
        CHECK_NOTHROW(atanh(-0.5f));
    }
}

#ifdef EULER_DEBUG
TEST_CASE("Domain check error messages") {
    SUBCASE("Check error messages contain function names") {
        // This test only makes sense in debug mode where we get detailed messages
        try {
            sqrt(-1.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("sqrt") != std::string::npos);
        }
        
        try {
            log(0.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("log") != std::string::npos);
        }
        
        try {
            asin(2.0f);
            FAIL("Should have thrown");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find("asin") != std::string::npos);
        }
    }
}
#endif