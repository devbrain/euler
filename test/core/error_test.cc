#include <doctest/doctest.h>
#include <euler/core/error.hh>

TEST_CASE("euler::error_code") {
    using namespace euler;
    
    SUBCASE("error code to string conversion") {
        CHECK(std::string(error_code_to_string(error_code::success)) == "success");
        CHECK(std::string(error_code_to_string(error_code::dimension_mismatch)) == "dimension_mismatch");
        CHECK(std::string(error_code_to_string(error_code::index_out_of_bounds)) == "index_out_of_bounds");
        CHECK(std::string(error_code_to_string(error_code::singular_matrix)) == "singular_matrix");
        CHECK(std::string(error_code_to_string(error_code::invalid_argument)) == "invalid_argument");
        CHECK(std::string(error_code_to_string(error_code::numerical_overflow)) == "numerical_overflow");
        CHECK(std::string(error_code_to_string(error_code::not_implemented)) == "not_implemented");
        CHECK(std::string(error_code_to_string(error_code::null_pointer)) == "null_pointer");
        CHECK(std::string(error_code_to_string(error_code::invalid_size)) == "invalid_size");
    }
}

#ifdef EULER_DEBUG
TEST_CASE("euler::error handling macros in debug mode") {
    using namespace euler;
    
    SUBCASE("EULER_CHECK") {
        // Should pass
        EULER_CHECK(true, error_code::invalid_argument, "This should not fail");
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK(false, error_code::invalid_argument, "This should fail"),
            std::runtime_error
        );
    }
    
    SUBCASE("EULER_CHECK_INDEX") {
        // Should pass
        EULER_CHECK_INDEX(5, 10);
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK_INDEX(10, 10),
            std::runtime_error
        );
        
        CHECK_THROWS_AS(
            EULER_CHECK_INDEX(15, 10),
            std::runtime_error
        );
    }
    
    SUBCASE("EULER_CHECK_DIMENSIONS") {
        // Should pass
        EULER_CHECK_DIMENSIONS(3, 4, 3, 4);
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK_DIMENSIONS(3, 4, 4, 3),
            std::runtime_error
        );
    }
    
    SUBCASE("EULER_CHECK_MULTIPLY_DIMENSIONS") {
        // Should pass
        EULER_CHECK_MULTIPLY_DIMENSIONS(3, 3);
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK_MULTIPLY_DIMENSIONS(3, 4),
            std::runtime_error
        );
    }
    
    SUBCASE("EULER_CHECK_NOT_NULL") {
        int value = 42;
        int* ptr = &value;
        int* null_ptr = nullptr;
        
        // Should pass
        EULER_CHECK_NOT_NULL(ptr);
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK_NOT_NULL(null_ptr),
            std::runtime_error
        );
    }
    
    SUBCASE("EULER_CHECK_POSITIVE") {
        // Should pass
        EULER_CHECK_POSITIVE(5, "test value");
        
        // Should fail
        CHECK_THROWS_AS(
            EULER_CHECK_POSITIVE(0, "test value"),
            std::runtime_error
        );
        
        CHECK_THROWS_AS(
            EULER_CHECK_POSITIVE(-5, "test value"),
            std::runtime_error
        );
    }
}
#endif

TEST_CASE("euler::critical checks always enabled") {
    using namespace euler;
    
    SUBCASE("EULER_CRITICAL_CHECK") {
        // Should pass
        EULER_CRITICAL_CHECK(true, error_code::invalid_argument, "This should not fail");
        
        // Should fail even in release mode
        CHECK_THROWS_AS(
            EULER_CRITICAL_CHECK(false, error_code::invalid_argument, "This should always fail"),
            std::runtime_error
        );
    }
}