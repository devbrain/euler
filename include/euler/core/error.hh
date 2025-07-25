/**
 * @file error.hh
 * @brief Error handling infrastructure for the Euler library
 */
#pragma once

#include <failsafe/failsafe.hh>
#include <euler/core/types.hh>
#include <cmath>
#include <stdexcept>

namespace euler {

/**
 * @enum error_code
 * @brief Error codes used throughout the Euler library
 */
enum class error_code : int {
    success = 0,              ///< No error
    dimension_mismatch = 1,   ///< Matrix/vector dimensions don't match for operation
    index_out_of_bounds = 2,  ///< Array/matrix index is out of valid range
    singular_matrix = 3,      ///< Matrix is singular (not invertible)
    invalid_argument = 4,     ///< Invalid argument passed to function
    numerical_overflow = 5,   ///< Numerical overflow detected
    not_implemented = 6,      ///< Feature not yet implemented
    null_pointer = 7,         ///< Null pointer passed where non-null expected
    invalid_size = 8,         ///< Invalid size parameter
    performance_warning = 9   ///< Performance warning (not an error)
};

/**
 * @brief Convert error code to human-readable string
 * @param ec The error code to convert
 * @return String representation of the error code
 */
constexpr const char* error_code_to_string(error_code ec) {
    switch (ec) {
        case error_code::success: return "success";
        case error_code::dimension_mismatch: return "dimension_mismatch";
        case error_code::index_out_of_bounds: return "index_out_of_bounds";
        case error_code::singular_matrix: return "singular_matrix";
        case error_code::invalid_argument: return "invalid_argument";
        case error_code::numerical_overflow: return "numerical_overflow";
        case error_code::not_implemented: return "not_implemented";
        case error_code::null_pointer: return "null_pointer";
        case error_code::invalid_size: return "invalid_size";
        case error_code::performance_warning: return "performance_warning";
        default: return "unknown_error";
    }
}

/**
 * @class euler_error
 * @brief Base exception class for all Euler library errors
 */
class euler_error : public std::runtime_error {
public:
    /**
     * @brief Construct an euler_error with a message
     * @param msg Error message
     */
    explicit euler_error(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @class dimension_error
 * @brief Exception thrown when matrix/vector dimensions are incompatible
 */
class dimension_error : public euler_error {
public:
    /**
     * @brief Construct a dimension_error with a message
     * @param msg Error message describing the dimension mismatch
     */
    explicit dimension_error(const std::string& msg) : euler_error(msg) {}
};

/**
 * @class index_error
 * @brief Exception thrown when an index is out of bounds
 */
class index_error : public euler_error {
public:
    /**
     * @brief Construct an index_error with a message
     * @param msg Error message describing the index violation
     */
    explicit index_error(const std::string& msg) : euler_error(msg) {}
};

/**
 * @class numerical_error  
 * @brief Exception thrown for numerical computation errors
 */
class numerical_error : public euler_error {
public:
    /**
     * @brief Construct a numerical_error with a message
     * @param msg Error message describing the numerical issue
     */
    explicit numerical_error(const std::string& msg) : euler_error(msg) {}
};

/**
 * @defgroup ErrorMacros Error Checking Macros
 * @brief Macros for runtime error checking with different behavior in debug/release
 * @{
 */

// If EULER_DISABLE_ENFORCE is defined, disable all runtime checks
#ifdef EULER_DISABLE_ENFORCE
    #define EULER_CHECK(condition, error_code, ...) ((void)0)
    #define EULER_CHECK_INDEX(idx, size) ((void)0)
    #define EULER_CHECK_DIMENSIONS(rows1, cols1, rows2, cols2) ((void)0)
    #define EULER_CHECK_MULTIPLY_DIMENSIONS(cols1, rows2) ((void)0)
    #define EULER_CHECK_NOT_NULL(ptr) ((void)0)
    #define EULER_DEBUG_CHECK(condition, error_code, ...) ((void)0)
    #define EULER_CHECK_POSITIVE(value, name) ((void)0)
    #define EULER_CRITICAL_CHECK(condition, error_code, ...) ((void)0)

// Debug mode checks - always enabled in debug builds
#elif defined(EULER_DEBUG)
    #define EULER_CHECK(condition, error_code, ...) \
        ENFORCE(condition)(::failsafe::detail::build_message(__VA_ARGS__))
    
    #define EULER_CHECK_INDEX(idx, size) \
        ENFORCE((idx) < (size))(::failsafe::detail::build_message( \
            "Index out of bounds: ", (idx), " >= ", (size)))
    
    #define EULER_CHECK_DIMENSIONS(rows1, cols1, rows2, cols2) \
        ENFORCE((rows1) == (rows2) && (cols1) == (cols2))( \
            ::failsafe::detail::build_message( \
                "Dimension mismatch: ", (rows1), "x", (cols1), " vs ", (rows2), "x", (cols2)))
    
    #define EULER_CHECK_MULTIPLY_DIMENSIONS(cols1, rows2) \
        ENFORCE((cols1) == (rows2))( \
            ::failsafe::detail::build_message( \
                "Cannot multiply: first matrix has ", (cols1), " columns, second has ", (rows2), " rows"))
    
    #define EULER_CHECK_NOT_NULL(ptr) \
        ENFORCE_NOT_NULL(ptr)
    
    #define EULER_DEBUG_CHECK(condition, error_code, ...) \
        ENFORCE(condition)(::failsafe::detail::build_message(__VA_ARGS__))
    
    #define EULER_CHECK_POSITIVE(value, name) \
        ENFORCE((value) > 0)(::failsafe::detail::build_message( \
            name, " must be positive, got ", (value)))

// Release mode with safety checks - can be enabled via EULER_SAFE_RELEASE
#elif defined(EULER_SAFE_RELEASE)
    #define EULER_CHECK(condition, error_code, ...) \
        ENFORCE(condition)(::failsafe::detail::build_message(__VA_ARGS__))
    
    #define EULER_CHECK_INDEX(idx, size) \
        ENFORCE((idx) < (size))("Index out of bounds")
    
    #define EULER_CHECK_DIMENSIONS(rows1, cols1, rows2, cols2) \
        ENFORCE((rows1) == (rows2) && (cols1) == (cols2))("Dimension mismatch")
    
    #define EULER_CHECK_MULTIPLY_DIMENSIONS(cols1, rows2) \
        ENFORCE((cols1) == (rows2))("Cannot multiply matrices")
    
    #define EULER_CHECK_NOT_NULL(ptr) \
        ENFORCE_NOT_NULL(ptr)
    
    #define EULER_DEBUG_CHECK(condition, error_code, ...) \
        ENFORCE(condition)(::failsafe::detail::build_message(__VA_ARGS__))
    
    #define EULER_CHECK_POSITIVE(value, name) \
        ENFORCE((value) > 0)("Invalid size")

// Full release mode - no runtime checks
#else
    #define EULER_CHECK(condition, error_code, ...) ((void)0)
    #define EULER_CHECK_INDEX(idx, size) ((void)0)
    #define EULER_CHECK_DIMENSIONS(rows1, cols1, rows2, cols2) ((void)0)
    #define EULER_CHECK_MULTIPLY_DIMENSIONS(cols1, rows2) ((void)0)
    #define EULER_CHECK_NOT_NULL(ptr) ((void)0)
    #define EULER_DEBUG_CHECK(condition, error_code, ...) ((void)0)
    #define EULER_CHECK_POSITIVE(value, name) ((void)0)
#endif

// Always-on checks for critical errors (even in release) - unless explicitly disabled
#ifndef EULER_DISABLE_ENFORCE
    #define EULER_CRITICAL_CHECK(condition, error_code, ...) \
        ENFORCE(condition)(::failsafe::detail::build_message(__VA_ARGS__))
#endif

// Numerical checks - optional based on EULER_NUMERICAL_CHECKS
#ifdef EULER_NUMERICAL_CHECKS
    #define EULER_CHECK_FINITE(value, name) \
        ENFORCE(std::isfinite(value))(::failsafe::detail::build_message( \
            name, " is not finite: ", (value)))
    
    #define EULER_CHECK_NOT_ZERO(value, name) \
        ENFORCE(std::abs(value) > epsilon)(::failsafe::detail::build_message( \
            name, " is too close to zero: ", (value)))
#else
    #define EULER_CHECK_FINITE(value, name) ((void)0)
    #define EULER_CHECK_NOT_ZERO(value, name) ((void)0)
#endif

// Exception throwing macros that wrap failsafe THROW macros
#define EULER_THROW(exception_type, ...) \
    THROW(exception_type, ::failsafe::detail::build_message(__VA_ARGS__))

#define EULER_THROW_IF(condition, exception_type, ...) \
    THROW_IF(condition, exception_type, ::failsafe::detail::build_message(__VA_ARGS__))

#define EULER_THROW_UNLESS(condition, exception_type, ...) \
    THROW_UNLESS(condition, exception_type, ::failsafe::detail::build_message(__VA_ARGS__))

// Convenience macros for common exception types
#define EULER_THROW_DIMENSION_ERROR(...) \
    EULER_THROW(dimension_error, __VA_ARGS__)

#define EULER_THROW_INDEX_ERROR(...) \
    EULER_THROW(index_error, __VA_ARGS__)

#define EULER_THROW_NUMERICAL_ERROR(...) \
    EULER_THROW(numerical_error, __VA_ARGS__)

// Feature not implemented macro
#define EULER_NOT_IMPLEMENTED() \
    EULER_THROW(euler_error, "Feature not implemented")

} // namespace euler