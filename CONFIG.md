# Euler Library Configuration Guide

The Euler library provides various configuration macros that allow you to customize its behavior at compile time. This guide explains all available configuration options and how to use them.

## How to Use Configuration Macros

Configuration macros can be set in three ways:

1. **Define in your source code before including Euler headers:**
   ```cpp
   #define EULER_DISABLE_SIMD
   #define EULER_DEFAULT_EPSILON 1e-8
   #include <euler/euler.hh>
   ```

2. **Pass as compiler flags:**
   ```bash
   g++ -DEULER_DISABLE_SIMD -DEULER_DEFAULT_EPSILON=1e-8 myprogram.cpp
   ```

3. **Set in CMake:**
   ```cmake
   add_compile_definitions(EULER_DISABLE_SIMD)
   add_compile_definitions(EULER_DEFAULT_EPSILON=1e-8)
   ```

## Available Configuration Macros

### Error Handling

#### `EULER_DISABLE_ENFORCE`
- **Purpose:** Disables all runtime checks and assertions
- **Use case:** Maximum performance in production when you're confident about input validity
- **Effect:** Removes bounds checking, dimension validation, and other safety checks
- **Warning:** May lead to undefined behavior if invalid inputs are provided

#### `EULER_ENABLE_TRAP`
- **Purpose:** Enables debugger traps on errors
- **Use case:** Debugging to immediately catch errors in a debugger
- **Effect:** Calls platform-specific debug break when an error occurs

### SIMD Optimization

#### `EULER_DISABLE_SIMD`
- **Purpose:** Disables all SIMD optimizations
- **Use case:** Debugging, compatibility with older processors, or deterministic behavior
- **Effect:** Forces use of scalar implementations for all operations
- **Performance impact:** Significant slowdown for vector/matrix operations

#### `EULER_SIMD_BATCH_SIZE`
- **Purpose:** Override the default SIMD batch size
- **Use case:** Tuning for specific architectures
- **Example:** `#define EULER_SIMD_BATCH_SIZE 8`
- **Note:** Must be a valid size for your target architecture

### Precision Control

#### `EULER_DEFAULT_EPSILON`
- **Purpose:** Sets the default epsilon for floating-point comparisons
- **Default:** Machine epsilon (`std::numeric_limits<T>::epsilon()`)
- **Use case:** Looser tolerance for approximate equality comparisons
- **Example:** `#define EULER_DEFAULT_EPSILON 1e-8`

#### `EULER_HIGH_PRECISION`
- **Purpose:** Use higher precision algorithms
- **Use case:** When accuracy is more important than speed
- **Effect:** May use more accurate but slower algorithms
- **Note:** Currently reserved for future use

#### `EULER_DEFAULT_PRECISION_DOUBLE`
- **Purpose:** Make `scalar` type default to `double` instead of `float`
- **Use case:** Applications requiring double precision by default
- **Effect:** Changes the default scalar type throughout the library

### Matrix Storage

#### `EULER_DEFAULT_COLUMN_MAJOR`
- **Purpose:** Make column-major the default storage order for matrices
- **Use case:** Interfacing with OpenGL or other column-major libraries
- **Effect:** New matrices default to column-major storage
- **Note:** You can still explicitly specify storage order per matrix

#### `EULER_FORCE_ROW_MAJOR`
- **Purpose:** Force all matrices to use row-major storage
- **Use case:** Consistency with row-major libraries
- **Effect:** Overrides any column-major specifications
- **Note:** Currently reserved for future use

### Debug Features

#### `EULER_ENABLE_BOUNDS_CHECK`
- **Purpose:** Enable bounds checking for all container access
- **Use case:** Development and testing to catch out-of-bounds errors
- **Effect:** Adds runtime checks for all array/matrix element access
- **Performance impact:** Minor overhead on element access

#### `EULER_ENABLE_NAN_CHECK`
- **Purpose:** Check for NaN values in computations
- **Use case:** Debugging numerical stability issues
- **Effect:** Adds checks for NaN after mathematical operations
- **Performance impact:** Moderate overhead on mathematical operations

#### `EULER_TRACE_EXPRESSIONS`
- **Purpose:** Enable expression template debugging output
- **Use case:** Understanding how expressions are evaluated
- **Effect:** Prints expression evaluation trees to stderr
- **Note:** Very verbose, use only for debugging specific issues

### Build Modes

#### `EULER_DEBUG`
- **Purpose:** Enable debug mode with verbose error messages
- **Use case:** Development and debugging
- **Effect:** Enables all assertions with detailed error messages
- **Note:** Usually set automatically in debug builds

#### `EULER_SAFE_RELEASE`
- **Purpose:** Enable safety checks in release mode
- **Use case:** Production code where safety is critical
- **Effect:** Keeps runtime checks but with minimal error messages

## Configuration Examples

### Maximum Performance Configuration
```cpp
#define EULER_DISABLE_ENFORCE      // Remove all checks
#define EULER_DEFAULT_EPSILON 1e-4  // Looser tolerance
#include <euler/euler.hh>
```

### Maximum Safety Configuration
```cpp
#define EULER_ENABLE_BOUNDS_CHECK   // Check all array access
#define EULER_ENABLE_NAN_CHECK      // Check for NaN values
#define EULER_SAFE_RELEASE          // Keep checks in release
#include <euler/euler.hh>
```

### Debug Configuration
```cpp
#define EULER_DEBUG                 // Verbose error messages
#define EULER_ENABLE_TRAP          // Break on errors
#define EULER_TRACE_EXPRESSIONS    // Trace expression evaluation
#define EULER_DISABLE_SIMD         // Simplify debugging
#include <euler/euler.hh>
```

### OpenGL Integration Configuration
```cpp
#define EULER_DEFAULT_COLUMN_MAJOR  // Match OpenGL's layout
#define EULER_DEFAULT_PRECISION_DOUBLE  // Use double precision
#include <euler/euler.hh>
```

## Performance Considerations

1. **SIMD Optimization**: Keeping SIMD enabled can provide 2-8x speedup for vector/matrix operations
2. **Runtime Checks**: Disabling checks (`EULER_DISABLE_ENFORCE`) can improve performance by 5-15%
3. **Precision**: Using `float` instead of `double` can double SIMD throughput
4. **Debug Features**: Debug features like `EULER_TRACE_EXPRESSIONS` have significant performance impact

## Best Practices

1. **Development**: Use safety checks and debug features
2. **Testing**: Enable `EULER_ENABLE_BOUNDS_CHECK` and `EULER_ENABLE_NAN_CHECK`
3. **Production**: Consider `EULER_SAFE_RELEASE` for critical applications
4. **Performance Critical**: Use `EULER_DISABLE_ENFORCE` only after thorough testing

## Compatibility Notes

- All configuration macros must be defined consistently across translation units
- Mixing different configurations in the same program may violate ODR (One Definition Rule)
- Some macros are reserved for future use and have no effect in the current version

## Troubleshooting

If you encounter issues:

1. Ensure macros are defined before including any Euler headers
2. Check that all translation units use the same configuration
3. Use the `config_demo` example to verify your configuration
4. Enable debug features to get more detailed error information