/**
 * @file euler.hh
 * @brief Main header file for the Euler mathematical library
 * 
 * @mainpage Euler Mathematical Library
 * 
 * @section intro Introduction
 * 
 * Euler is a modern C++ mathematical library providing efficient implementations of:
 * - Vectors and matrices with expression templates
 * - Type-safe angles with automatic unit conversion
 * - Quaternions for 3D rotations
 * - Complex numbers
 * - Mathematical functions with SIMD optimization
 * - Random number generation for geometric objects
 * 
 * @section features Key Features
 * 
 * - **Expression Templates**: Efficient lazy evaluation of mathematical expressions
 * - **Type Safety**: Strong typing for angles, preventing degree/radian confusion
 * - **SIMD Support**: Automatic vectorization using xsimd
 * - **Header-Only**: Easy integration, just include and use
 * - **Modern C++**: Written in C++17 with clean, readable code
 * 
 * @section usage Basic Usage
 * 
 * @code{.cpp}
 * #include <euler/euler.hh>
 * 
 * using namespace euler;
 * 
 * // Vectors
 * vector<float, 3> v1(1, 2, 3);
 * vector<float, 3> v2(4, 5, 6);
 * auto v3 = cross(v1, v2);
 * 
 * // Matrices
 * matrix<float, 3, 3> m = matrix<float, 3, 3>::identity();
 * auto m_inv = inverse(m);
 * 
 * // Quaternions
 * auto q = quaternion<float>::from_axis_angle(
 *     vector<float, 3>::unit_z(), 
 *     degree<float>(90)
 * );
 * 
 * // Type-safe angles
 * auto angle1 = degree<float>(45);
 * auto angle2 = radian<float>(pi/4);
 * auto sum = angle1 + angle2; // Automatic conversion
 * @endcode
 * 
 * @section modules Library Modules
 * 
 * - @ref CoreModule "Core": Basic types, error handling, and SIMD support
 * - @ref VectorModule "Vectors": Fixed-size mathematical vectors
 * - @ref MatrixModule "Matrices": Fixed-size matrices with various operations
 * - @ref AngleModule "Angles": Type-safe angle representation
 * - @ref QuaternionModule "Quaternions": Unit quaternions for 3D rotations
 * - @ref ComplexModule "Complex": Complex number support
 * - @ref CoordinatesModule "Coordinates": 2D/3D points with projective support
 * - @ref DDAModule "DDA": Digital differential analyzer for rasterization
 * - @ref MathModule "Math": Mathematical functions (trig, exponential, etc.)
 * - @ref RandomModule "Random": Random generation for geometric objects
 * 
 * @section config Configuration Macros
 * 
 * The Euler library behavior can be customized using the following preprocessor macros:
 * 
 * **Error Handling:**
 * - `EULER_DISABLE_ENFORCE`: Disables runtime checks and enforcement (for performance)
 * - `EULER_ENABLE_TRAP`: Enables debugger traps on errors (useful for debugging)
 * 
 * **SIMD Optimization:**
 * - `EULER_DISABLE_SIMD`: Disables all SIMD optimizations
 * - `EULER_SIMD_BATCH_SIZE`: Override the default SIMD batch size
 * 
 * **Precision Control:**
 * - `EULER_DEFAULT_EPSILON`: Default epsilon for floating-point comparisons (default: 1e-6)
 * - `EULER_HIGH_PRECISION`: Use higher precision algorithms (may be slower)
 * 
 * **Matrix Storage:**
 * - `EULER_DEFAULT_COLUMN_MAJOR`: Make column-major the default storage order
 * - `EULER_FORCE_ROW_MAJOR`: Force all matrices to use row-major storage
 * 
 * **Debug Features:**
 * - `EULER_ENABLE_BOUNDS_CHECK`: Enable bounds checking for all container access
 * - `EULER_ENABLE_NAN_CHECK`: Check for NaN values in computations
 * - `EULER_TRACE_EXPRESSIONS`: Enable expression template debugging output
 * 
 * **Example Usage:**
 * @code{.cpp}
 * // Define before including euler.hh
 * #define EULER_DISABLE_SIMD       // Disable SIMD for debugging
 * #define EULER_ENABLE_BOUNDS_CHECK // Enable bounds checking
 * #define EULER_DEFAULT_EPSILON 1e-8 // Use tighter epsilon
 * 
 * #include <euler/euler.hh>
 * @endcode
 */

/**
 * @defgroup CoreModule Core Components
 * @brief Core functionality including types, traits, and error handling
 */

/**
 * @defgroup VectorModule Vector Operations
 * @brief Mathematical vectors and operations
 */

/**
 * @defgroup MatrixModule Matrix Operations
 * @brief Matrix class and operations
 */

/**
 * @defgroup AngleModule Angle Types
 * @brief Type-safe angle representation with automatic conversion
 */

/**
 * @defgroup QuaternionModule Quaternions
 * @brief Quaternion class for 3D rotations
 */

/**
 * @defgroup ComplexModule Complex Numbers
 * @brief Complex number support
 */

/**
 * @defgroup CoordinatesModule Coordinates
 * @brief 2D and 3D point types with projective coordinate support
 */

/**
 * @defgroup MathModule Mathematical Functions
 * @brief Trigonometric, exponential, and other mathematical functions
 */

/**
 * @defgroup DDAModule Digital Differential Analyzer
 * @brief Rasterization algorithms for lines, circles, curves, and more
 */

/**
 * @defgroup RandomModule Random Generation
 * @brief Random number generation for geometric objects
 */

#pragma once

// Configuration macro processing
// Check for user-defined configuration macros and set defaults

#ifndef EULER_DEFAULT_EPSILON
  #define EULER_DEFAULT_EPSILON 1e-6
#endif

#ifdef EULER_DEFAULT_COLUMN_MAJOR
  #define EULER_DEFAULT_STORAGE_ORDER true
#else
  #define EULER_DEFAULT_STORAGE_ORDER false
#endif

// Core components
#include <euler/core/types.hh>        // Basic type definitions (size_t, etc.)
#include <euler/core/traits.hh>       // Type traits and SFINAE helpers
#include <euler/core/error.hh>        // Error handling and assertions
#include <euler/core/expression.hh>   // Expression template base classes
#include <euler/core/simd.hh>         // SIMD support via xsimd
#include <euler/core/approx_equal.hh> // Floating-point comparison utilities

// Angles - Type-safe angle representation
#include <euler/angles/angle.hh>      // Base angle template
#include <euler/angles/degree.hh>     // Degree angle type
#include <euler/angles/radian.hh>     // Radian angle type
#include <euler/angles/angle_ops.hh>  // Angle operations and conversions

// Vectors - Fixed-size mathematical vectors
#include <euler/vector/vector.hh>         // Vector class template
#include <euler/vector/vector_expr.hh>    // Vector expression templates
#include <euler/vector/vector_ops.hh>     // Vector operations (dot, cross, etc.)
#include <euler/vector/vector_traits.hh>  // Vector type traits

// Matrices - Fixed-size matrices with expression templates
#include <euler/matrix/matrix.hh>         // Matrix class template
#include <euler/matrix/matrix_expr.hh>    // Matrix expression templates
#include <euler/matrix/matrix_ops.hh>     // Matrix operations (multiply, inverse, etc.)
#include <euler/matrix/matrix_view.hh>    // Matrix views and submatrices
#include <euler/matrix/specialized.hh>    // Specialized matrix operations

// Quaternions - Unit quaternions for 3D rotations
#include <euler/quaternion/quaternion.hh>     // Quaternion class
#include <euler/quaternion/quaternion_ops.hh> // Quaternion operations

// Complex numbers
#include <euler/complex/complex.hh>       // Complex number class
#include <euler/complex/complex_ops.hh>   // Complex operations

// Coordinates - Points and projective coordinates
#include <euler/coordinates/point2.hh>        // 2D point type
#include <euler/coordinates/point3.hh>        // 3D point type
#include <euler/coordinates/projective2.hh>   // 2D projective coordinates
#include <euler/coordinates/projective3.hh>   // 3D projective coordinates
#include <euler/coordinates/point_ops.hh>     // Point operations
#include <euler/coordinates/coord_transform.hh> // Coordinate transformations
#include <euler/coordinates/io.hh>            // I/O stream operators

// Mathematical functions
#include <euler/math/basic.hh>            // Basic math functions (abs, sqrt, etc.)
#include <euler/math/trigonometry.hh>     // Trigonometric functions

// Random number generation
#include <euler/random/random.hh>             // Core RNG interface
#include <euler/random/distributions.hh>      // Probability distributions
#include <euler/random/random_angle.hh>       // Random angle generation
#include <euler/random/random_complex.hh>     // Random complex numbers
#include <euler/random/random_vec.hh>         // Random vectors
#include <euler/random/random_quaternion.hh>  // Random quaternions

// DDA - Digital Differential Analyzer for rasterization
#include <euler/dda/dda.hh>

// I/O support - Pretty printing with stream operators
#include <euler/io/io.hh>

// Direct Ops
#include <euler/direct/direct_ops.hh>
/**
 * @namespace euler
 * @brief Main namespace for the Euler mathematical library
 * 
 * All Euler library components are contained within this namespace.
 * 
 * @note Configuration macros must be defined before including any Euler headers
 * to ensure consistent behavior across translation units.
 * 
 * @warning Mixing different configuration settings across translation units
 * may lead to ODR (One Definition Rule) violations.
 */