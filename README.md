# Euler Mathematical Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Documentation](https://img.shields.io/badge/docs-doxygen-brightgreen.svg)](https://your-docs-url.com)

A modern C++ mathematical library featuring expression templates, SIMD optimization, and compile-time safety for high-performance scientific computing.

## Features

- **Expression Templates**: Zero-overhead mathematical expressions with lazy evaluation
- **SIMD Optimization**: Automatic vectorization using xsimd for maximum performance
- **Type Safety**: Compile-time dimension checking and strong typing
- **Comprehensive Coverage**: Vectors, matrices, quaternions, complex numbers, and more
- **Direct Operations**: Bypass expression templates when needed for optimal performance
- **Header-Only**: Easy integration with no linking required

## Quick Start

```cpp
#include <euler/euler.hh>

using namespace euler;

int main() {
    // Vectors with compile-time dimensions
    vec3<float> v1(1.0f, 2.0f, 3.0f);
    vec3<float> v2(4.0f, 5.0f, 6.0f);
    
    // Expression templates - no temporaries
    auto v3 = v1 + 2.0f * v2;
    
    // Matrix operations
    mat3<float> rotation = mat3<float>::rotation_z(radians(45.0f));
    vec3<float> rotated = rotation * v1;
    
    // Quaternions for 3D rotations
    quatf q = quatf::from_axis_angle(vec3f(0, 0, 1), radians(90));
    
    // Complex numbers
    complex<double> z(3.0, 4.0);
    auto magnitude = abs(z);  // 5.0
    
    return 0;
}
```

## Installation

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.20 or higher
- Optional: xsimd for SIMD optimization
- Optional: Doxygen for documentation generation

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/euler.git
cd euler

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build and run tests
make
make test
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `EULER_BUILD_WITH_TESTS` | ON | Build unit tests |
| `EULER_BUILD_EXAMPLES` | ON | Build example programs |
| `EULER_BUILD_WITH_XSIMD` | ON | Enable SIMD optimization |
| `EULER_BUILD_DOCUMENTATION` | OFF | Generate Doxygen documentation |
| `EULER_DISABLE_SIMD` | OFF | Disable SIMD even if xsimd is available |
| `EULER_ENABLE_BOUNDS_CHECK` | OFF | Enable runtime bounds checking |

### Integration

Euler is header-only. Simply add the include directory to your project:

```cmake
add_subdirectory(euler)
target_link_libraries(your_target PRIVATE euler)
```

Or manually:

```cmake
target_include_directories(your_target PRIVATE ${EULER_INCLUDE_DIR})
```

## Components

### Core Types

- **Vectors**: Fixed-size mathematical vectors with dimensions 2-4
  ```cpp
  vec2<T>, vec3<T>, vec4<T>  // Common aliases
  vector<T, N>               // Generic N-dimensional
  ```

- **Matrices**: Row-major and column-major matrices
  ```cpp
  mat2<T>, mat3<T>, mat4<T>  // Square matrices
  matrix<T, M, N>            // M×N matrices
  ```

- **Quaternions**: Unit quaternions for 3D rotations
  ```cpp
  quaternion<T>
  quatf, quatd              // Float/double aliases
  ```

- **Complex Numbers**: Complex arithmetic with transcendental functions
  ```cpp
  complex<T>
  ```

- **Angles**: Type-safe angle representations
  ```cpp
  radian<T>, degree<T>
  radians(45.0f)           // Conversion helpers
  ```

### Expression Templates

Euler uses expression templates to eliminate temporaries:

```cpp
// No temporaries created - single loop at assembly level
vec3f result = a + b * c - d / 2.0f;

// Force evaluation when needed
vec3f forced = eval(a + b);
```

### Direct Operations

For performance-critical code, bypass expression templates:

```cpp
#include <euler/direct/direct_ops.hh>

using namespace euler::direct;

vec3f a, b, c;
add(a, b, c);        // c = a + b (no expression template)
mul(a, 2.0f, a);     // a = 2 * a (in-place)
float d = dot(a, b); // Direct dot product
```

### SIMD Support

Automatic SIMD optimization when xsimd is available:

```cpp
// Automatically uses SIMD instructions
vec4f a, b, c;
c = a + b;  // Uses SSE/AVX/NEON when available
```

## Mathematical Operations

### Vector Operations
- Arithmetic: `+`, `-`, `*`, `/`
- Geometric: `dot()`, `cross()`, `norm()`, `normalize()`
- Component-wise: `min()`, `max()`, `abs()`, `clamp()`

### Matrix Operations
- Arithmetic: `+`, `-`, `*`
- Linear algebra: `transpose()`, `inverse()`, `determinant()`
- Transformations: `rotation_x()`, `scale()`, `translation()`

### Quaternion Operations
- Multiplication (composition)
- Conversion to/from matrices
- Interpolation: `slerp()`, `nlerp()`
- Creation: `from_axis_angle()`, `from_euler()`

### Transcendental Functions
- Exponential: `exp()`, `log()`, `pow()`
- Trigonometric: `sin()`, `cos()`, `tan()`, `atan2()`
- Hyperbolic: `sinh()`, `cosh()`, `tanh()`

## Examples

See the `examples/` directory for more comprehensive examples:

- `01_vector_basics.cc` - Vector fundamentals
- `02_matrix_operations.cc` - Matrix mathematics
- `03_quaternion_rotations.cc` - 3D rotations with quaternions
- `04_angle_types.cc` - Type-safe angle handling
- `05_complex_numbers.cc` - Complex arithmetic
- `06_expression_templates.cc` - Understanding expression templates
- `07_random_generation.cc` - Random number generation
- `08_3d_graphics_pipeline.cc` - 3D graphics transformations
- `09_coordinates.cc` - Coordinate system transformations

## Performance

Euler is designed for maximum performance:

- **Zero-overhead abstractions** through expression templates
- **SIMD optimization** with automatic vectorization
- **Cache-friendly** memory layouts
- **Compile-time optimization** through constexpr and templates

Benchmarks show performance on par with hand-optimized code while maintaining readability and safety.

## Documentation

Full API documentation is available at [docs-url] or can be generated locally:

```bash
cmake -DEULER_BUILD_DOCUMENTATION=ON ..
make docs
# Open build/docs/html/index.html
```

## Testing

Euler includes comprehensive unit tests using doctest:

```bash
cd build
make test
# Or run directly
./test/unittest
```

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/euler.git

# Build with all features
cmake -DEULER_BUILD_WITH_TESTS=ON \
      -DEULER_BUILD_EXAMPLES=ON \
      -DEULER_BUILD_DOCUMENTATION=ON \
      -DCMAKE_BUILD_TYPE=Debug ..
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [xsimd](https://github.com/xtensor-stack/xsimd) for SIMD abstractions
- [doctest](https://github.com/onqtam/doctest) for unit testing


