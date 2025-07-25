# Euler Performance Benchmarks

This directory contains performance benchmarks for the Euler library, focusing on comparing operations with and without SIMD optimization, as well as regular vs batched operations.

## Building the Benchmarks

The benchmarks can be built either as part of the main Euler project or standalone.

### As Part of Main Project

When building Euler, the benchmarks are automatically built if you add `add_subdirectory(benchmark)` to the main CMakeLists.txt:

```bash
# From euler root directory
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run benchmarks
cd benchmark
./benchmark_matrix_vector
./benchmark_dda
./benchmark_simd

# Or use the convenience script
cd ../..
./benchmark/run_from_build.sh build
```

### Standalone Build

For standalone benchmark comparisons (SIMD vs no-SIMD):

```bash
cd benchmark
./run_benchmarks.sh all      # Run all configurations
./run_benchmarks.sh simd     # SIMD only
./run_benchmarks.sh no-simd  # Scalar only
./run_benchmarks.sh clean    # Clean build artifacts
```

### Manual Build Options

```bash
# Build with SIMD (default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build without SIMD for comparison
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_NO_SIMD=ON

# Build with profiling support
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING=ON
```

## Benchmark Suites

### 1. Matrix/Vector Operations (`benchmark_matrix_vector`)

Tests fundamental linear algebra operations:
- Vector dot product
- Vector cross product  
- Matrix multiplication (4x4)
- Matrix-vector multiplication
- Vector normalization

Compares scalar implementations against Euler's potentially SIMD-optimized versions.

### 2. DDA Operations (`benchmark_dda`)

Tests rasterization algorithms:
- Line rasterization (regular vs batched)
- Antialiased line rasterization  
- Cubic Bezier curve rasterization
- Pixel processing (individual vs batch)
- Span operations

Focuses on comparing regular iterators with batched iterators that improve cache utilization.

### 3. SIMD Operations (`benchmark_simd`)

Direct comparison of scalar vs SIMD implementations:
- Array operations (add, multiply, FMA)
- Array reductions (sum)
- Bezier curve batch evaluation
- Transcendental functions (sqrt)

Shows the raw performance difference when SIMD is available.

## Optimization Flags

The benchmarks are compiled with aggressive optimization:
- `-O3` optimization level
- `-march=native` for CPU-specific optimizations
- `-ffast-math` for faster floating-point operations
- `-funroll-loops` for loop unrolling
- `-ftree-vectorize` for auto-vectorization

## Understanding Results

Each benchmark reports:
- **Min/Max/Mean/Median**: Statistical summary of execution times
- **StdDev**: Standard deviation showing consistency
- **Iterations**: Number of test runs
- **Speedup**: Performance improvement ratio

### Expected Results

With SIMD enabled, typical speedups are:
- Vector operations: 2-4x
- Matrix operations: 2-3x  
- Array operations: 3-4x (depending on SIMD width)
- Batched operations: 1.5-2x (from better cache usage)

## Profiling

To enable profiling support:
```bash
cmake .. -DENABLE_PROFILING=ON
```

This adds appropriate compiler flags for profiling tools like gprof or perf.

## Tips for Best Performance

1. **Enable SIMD**: Ensure xsimd is available and detected
2. **Use Native Architecture**: Build with `-march=native`
3. **Use Batched Operations**: For processing many pixels/primitives
4. **Align Data**: Ensure data alignment for SIMD operations
5. **Profile First**: Measure before optimizing

## Troubleshooting

### SIMD Not Detected
Check the output of `benchmark_simd` which reports available SIMD extensions.

### Inconsistent Results
- Ensure no other CPU-intensive processes are running
- Disable CPU frequency scaling for consistent results
- Run benchmarks multiple times and average results

### Build Errors
- Ensure Euler library is properly installed
- Check CMake can find the euler package
- Verify C++17 compiler support