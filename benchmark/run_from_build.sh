#!/bin/bash

# Simple benchmark runner for when benchmarks are built as part of the main project
# Usage: ./run_from_build.sh <path-to-build-dir>

set -e

BUILD_DIR="${1:-../build}"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Usage: $0 <path-to-build-dir>"
    echo "Example: $0 ../build"
    exit 1
fi

BENCHMARK_DIR="$BUILD_DIR/benchmark"

if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Error: Benchmark directory not found: $BENCHMARK_DIR"
    echo "Make sure you've built the project with benchmarks enabled"
    exit 1
fi

echo "Euler Performance Benchmarks"
echo "============================"
echo ""

cd "$BENCHMARK_DIR"

# Run each benchmark if it exists
for bench in benchmark_matrix_vector benchmark_dda benchmark_simd; do
    if [ -x "./$bench" ]; then
        echo "Running $bench..."
        echo "----------------------------------------"
        ./$bench
        echo ""
    else
        echo "Warning: $bench not found or not executable"
    fi
done

echo "All benchmarks completed!"