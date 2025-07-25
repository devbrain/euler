#!/bin/bash

# Benchmark runner script for Euler library
# This script builds and runs benchmarks with various configurations

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build benchmarks
build_benchmarks() {
    local config=$1
    local build_subdir="${BUILD_DIR}/${config}"
    
    print_info "Building benchmarks with configuration: ${config}"
    
    mkdir -p "${build_subdir}"
    cd "${build_subdir}"
    
    # Configure based on the build type
    case ${config} in
        "simd")
            cmake .. -DCMAKE_BUILD_TYPE=Release
            ;;
        "no-simd")
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_NO_SIMD=ON
            ;;
        "debug")
            cmake .. -DCMAKE_BUILD_TYPE=Debug
            ;;
        *)
            print_error "Unknown configuration: ${config}"
            exit 1
            ;;
    esac
    
    # Build with parallel jobs
    cmake --build . --parallel $(nproc)
    
    cd "${SCRIPT_DIR}"
}

# Function to run benchmarks
run_benchmark_suite() {
    local config=$1
    local build_subdir="${BUILD_DIR}/${config}"
    
    print_info "Running benchmarks for configuration: ${config}"
    
    if [ ! -d "${build_subdir}" ]; then
        print_error "Build directory not found: ${build_subdir}"
        return 1
    fi
    
    cd "${build_subdir}"
    
    # Run each benchmark
    for bench in benchmark_matrix_vector benchmark_dda benchmark_simd; do
        if [ -f "./${bench}" ]; then
            print_info "Running ${bench}..."
            ./${bench} || print_warn "${bench} failed"
            echo ""
        else
            print_warn "${bench} not found"
        fi
    done
    
    cd "${SCRIPT_DIR}"
}

# Function to compare results
compare_results() {
    print_info "Comparing SIMD vs No-SIMD results..."
    
    # This is a placeholder - in a real scenario, you would parse
    # and compare the benchmark outputs
    echo "To properly compare results, pipe the output to files and analyze:"
    echo "  ./run_benchmarks.sh simd > results_simd.txt"
    echo "  ./run_benchmarks.sh no-simd > results_no_simd.txt"
}

# Main script
main() {
    local mode=${1:-"all"}
    
    case ${mode} in
        "all")
            print_info "Running full benchmark suite"
            build_benchmarks "simd"
            build_benchmarks "no-simd"
            
            echo -e "\n${GREEN}=== SIMD Enabled ===${NC}"
            run_benchmark_suite "simd"
            
            echo -e "\n${GREEN}=== SIMD Disabled ===${NC}"
            run_benchmark_suite "no-simd"
            
            compare_results
            ;;
        "simd")
            build_benchmarks "simd"
            run_benchmark_suite "simd"
            ;;
        "no-simd")
            build_benchmarks "no-simd"
            run_benchmark_suite "no-simd"
            ;;
        "build-only")
            build_benchmarks "simd"
            build_benchmarks "no-simd"
            ;;
        "clean")
            print_info "Cleaning build directory"
            rm -rf "${BUILD_DIR}"
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [mode]"
            echo "Modes:"
            echo "  all        - Build and run all configurations (default)"
            echo "  simd       - Build and run with SIMD enabled"
            echo "  no-simd    - Build and run with SIMD disabled"
            echo "  build-only - Build all configurations without running"
            echo "  clean      - Clean build directory"
            echo "  help       - Show this help message"
            ;;
        *)
            print_error "Unknown mode: ${mode}"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Check if we're in the correct directory
if [ ! -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found in ${SCRIPT_DIR}"
    print_error "Please run this script from the benchmark directory"
    exit 1
fi

# Run main function with all arguments
main "$@"