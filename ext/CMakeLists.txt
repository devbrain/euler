# =============================================================================
# euler dependencies
# =============================================================================

# Include neutrino dependency modules
include(deps/failsafe)
include(deps/xsimd)
include(deps/doctest)
include(deps/benchmark)

# =============================================================================
# Failsafe - Error handling library (required)
# =============================================================================

neutrino_fetch_failsafe()

# =============================================================================
# xsimd - SIMD library (optional)
# =============================================================================

if(NEUTRINO_EULER_ENABLE_XSIMD)
    neutrino_fetch_xsimd()
endif()

# =============================================================================
# Doctest - Unit testing framework (for tests)
# =============================================================================

if(NEUTRINO_EULER_BUILD_TESTS)
    neutrino_fetch_doctest()
endif()

# =============================================================================
# Google Benchmark (for benchmarks)
# =============================================================================

if(NEUTRINO_EULER_BUILD_BENCHMARKS)
    neutrino_fetch_benchmark()
endif()

# =============================================================================
# Optional: SDL2 for graphics examples
# =============================================================================

if(NEUTRINO_EULER_BUILD_EXAMPLES)
    find_package(SDL2 QUIET)
    if(SDL2_FOUND)
        message(STATUS "[euler] Found SDL2, graphics examples will be built")

        # Optional: ImGui for interactive examples
        include(deps/imgui)
        neutrino_fetch_imgui()
    else()
        message(STATUS "[euler] SDL2 not found, some graphics examples will be skipped")
    endif()
endif()
