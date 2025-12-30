cmake_minimum_required(VERSION 3.20)

project(euler
    VERSION 1.0.0
    DESCRIPTION "Modern C++ linear algebra and graphics math library"
    HOMEPAGE_URL "https://github.com/devbrain/euler"
    LANGUAGES CXX
)

# =============================================================================
# Neutrino CMake Integration
# =============================================================================

include(FetchContent)

if(NOT DEFINED NEUTRINO_CMAKE_DIR)
    # Check for local neutrino-cmake first (sibling directory)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../neutrino-cmake/cmake/NeutrinoInit.cmake")
        set(NEUTRINO_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../neutrino-cmake/cmake")
    else()
        FetchContent_Declare(neutrino_cmake
            GIT_REPOSITORY https://github.com/devbrain/neutrino-cmake.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(neutrino_cmake)
        set(NEUTRINO_CMAKE_DIR "${neutrino_cmake_SOURCE_DIR}/cmake")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${NEUTRINO_CMAKE_DIR}")
include(NeutrinoInit)

# =============================================================================
# Options
# =============================================================================

neutrino_define_options(euler)

# Euler-specific options
option(NEUTRINO_EULER_ENABLE_XSIMD "Enable XSIMD support" ON)
option(NEUTRINO_EULER_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(NEUTRINO_EULER_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(NEUTRINO_EULER_DEVELOPER_MODE "Enable developer mode (strict warnings)" OFF)

# =============================================================================
# Developer mode settings
# =============================================================================

neutrino_is_top_level(_euler_is_top_level)
if(_euler_is_top_level AND NEUTRINO_EULER_DEVELOPER_MODE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Include static analysis configuration if available
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StaticAnalysis.cmake")
        include(cmake/StaticAnalysis.cmake)
    endif()
endif()

# =============================================================================
# Dependencies
# =============================================================================

# Failsafe - Error handling library (required)
include(${NEUTRINO_CMAKE_DIR}/deps/failsafe.cmake)
neutrino_fetch_failsafe()

# xsimd - SIMD library (optional)
if(NEUTRINO_EULER_ENABLE_XSIMD)
    include(${NEUTRINO_CMAKE_DIR}/deps/xsimd.cmake)
    neutrino_fetch_xsimd()
endif()

# Doctest - Unit testing framework (for tests)
if(NEUTRINO_EULER_BUILD_TESTS)
    include(${NEUTRINO_CMAKE_DIR}/deps/doctest.cmake)
    neutrino_fetch_doctest()
endif()

# Google Benchmark (for benchmarks)
if(NEUTRINO_EULER_BUILD_BENCHMARKS)
    include(${NEUTRINO_CMAKE_DIR}/deps/benchmark.cmake)
    neutrino_fetch_benchmark()
endif()

# SDL2 and ImGui for examples
if(NEUTRINO_EULER_BUILD_EXAMPLES)
    find_package(SDL2 QUIET)
    if(SDL2_FOUND)
        message(STATUS "[euler] Found SDL2, graphics examples will be built")
        include(${NEUTRINO_CMAKE_DIR}/deps/imgui.cmake)
        neutrino_fetch_imgui()
    else()
        message(STATUS "[euler] SDL2 not found, some graphics examples will be skipped")
    endif()
endif()

# =============================================================================
# Main library target (header-only)
# =============================================================================

add_library(euler INTERFACE)
add_library(euler::euler ALIAS euler)
add_library(neutrino::euler ALIAS euler)

target_compile_features(euler INTERFACE cxx_std_20)

target_include_directories(euler
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(euler INTERFACE neutrino::failsafe)

# XSIMD support
if(NEUTRINO_EULER_ENABLE_XSIMD)
    if(TARGET xsimd OR TARGET xsimd::xsimd)
        if(TARGET xsimd::xsimd)
            target_link_libraries(euler INTERFACE xsimd::xsimd)
        else()
            target_link_libraries(euler INTERFACE xsimd)
        endif()
        target_compile_definitions(euler INTERFACE EULER_HAS_XSIMD)
        message(STATUS "[euler] XSIMD support enabled")
    else()
        message(WARNING "[euler] XSIMD enabled but xsimd target not found")
    endif()
endif()

# Debug mode definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NEUTRINO_EULER_DEVELOPER_MODE)
    target_compile_definitions(euler INTERFACE EULER_DEBUG)
endif()

# User-defined configuration macros
foreach(MACRO IN ITEMS
    EULER_DISABLE_SIMD
    EULER_DISABLE_ENFORCE
    EULER_ENABLE_BOUNDS_CHECK)
    if(DEFINED ${MACRO})
        target_compile_definitions(euler INTERFACE ${MACRO})
    endif()
endforeach()

if(DEFINED EULER_DEFAULT_EPSILON)
    target_compile_definitions(euler INTERFACE EULER_DEFAULT_EPSILON=${EULER_DEFAULT_EPSILON})
endif()

# =============================================================================
# Tests
# =============================================================================

if(NEUTRINO_EULER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# =============================================================================
# Examples
# =============================================================================

if(NEUTRINO_EULER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Benchmarks
# =============================================================================

if(NEUTRINO_EULER_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# =============================================================================
# Documentation
# =============================================================================

if(NEUTRINO_EULER_BUILD_DOCS)
    find_package(Doxygen QUIET)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
        set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

        file(GLOB_RECURSE EULER_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

            add_custom_command(
                OUTPUT ${DOXYGEN_INDEX_FILE}
                COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                DEPENDS ${EULER_PUBLIC_HEADERS}
                MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM
            )

            add_custom_target(euler_docs ALL DEPENDS ${DOXYGEN_INDEX_FILE})

            if(NEUTRINO_EULER_INSTALL)
                install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    COMPONENT Documentation
                    OPTIONAL
                )
            endif()
        endif()
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================

# Check if failsafe was fetched via FetchContent (vs system-installed)
# We can't install euler if failsafe is a FetchContent dependency due to export set issues
FetchContent_GetProperties(failsafe)
if(NEUTRINO_EULER_INSTALL AND NOT failsafe_POPULATED)
    include(NeutrinoInstall)

    neutrino_install_headers(euler)

    neutrino_install_library(euler
        NAMESPACE neutrino::
        COMPATIBILITY SameMajorVersion
        DEPENDENCIES "find_dependency(failsafe REQUIRED)"
    )
endif()

# =============================================================================
# Summary
# =============================================================================

if(_euler_is_top_level)
    neutrino_print_options(euler)
    message(STATUS "  XSIMD Support:    ${NEUTRINO_EULER_ENABLE_XSIMD}")
    message(STATUS "  Developer Mode:   ${NEUTRINO_EULER_DEVELOPER_MODE}")
    message(STATUS "  Sanitizers:       ${NEUTRINO_EULER_ENABLE_SANITIZERS}")
    message(STATUS "  Coverage:         ${NEUTRINO_EULER_ENABLE_COVERAGE}")
    message(STATUS "")
endif()
