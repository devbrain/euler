include(FetchContent)

# FetchContent configuration
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)

# Option to prefer system packages
option(EULER_PREFER_SYSTEM_PACKAGES "Prefer system-installed packages over FetchContent" ON)

# Helper function to find or fetch dependencies
function(euler_find_or_fetch package_name)
    set(options REQUIRED)
    set(oneValueArgs VERSION GIT_REPOSITORY GIT_TAG TARGET_NAME)
    set(multiValueArgs FIND_PACKAGE_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # First try to find the package if system packages are preferred
    if(EULER_PREFER_SYSTEM_PACKAGES)
        find_package(${package_name} ${ARG_VERSION} QUIET ${ARG_FIND_PACKAGE_ARGS})
    endif()
    
    if(${package_name}_FOUND)
        message(STATUS "Found system ${package_name} version ${${package_name}_VERSION}")
    else()
        if(ARG_REQUIRED OR NOT EULER_PREFER_SYSTEM_PACKAGES)
            message(STATUS "Fetching ${package_name} ${ARG_GIT_TAG}")
            FetchContent_Declare(
                ${package_name}
                GIT_REPOSITORY ${ARG_GIT_REPOSITORY}
                GIT_TAG ${ARG_GIT_TAG}
                GIT_SHALLOW TRUE
                GIT_PROGRESS TRUE
            )
            FetchContent_MakeAvailable(${package_name})
        endif()
    endif()
endfunction()

# Doctest - Unit testing framework
if(EULER_BUILD_TESTS)
    # Check if we should use the old subdirectory approach
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/doctest/CMakeLists.txt)
        add_subdirectory(doctest)
    else()
        # Use the new FetchContent approach
        euler_find_or_fetch(doctest
            VERSION 2.4.11
            GIT_REPOSITORY https://github.com/onqtam/doctest.git
            GIT_TAG v2.4.11
            TARGET_NAME doctest::doctest
        )
        
        # Create alias if doctest was fetched
        if(TARGET doctest AND NOT TARGET doctest::doctest)
            add_library(doctest::doctest ALIAS doctest)
        endif()
    endif()
endif()

# Failsafe - Error handling library
# Note: This is a local dependency, so we always use the bundled version
if(NOT TARGET failsafe)
    add_subdirectory(failsafe)
endif()

# Create alias for consistent naming
if(TARGET failsafe AND NOT TARGET failsafe::failsafe)
    add_library(failsafe::failsafe ALIAS failsafe)
endif()

# xsimd - SIMD library
if(EULER_ENABLE_XSIMD)
    # Check if we should use the subdirectory approach
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xsimd/CMakeLists.txt)
        add_subdirectory(xsimd)
    else()
        euler_find_or_fetch(xsimd
            VERSION 11.1.0
            GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
            GIT_TAG 11.1.0
            TARGET_NAME xsimd
        )
    endif()
endif()

# Google Benchmark - Microbenchmarking library
if(EULER_BUILD_BENCHMARKS)
    # Disable benchmark's own tests
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    
    euler_find_or_fetch(benchmark
        VERSION 1.8.0
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
        TARGET_NAME benchmark::benchmark
    )
endif()

# Optional: SDL2 for graphics examples
if(EULER_BUILD_EXAMPLES)
    find_package(SDL2 QUIET)
    if(SDL2_FOUND)
        message(STATUS "Found SDL2, graphics examples will be built")
    else()
        message(STATUS "SDL2 not found, some graphics examples will be skipped")
    endif()
endif()

# Optional: ImGui for interactive examples
if(EULER_BUILD_EXAMPLES AND SDL2_FOUND)
    euler_find_or_fetch(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.0
        TARGET_NAME imgui
    )
    
    # If ImGui was fetched, we need to create a library target
    if(imgui_POPULATED)
        add_library(imgui STATIC
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp
        )
        target_include_directories(imgui PUBLIC 
            ${imgui_SOURCE_DIR}
            ${imgui_SOURCE_DIR}/backends
        )
        target_link_libraries(imgui PUBLIC SDL2::SDL2)
        
        # Disable warnings for third-party library
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(imgui PRIVATE -w)
        elseif(MSVC)
            target_compile_options(imgui PRIVATE /w)
        endif()
    endif()
endif()

# Summary of dependencies
if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "")
    message(STATUS "Euler Dependencies Summary:")
    message(STATUS "  Failsafe: bundled")
    if(EULER_BUILD_TESTS)
        if(doctest_FOUND)
            message(STATUS "  Doctest: system (${doctest_VERSION})")
        else()
            message(STATUS "  Doctest: fetched")
        endif()
    endif()
    if(EULER_ENABLE_XSIMD)
        if(xsimd_FOUND)
            message(STATUS "  xsimd: system (${xsimd_VERSION})")
        else()
            message(STATUS "  xsimd: fetched")
        endif()
    endif()
    if(EULER_BUILD_BENCHMARKS)
        if(benchmark_FOUND)
            message(STATUS "  Google Benchmark: system (${benchmark_VERSION})")
        else()
            message(STATUS "  Google Benchmark: fetched")
        endif()
    endif()
    message(STATUS "")
endif()