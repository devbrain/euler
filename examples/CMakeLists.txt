cmake_minimum_required(VERSION 3.20)

include(FetchContent)

# Example programs for Euler library
set(EXAMPLE_PROGRAMS
    01_vector_basics
    02_matrix_operations
    03_quaternion_rotations
    04_angle_types
    05_complex_numbers
    06_expression_templates
    07_random_generation
    08_3d_graphics_pipeline
    09_coordinates
    config_demo
    dda_demo
)

# Create executable for each example
foreach(example ${EXAMPLE_PROGRAMS})
    add_executable(${example} ${example}.cc)
    target_link_libraries(${example} PRIVATE euler::euler)
    
    # Set output directory for examples
    set_target_properties(${example} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endforeach()

# Create a custom target to build all examples
add_custom_target(examples
    DEPENDS ${EXAMPLE_PROGRAMS}
    COMMENT "Building all example programs"
)

# Optionally add examples to CTest
if(BUILD_TESTING)
    foreach(example ${EXAMPLE_PROGRAMS})
        add_test(
            NAME example_${example}
            COMMAND ${example}
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
        )
    endforeach()
endif()

# =============================================================================
# DDA ImGui Demo (requires SDL3)
# =============================================================================

option(EULER_BUILD_IMGUI_DEMO "Build DDA ImGui demo (requires SDL3)" ON)

if(EULER_BUILD_IMGUI_DEMO)
    # Fetch SDL3 using neutrino-cmake
    include(${NEUTRINO_CMAKE_DIR}/deps/SDL3.cmake)
    neutrino_fetch_sdl3()

    # Fetch ImGui from GitHub
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.92.1
    )
    FetchContent_MakeAvailable(imgui)

    # Set ImGui source directory
    set(IMGUI_DIR ${imgui_SOURCE_DIR})

    # ImGui core sources + SDL3 backend
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp
    )

    # Create the ImGui demo executable
    add_executable(dda_imgui_demo
        dda/dda_imgui_demo.cc
        ${IMGUI_SOURCES}
    )

    # Set include directories
    target_include_directories(dda_imgui_demo PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${CMAKE_SOURCE_DIR}/include
    )

    # Link libraries
    target_link_libraries(dda_imgui_demo PRIVATE
        SDL3::SDL3
        euler::euler
    )

    # Platform-specific settings
    if(UNIX AND NOT APPLE)
        find_package(OpenGL REQUIRED)
        target_link_libraries(dda_imgui_demo PRIVATE ${OPENGL_LIBRARIES} dl)
    elseif(APPLE)
        find_library(OPENGL_LIBRARY OpenGL)
        find_library(COCOA_LIBRARY Cocoa)
        find_library(IOKIT_LIBRARY IOKit)
        find_library(COREVIDEO_LIBRARY CoreVideo)
        target_link_libraries(dda_imgui_demo PRIVATE
            ${OPENGL_LIBRARY}
            ${COCOA_LIBRARY}
            ${IOKIT_LIBRARY}
            ${COREVIDEO_LIBRARY}
        )
    elseif(WIN32)
        target_link_libraries(dda_imgui_demo PRIVATE opengl32 imm32)
    endif()

    # Define SDL version for the source code
    target_compile_definitions(dda_imgui_demo PRIVATE SDL_VERSION_MACRO=3 EULER_HAS_XSIMD)

    # Set output directory
    set_target_properties(dda_imgui_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )

    # Add to examples target
    add_dependencies(examples dda_imgui_demo)

    message(STATUS "DDA ImGui demo enabled (SDL3)")
endif()