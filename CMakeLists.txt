cmake_minimum_required(VERSION 3.20)
project(euler)

set(CMAKE_CXX_STANDARD 17)

# Build options
option(EULER_BUILD_WITH_TESTS "Build tests" ON)
option(EULER_BUILD_EXAMPLES "Build examples" ON)
option(EULER_BUILD_WITH_XSIMD "Build with XSIMD support" ON)
option(EULER_BUILD_DOCUMENTATION "Build documentation with Doxygen" OFF)

add_subdirectory(ext)

add_library(euler INTERFACE)
target_include_directories(euler INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(euler INTERFACE failsafe)

if(EULER_BUILD_WITH_XSIMD AND TARGET xsimd)
    target_link_libraries(euler INTERFACE xsimd)
endif()

# Define EULER_HAS_XSIMD when xsimd is available and SIMD is not disabled
if(EULER_BUILD_WITH_XSIMD AND TARGET xsimd AND NOT DEFINED EULER_DISABLE_SIMD)
    target_compile_definitions(euler INTERFACE EULER_HAS_XSIMD)
endif()

# Pass through user-defined configuration macros
if(DEFINED EULER_DISABLE_SIMD)
    target_compile_definitions(euler INTERFACE EULER_DISABLE_SIMD)
endif()

if(DEFINED EULER_DEFAULT_EPSILON)
    target_compile_definitions(euler INTERFACE EULER_DEFAULT_EPSILON=${EULER_DEFAULT_EPSILON})
endif()

if(DEFINED EULER_DISABLE_ENFORCE)
    target_compile_definitions(euler INTERFACE EULER_DISABLE_ENFORCE)
endif()

if(DEFINED EULER_ENABLE_BOUNDS_CHECK)
    target_compile_definitions(euler INTERFACE EULER_ENABLE_BOUNDS_CHECK)
endif()

if(EULER_BUILD_WITH_TESTS)
    add_subdirectory(test)
endif()

if(EULER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(benchmark)

# Documentation generation with Doxygen
if(EULER_BUILD_DOCUMENTATION)
    find_package(Doxygen QUIET)
    
    if(DOXYGEN_FOUND)
        message(STATUS "Doxygen found: ${DOXYGEN_EXECUTABLE}")
        
        # Set Doxygen input and output directories
        set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR})
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
        set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
        
        # Find all public header files
        file(GLOB_RECURSE EULER_PUBLIC_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hh)
        
        # Configure Doxyfile
        configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
        
        # Add custom target for documentation
        add_custom_command(
            OUTPUT ${DOXYGEN_INDEX_FILE}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${EULER_PUBLIC_HEADERS}
            MAIN_DEPENDENCY ${CMAKE_BINARY_DIR}/Doxyfile
            COMMENT "Generating documentation with Doxygen"
            VERBATIM
        )
        
        add_custom_target(docs ALL DEPENDS ${DOXYGEN_INDEX_FILE})
        
        # Install documentation
        install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html 
                DESTINATION share/doc/euler
                OPTIONAL)
                
        message(STATUS "Documentation will be built in: ${DOXYGEN_OUTPUT_DIR}")
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()

