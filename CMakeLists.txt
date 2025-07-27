cmake_minimum_required(VERSION 3.20)

# Project definition with metadata
project(euler
    VERSION 1.0.0
    DESCRIPTION "Modern C++ linear algebra and graphics math library"
    HOMEPAGE_URL "https://github.com/devbrain/euler"
    LANGUAGES CXX
)

# Include modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(EULER_BUILD_TESTS "Build tests" ON)
option(EULER_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(EULER_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(EULER_BUILD_DOCUMENTATION "Build documentation with Doxygen" ${PROJECT_IS_TOP_LEVEL})
option(EULER_ENABLE_XSIMD "Enable XSIMD support" ON)
option(EULER_ENABLE_INSTALL "Enable installation" ${PROJECT_IS_TOP_LEVEL})
option(EULER_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(EULER_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(EULER_DEVELOPER_MODE "Enable developer mode (strict warnings)" ${PROJECT_IS_TOP_LEVEL})

# Developer mode settings
if(EULER_DEVELOPER_MODE)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    
    # Add strict warning flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            -Wall -Wextra -Wpedantic
            -Wcast-align -Wcast-qual
            -Wconversion -Wctor-dtor-privacy
            -Wdisabled-optimization -Wformat=2
            -Winit-self -Wmissing-declarations
            -Wmissing-include-dirs -Wold-style-cast
            -Woverloaded-virtual -Wredundant-decls
            -Wshadow -Wsign-conversion -Wsign-promo
            -Wstrict-overflow=5 -Wswitch-default
            -Wundef -Wunused
        )
    elseif(MSVC)
        add_compile_options(/W4 /permissive-)
    endif()
    
    # Include static analysis configuration
    include(cmake/StaticAnalysis.cmake)
endif()

# Add dependencies
add_subdirectory(ext)

# Create main library target
add_library(euler INTERFACE)
add_library(euler::euler ALIAS euler)

# Set include directories
target_include_directories(euler 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(euler INTERFACE failsafe::failsafe)

# XSIMD support
if(EULER_ENABLE_XSIMD)
    if(TARGET xsimd)
        target_link_libraries(euler INTERFACE xsimd)
        target_compile_definitions(euler INTERFACE EULER_HAS_XSIMD)
        message(STATUS "Linked xsimd to euler library")
    else()
        message(WARNING "XSIMD enabled but xsimd target not found")
    endif()
endif()

# Debug mode definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR EULER_DEVELOPER_MODE)
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

# Build subdirectories
if(EULER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(EULER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(EULER_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# Documentation
if(EULER_BUILD_DOCUMENTATION)
    find_package(Doxygen QUIET)
    
    if(DOXYGEN_FOUND)
        set(DOXYGEN_INPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
        set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
        
        file(GLOB_RECURSE EULER_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh)
        
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
        
        if(EULER_ENABLE_INSTALL)
            install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html 
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    COMPONENT Documentation
                    OPTIONAL)
        endif()
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()

# Installation
if(EULER_ENABLE_INSTALL)
    # Install headers
    install(DIRECTORY include/euler
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT Development
        FILES_MATCHING PATTERN "*.hh"
    )
    
    # Install targets
    install(TARGETS euler failsafe
        EXPORT eulerTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Install export set
    install(EXPORT eulerTargets
        FILE eulerTargets.cmake
        NAMESPACE euler::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/euler
        COMPONENT Development
    )
    
    # Generate and install config files
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/eulerConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/eulerConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/euler
    )
    
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/eulerConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/eulerConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/eulerConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/euler
        COMPONENT Development
    )
    
    # Export for build tree
    export(EXPORT eulerTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/eulerTargets.cmake
        NAMESPACE euler::
    )
    
    # Register package in user registry
    export(PACKAGE euler)
endif()

# CPack configuration
if(EULER_ENABLE_INSTALL AND PROJECT_IS_TOP_LEVEL)
    set(CPACK_PACKAGE_VENDOR "Euler Contributors")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    
    set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
    set(CPACK_SOURCE_IGNORE_FILES
        /.git/
        /build/
        /.vscode/
        /.idea/
        .gitignore
        .clang-format
        .clang-tidy
    )
    
    include(CPack)
endif()

# Summary
if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "")
    message(STATUS "Euler ${PROJECT_VERSION} Configuration Summary:")
    message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Build Tests: ${EULER_BUILD_TESTS}")
    message(STATUS "  Build Examples: ${EULER_BUILD_EXAMPLES}")
    message(STATUS "  Build Benchmarks: ${EULER_BUILD_BENCHMARKS}")
    message(STATUS "  Build Documentation: ${EULER_BUILD_DOCUMENTATION}")
    message(STATUS "  XSIMD Support: ${EULER_ENABLE_XSIMD}")
    message(STATUS "  Developer Mode: ${EULER_DEVELOPER_MODE}")
    message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "")
endif()