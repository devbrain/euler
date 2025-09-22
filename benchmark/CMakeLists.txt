# Benchmark configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Release build with O3 optimization for benchmarks
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add aggressive optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")
    
    # Additional optimization flags
    add_compile_options(
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -fomit-frame-pointer
    )
    
    # Enable all warnings but disable for benchmarks
    add_compile_options(-w)
    
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /GL /DNDEBUG")
    add_compile_options(/fp:fast)
endif()

# Benchmark executables
set(BENCHMARK_SOURCES
    benchmark_matrix_vector.cc
    benchmark_dda.cc
    benchmark_simd.cc
)

# Create executables for each benchmark
foreach(benchmark_source ${BENCHMARK_SOURCES})
    get_filename_component(benchmark_name ${benchmark_source} NAME_WE)
    add_executable(${benchmark_name} ${benchmark_source})
    
    target_link_libraries(${benchmark_name} 
        PRIVATE 
            euler
    )
    
    # Ensure benchmarks can find the header
    target_include_directories(${benchmark_name} 
        PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Enable interprocedural optimization if available
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported)
        if(ipo_supported)
            set_target_properties(${benchmark_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    endif()
endforeach()

# Combined benchmark executable
add_executable(benchmark_all benchmark_all.cc)
target_link_libraries(benchmark_all 
    PRIVATE 
        euler
)
target_include_directories(benchmark_all 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create a custom target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "Running all benchmarks..."
    COMMAND ${CMAKE_COMMAND} -E echo "======================="
)

foreach(benchmark_source ${BENCHMARK_SOURCES})
    get_filename_component(benchmark_name ${benchmark_source} NAME_WE)
    add_custom_command(TARGET run_benchmarks
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "Running ${benchmark_name}..."
        COMMAND ${CMAKE_COMMAND} -E echo "----------------------------------------"
        COMMAND $<TARGET_FILE:${benchmark_name}>
    )
endforeach()

# Print configuration info
message(STATUS "Benchmark Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  C++ Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  Additional flags: ${CMAKE_CXX_FLAGS}")

# Option to enable profiling
option(ENABLE_PROFILING "Enable profiling flags" OFF)
if(ENABLE_PROFILING)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-pg)
        add_link_options(-pg)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fprofile-instr-generate)
        add_link_options(-fprofile-instr-generate)
    endif()
endif()

# Option to compare with/without SIMD
option(BENCHMARK_NO_SIMD "Build benchmarks without SIMD for comparison" OFF)
if(BENCHMARK_NO_SIMD)
    add_compile_definitions(XSIMD_DISABLED)
    message(STATUS "Building benchmarks WITHOUT SIMD support for comparison")
endif()