# =============================================================================
# euler tests
# =============================================================================

include(CTest)

# Collect all test sources
set(EULER_TEST_SOURCES
    main.cc
    # Core tests
    core/types_test.cc
    core/expression_test.cc
    core/simd_test.cc
    core/traits_test.cc
    core/error_test.cc
    core/approx_equal_test.cc
    # Matrix tests
    matrix/matrix_test.cc
    matrix/matrix_layout_test.cc
    matrix/matrix_view_test.cc
    matrix/matrix_from_basis_test.cc
    matrix/matrix_ops_test.cc
    matrix/matrix_expr_test.cc
    matrix/square_matrix_ops_test.cc
    matrix/matrix_multiplication_chain_test.cc
    matrix/scalar_matrix_ops_test.cc
    matrix/transpose_multiplication_test.cc
    matrix/vector_matrix_ops_test.cc
    # Vector tests
    vector/vector_test.cc
    vector/vector_view_test.cc
    vector/vector_expr_test.cc
    vector/vector_expr_comprehensive_test.cc
    vector/vector_expr_simple_test.cc
    vector/vector_basic_all_dims_test.cc
    vector/vector_ops_all_dims_test.cc
    vector/vector_ops_expr_test.cc
    vector/vector_orthonormal_test.cc
    vector/scalar_vector_ops_test.cc
    # Quaternion tests
    quaternion/quaternion_test.cc
    quaternion/quaternion_ops_test.cc
    # Angles tests
    angles/angle_test.cc
    angles/angle_ops_test.cc
    angles/angle_loop_example.cc
    # Complex tests
    complex/complex_test.cc
    complex/complex_trig_test.cc
    # Coordinates tests
    coordinates/point2_test.cc
    coordinates/point3_test.cc
    coordinates/projective2_test.cc
    coordinates/projective3_test.cc
    coordinates/coord_transform_test.cc
    coordinates/coordinates_io_test.cc
    # DDA tests
    dda/test_line_iterator.cc
    dda/test_circle_iterator.cc
    dda/test_arc_iterator.cc
    dda/test_ellipse_iterator.cc
    dda/test_bezier_iterator.cc
    dda/test_bspline_iterator.cc
    dda/test_aa_iterators.cc
    dda/test_batched_iterators.cc
    dda/test_curve_iterator.cc
    dda/test_bspline_debug.cc
    dda/dda_io_test.cc
    # Random tests
    random/random_test.cc
    random/random_geom_test.cc
    random/random_angle_test.cc
    random/random_complex_test.cc
    random/random_quaternion_test.cc
    random/random_vec_test.cc
    random/distributions_test.cc
    # Math tests
    math/basic_test.cc
    math/basic_expr_test.cc
    math/basic_complex_test.cc
    math/trigonometry_test.cc
    math/trigonometry_expr_test.cc
    math/trigonometry_expr_integration_test.cc
    math/domain_check_test.cc
    # Direct operation tests
    direct/test_matrix_ops.cc
    direct/test_quaternion_ops.cc
    direct/test_transcendental_ops.cc
    direct/test_vector_ops.cc
    # Integration tests
    integration/comprehensive_ops_test.cc
    # I/O tests
    io/io_test.cc
)

# Create test executable
add_executable(euler_unittest ${EULER_TEST_SOURCES})

# Configure test target
neutrino_target_warnings(euler_unittest)
target_link_libraries(euler_unittest PRIVATE euler::euler doctest::doctest)

# Add sanitizer flags if enabled
if(NEUTRINO_EULER_ENABLE_SANITIZERS)
    if(NEUTRINO_COMPILER_IS_GCC OR NEUTRINO_COMPILER_IS_CLANG)
        target_compile_options(euler_unittest PRIVATE -fno-omit-frame-pointer)
        target_link_options(euler_unittest PRIVATE -fno-omit-frame-pointer)
    endif()
endif()

# Coverage configuration
if(NEUTRINO_EULER_ENABLE_COVERAGE)
    if(NEUTRINO_COMPILER_IS_GCC OR NEUTRINO_COMPILER_IS_CLANG)
        target_compile_options(euler_unittest PRIVATE --coverage)
        target_link_options(euler_unittest PRIVATE --coverage)
    endif()
endif()

# Register with CTest
add_test(NAME euler_all_tests COMMAND euler_unittest)

# Add individual test suites for granular testing
set(TEST_SUITES
    core
    matrix
    vector
    quaternion
    angles
    complex
    coordinates
    dda
    random
    math
    direct
    integration
    io
)

foreach(suite ${TEST_SUITES})
    add_test(
        NAME euler_${suite}_tests
        COMMAND euler_unittest --test-case="${suite}/*"
    )
    set_tests_properties(euler_${suite}_tests PROPERTIES
        LABELS ${suite}
        TIMEOUT 60
    )
endforeach()

# Add memory leak detection for valgrind
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE AND NOT WIN32)
    add_test(
        NAME euler_valgrind_test
        COMMAND ${VALGRIND_EXECUTABLE}
            --leak-check=full
            --track-origins=yes
            --error-exitcode=1
            --quiet
            $<TARGET_FILE:euler_unittest>
    )
    set_tests_properties(euler_valgrind_test PROPERTIES
        LABELS "valgrind;memory"
        TIMEOUT 300
    )
endif()

# Coverage target
if(NEUTRINO_EULER_ENABLE_COVERAGE)
    find_program(GCOV_EXECUTABLE gcov)
    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)

    if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        add_custom_target(coverage
            COMMAND ${LCOV_EXECUTABLE} --directory . --zerocounters
            COMMAND $<TARGET_FILE:euler_unittest>
            COMMAND ${LCOV_EXECUTABLE} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' '*/ext/*' '*/test/*' --output-file coverage.filtered.info
            COMMAND ${GENHTML_EXECUTABLE} coverage.filtered.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report"
        )
        add_dependencies(coverage euler_unittest)
    endif()
endif()

# Test discovery for IDEs
set_target_properties(euler_unittest PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Standalone test executables (have their own main())
set(STANDALONE_TESTS
    dda/test_subpixel_accuracy
    dda/test_subpixel_visual
    dda/test_arc_final
    test_matrix_simd
)

foreach(test ${STANDALONE_TESTS})
    get_filename_component(test_name ${test} NAME_WE)
    add_executable(${test_name} ${test}.cc)
    neutrino_target_warnings(${test_name})
    target_link_libraries(${test_name} PRIVATE euler::euler)

    # Add as a separate test
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "standalone"
        TIMEOUT 60
    )
endforeach()
