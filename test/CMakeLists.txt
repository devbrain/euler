if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(EULER_WARNING_FLAGS
            -Wall           # All standard warnings
            -Wextra         # Extra warnings
            -Wpedantic      # Strict ISO C++ compliance
            -Wcast-align    # Warn about potential performance problem casts
            -Wcast-qual     # Warn about casts that remove qualifiers
            -Wconversion    # Warn about type conversions that may lose data
            -Wdouble-promotion  # Warn about float to double promotion
            -Wformat=2      # Additional format string warnings
            -Wnon-virtual-dtor  # Warn about non-virtual destructors
            -Wold-style-cast    # Warn about C-style casts
            -Woverloaded-virtual # Warn about overloaded virtual functions
            -Wshadow        # Warn about variable shadowing
            -Wsign-conversion   # Warn about sign conversions
            -Wundef         # Warn about undefined identifiers in #if
            -Wunused        # Warn about unused entities
            -Wzero-as-null-pointer-constant # Warn about using 0 as nullptr
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND EULER_WARNING_FLAGS
                -Wlogical-op    # Warn about logical operations being used where bitwise were probably wanted
                -Wuseless-cast  # Warn about useless casts
        )
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(EULER_WARNING_FLAGS
            /W4             # Highest warning level
            /permissive-    # Strict standard conformance
            /Zc:__cplusplus # Report correct __cplusplus value
    )
endif ()

add_executable(euler_unittest
        main.cc
        core/types_test.cc
        core/expression_test.cc
        core/simd_test.cc
        core/traits_test.cc
        core/error_test.cc
        core/approx_equal_test.cc
        matrix/matrix_test.cc
        matrix/matrix_layout_test.cc
        matrix/matrix_view_test.cc
        matrix/matrix_from_basis_test.cc
        matrix/matrix_ops_test.cc
        matrix/matrix_expr_test.cc
        matrix/square_matrix_ops_test.cc
        matrix/matrix_multiplication_chain_test.cc
        matrix/transpose_multiplication_test.cc
        matrix/vector_matrix_ops_test.cc
        vector/vector_test.cc
        vector/vector_view_test.cc
        vector/vector_expr_test.cc
        vector/vector_expr_simple_test.cc
        vector/vector_expr_comprehensive_test.cc
        vector/vector_ops_expr_test.cc
        vector/vector_orthonormal_test.cc
        vector/vector_ops_all_dims_test.cc
        vector/vector_basic_all_dims_test.cc
        vector/scalar_vector_ops_test.cc
        matrix/scalar_matrix_ops_test.cc
        integration/comprehensive_ops_test.cc
        angles/angle_test.cc
        angles/angle_ops_test.cc
        angles/angle_loop_example.cc
        math/trigonometry_test.cc
        math/trigonometry_expr_test.cc
        math/trigonometry_expr_integration_test.cc
        complex/complex_test.cc
        complex/complex_trig_test.cc
        math/basic_test.cc
        math/basic_complex_test.cc
        math/basic_expr_test.cc
        math/domain_check_test.cc
        quaternion/quaternion_test.cc
        quaternion/quaternion_ops_test.cc
        random/random_test.cc
        random/distributions_test.cc
        random/random_angle_test.cc
        random/random_complex_test.cc
        random/random_quaternion_test.cc
        random/random_vec_test.cc
        random/random_geom_test.cc
        io/io_test.cc
        coordinates/point2_test.cc
        coordinates/point3_test.cc
        coordinates/projective2_test.cc
        coordinates/projective3_test.cc
        coordinates/coord_transform_test.cc
        coordinates/coordinates_io_test.cc
        dda/test_line_iterator.cc
        dda/test_circle_iterator.cc
        dda/test_ellipse_iterator.cc
        dda/test_curve_iterator.cc
        dda/test_bezier_iterator.cc
        dda/test_bspline_iterator.cc
        dda/test_arc_iterator.cc
        dda/test_aa_iterators.cc
        dda/test_batched_iterators.cc
        dda/dda_io_test.cc
        direct/test_vector_ops.cc
        direct/test_matrix_ops.cc
        direct/test_quaternion_ops.cc
        direct/test_transcendental_ops.cc
)

target_link_libraries(euler_unittest PRIVATE euler doctest::doctest)
target_compile_options(euler_unittest PRIVATE ${EULER_WARNING_FLAGS})

# Enable debug checks for tests
target_compile_definitions(euler_unittest PRIVATE EULER_DEBUG)

