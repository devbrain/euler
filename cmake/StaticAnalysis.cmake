# Static Analysis Configuration

# Find clang-tidy
find_program(CLANG_TIDY_EXECUTABLE
    NAMES clang-tidy
    DOC "Path to clang-tidy executable"
)

# Find cppcheck
find_program(CPPCHECK_EXECUTABLE
    NAMES cppcheck
    DOC "Path to cppcheck executable"
)

# Find include-what-you-use
find_program(IWYU_EXECUTABLE
    NAMES include-what-you-use iwyu
    DOC "Path to include-what-you-use executable"
)

# Find clang-format
find_program(CLANG_FORMAT_EXECUTABLE
    NAMES clang-format
    DOC "Path to clang-format executable"
)

# Function to enable clang-tidy for a target
function(euler_enable_clang_tidy target)
    if(CLANG_TIDY_EXECUTABLE)
        set_target_properties(${target} PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};-p=${CMAKE_BINARY_DIR}"
        )
    endif()
endfunction()

# Function to enable cppcheck for a target
function(euler_enable_cppcheck target)
    if(CPPCHECK_EXECUTABLE)
        set_target_properties(${target} PROPERTIES
            CXX_CPPCHECK "${CPPCHECK_EXECUTABLE};--enable=all;--suppress=missingIncludeSystem"
        )
    endif()
endfunction()

# Function to enable include-what-you-use for a target
function(euler_enable_iwyu target)
    if(IWYU_EXECUTABLE)
        set_target_properties(${target} PROPERTIES
            CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXECUTABLE}"
        )
    endif()
endfunction()

# Create static analysis targets
if(EULER_DEVELOPER_MODE AND PROJECT_IS_TOP_LEVEL)
    # Get all header files
    file(GLOB_RECURSE EULER_HEADERS
        ${CMAKE_SOURCE_DIR}/include/*.hh
    )
    
    # Get all source files
    file(GLOB_RECURSE EULER_SOURCES
        ${CMAKE_SOURCE_DIR}/test/*.cc
        ${CMAKE_SOURCE_DIR}/examples/*.cc
        ${CMAKE_SOURCE_DIR}/benchmark/*.cc
    )
    
    # Clang-tidy target
    if(CLANG_TIDY_EXECUTABLE)
        add_custom_target(clang-tidy
            COMMAND ${CLANG_TIDY_EXECUTABLE}
                -p ${CMAKE_BINARY_DIR}
                ${EULER_HEADERS}
                ${EULER_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-tidy"
            VERBATIM
        )
        
        # Clang-tidy fix target
        add_custom_target(clang-tidy-fix
            COMMAND ${CLANG_TIDY_EXECUTABLE}
                -p ${CMAKE_BINARY_DIR}
                -fix
                -fix-errors
                ${EULER_HEADERS}
                ${EULER_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-tidy with fixes"
            VERBATIM
        )
    else()
        message(STATUS "clang-tidy not found, static analysis targets will not be available")
    endif()
    
    # Cppcheck target
    if(CPPCHECK_EXECUTABLE)
        add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --suppress=missingIncludeSystem
                --project=${CMAKE_BINARY_DIR}/compile_commands.json
                --error-exitcode=1
                --inline-suppr
                --quiet
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running cppcheck"
            VERBATIM
        )
    else()
        message(STATUS "cppcheck not found, cppcheck target will not be available")
    endif()
    
    # Format target
    if(CLANG_FORMAT_EXECUTABLE)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXECUTABLE}
                -i
                ${EULER_HEADERS}
                ${EULER_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Formatting code with clang-format"
            VERBATIM
        )
        
        # Format check target
        add_custom_target(format-check
            COMMAND ${CLANG_FORMAT_EXECUTABLE}
                --dry-run
                --Werror
                ${EULER_HEADERS}
                ${EULER_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Checking code format with clang-format"
            VERBATIM
        )
    else()
        message(STATUS "clang-format not found, format targets will not be available")
    endif()
    
    # Combined static analysis target
    add_custom_target(static-analysis)
    if(TARGET clang-tidy)
        add_dependencies(static-analysis clang-tidy)
    endif()
    if(TARGET cppcheck)
        add_dependencies(static-analysis cppcheck)
    endif()
    if(TARGET format-check)
        add_dependencies(static-analysis format-check)
    endif()
endif()