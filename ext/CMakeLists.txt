include(FetchContent)

set(FETCHCONTENT_QUIET off)

if(EULER_BUILD_WITH_TESTS)
    # First try to find doctest
    find_package(doctest QUIET)
    
    if(NOT doctest_FOUND)
        # If not found, fetch it
        message(STATUS "doctest not found, fetching as fallback")
        add_subdirectory(doctest)
    else()
        message(STATUS "Found installed doctest")
    endif()
endif()

add_subdirectory(failsafe)

if(EULER_BUILD_WITH_XSIMD)
    # First try to find xsimd
    find_package(xsimd QUIET)
    
    if(NOT xsimd_FOUND)
        # If not found, fetch it
        message(STATUS "xsimd not found, fetching as fallback")
        add_subdirectory(xsimd)
    else()
        message(STATUS "Found installed xsimd")
    endif()
endif()




