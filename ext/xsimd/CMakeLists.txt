FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG        13.2.0         # or whichever release you prefer
)
FetchContent_MakeAvailable(xsimd)

# Suppress warnings from xsimd headers
if(TARGET xsimd)
    get_target_property(XSIMD_INCLUDE_DIRS xsimd INTERFACE_INCLUDE_DIRECTORIES)
    if(XSIMD_INCLUDE_DIRS)
        # Mark xsimd headers as system headers to suppress warnings
        set_target_properties(xsimd PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${XSIMD_INCLUDE_DIRS}")
    endif()
endif()
