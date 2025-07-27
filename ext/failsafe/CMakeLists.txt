set(FAILSAFE_BUILD_WITH_TESTS    CACHE INTERNAL OFF)
set(FAILSAFE_BUILD_EXAMPLES      CACHE INTERNAL OFF)
set(FAILSAFE_BUILD_DOCUMENTATION OFF CACHE INTERNAL OFF)

fetchcontent_declare(
            FAILSAFE
            GIT_REPOSITORY "https://github.com/devbrain/failsafe.git"
            GIT_TAG master
            GIT_PROGRESS TRUE
            UPDATE_COMMAND ""
    )
fetchcontent_makeavailable(FAILSAFE)

