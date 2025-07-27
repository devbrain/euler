set(DOCTEST_WITH_TESTS OFF CACHE INTERNAL "")
set(DOCTEST_WITH_MAIN_IN_STATIC_LIB OFF CACHE INTERNAL "")
fetchcontent_declare(
        doctest
        GIT_REPOSITORY "https://github.com/onqtam/doctest.git"
        GIT_TAG v2.4.12
        GIT_PROGRESS TRUE
        UPDATE_COMMAND ""
)
fetchcontent_makeavailable(doctest)

# Create the expected alias
if(TARGET doctest AND NOT TARGET doctest::doctest)
    add_library(doctest::doctest ALIAS doctest)
endif()

