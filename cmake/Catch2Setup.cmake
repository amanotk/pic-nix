include(FetchContent)

set(PICNIX_CATCH2_CONFIG "" CACHE FILEPATH "Full path to Catch2Config.cmake")
if(NOT PICNIX_CATCH2_CONFIG AND DEFINED ENV{PICNIX_CATCH2_CONFIG})
  set(PICNIX_CATCH2_CONFIG "$ENV{PICNIX_CATCH2_CONFIG}" CACHE FILEPATH
    "Full path to Catch2Config.cmake" FORCE)
endif()

if(PICNIX_CATCH2_CONFIG)
  if(EXISTS "${PICNIX_CATCH2_CONFIG}")
    get_filename_component(_catch2_dir "${PICNIX_CATCH2_CONFIG}" DIRECTORY)
    set(Catch2_DIR "${_catch2_dir}")
  else()
    message(FATAL_ERROR
      "PICNIX_CATCH2_CONFIG was set but does not exist: "
      "${PICNIX_CATCH2_CONFIG}"
    )
  endif()
endif()

find_package(Catch2 3 CONFIG QUIET)

if(NOT Catch2_FOUND)
  message(STATUS "Catch2 v3 not found; fetching from upstream.")
  set(CATCH2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(CATCH2_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CATCH2_BUILD_EXTRA_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)
endif()
