include(FetchContent)

if(NOT DEFINED PICNIX_USE_SYSTEM_LIBS)
  set(PICNIX_USE_SYSTEM_LIBS ON)
endif()

set(PICNIX_CATCH2_CONFIG "" CACHE FILEPATH "Full path to Catch2Config.cmake")
if(NOT PICNIX_CATCH2_CONFIG AND DEFINED ENV{PICNIX_CATCH2_CONFIG})
  set(PICNIX_CATCH2_CONFIG "$ENV{PICNIX_CATCH2_CONFIG}" CACHE FILEPATH
    "Full path to Catch2Config.cmake" FORCE)
endif()

function(_picnix_setup_catch2)
  if(TARGET Catch2::Catch2)
    return()
  endif()

  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

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

  if(PICNIX_USE_SYSTEM_LIBS)
    find_package(Catch2 3 CONFIG QUIET)
  endif()

  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 v3 not found; fetching from upstream.")
    set(CATCH_BUILD_TESTING OFF)
    set(CATCH_BUILD_EXAMPLES OFF)
    set(CATCH_BUILD_EXTRA_TESTS OFF)

    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)
  endif()
endfunction()

_picnix_setup_catch2()
