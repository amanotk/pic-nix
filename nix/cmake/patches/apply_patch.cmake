# Idempotent patch applicator called from FetchContent PATCH_COMMAND.
# Usage: cmake -DSOURCE_DIR=<source> -P apply_patch.cmake
# The .patch file must live in the same directory as this script.

cmake_minimum_required(VERSION 3.20)

set(PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/xtensor-0.24.7-llvm19.patch")

if(NOT EXISTS "${PATCH_FILE}")
  message(FATAL_ERROR "Patch file not found: ${PATCH_FILE}")
endif()

find_program(GIT git REQUIRED)

execute_process(
  COMMAND ${GIT} apply --reverse --check "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE rev_result
  ERROR_QUIET
)

if(rev_result EQUAL 0)
  message(STATUS "xtensor patch: already applied")
else()
  execute_process(
    COMMAND ${GIT} apply "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE fwd_result
    ERROR_VARIABLE fwd_error
  )
  if(NOT fwd_result EQUAL 0)
    message(FATAL_ERROR "xtensor patch failed:\n${fwd_error}")
  endif()
  message(STATUS "xtensor patch: applied")
endif()
