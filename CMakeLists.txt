cmake_minimum_required(VERSION 3.14)
include(common.cmake)

# project
project(pic-nix CXX)

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# nix as an external project
include(FetchContent)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nix)
  # use the existing nix directory by default
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
else()
  # otherwise, fetch the nix repository from github
  FetchContent_Declare(
    nix
    GIT_REPOSITORY https://github.com/amanotk/nix.git
    GIT_TAG acb93ba926d3aec4cd3ab4cf80840fc0d1ec474e
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nix
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nix
  )
  FetchContent_MakeAvailable(nix)
endif()

# Filesystem package
set(CMAKE_MODULE_PATH ${nix_SOURCE_DIR})
find_package(Filesystem REQUIRED)

# explicit PIC library
add_subdirectory(${PICNIX_DIR}/expic ${CMAKE_BINARY_DIR}/expic)

# build all executables in example directory
option(BUILD_EXAMPLE "Build example" ON)

if(BUILD_EXAMPLE)
  add_subdirectory(example)
endif()
