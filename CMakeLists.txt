cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

# project
project(pic-nix CXX)

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# xsimd for xtensor
add_compile_definitions(XTENSOR_USE_XSIMD)

# MPI_THREAD_MULTIPLE
option(MPI_THREAD_MULTIPLE "Enable MPI_THREAD_MULTIPLE" ON)
if(MPI_THREAD_MULTIPLE)
    add_compile_definitions(HAS_MPI_THREAD_MULTIPLE)
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
    GIT_TAG v0.2.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nix
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nix
  )
  FetchContent_MakeAvailable(nix)
endif()

# Filesystem package
set(CMAKE_MODULE_PATH ${nix_SOURCE_DIR})
find_package(Filesystem REQUIRED)

# explicit PIC library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pic ${CMAKE_CURRENT_BINARY_DIR}/pic)

# build all executables in example directory
option(PICNIX_BUILD_EXAMPLE "Build example" ON)

if(PICNIX_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()
