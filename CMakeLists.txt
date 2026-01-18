cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

# project
project(pic-nix CXX)

option(BUILD_TESTING "Build tests" OFF)
include(CTest)

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

# nix
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)

# Filesystem package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nix")
find_package(Filesystem REQUIRED)

# explicit PIC library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pic ${CMAKE_CURRENT_BINARY_DIR}/pic)

# build all executables in pic/example directory
option(PICNIX_BUILD_EXAMPLE "Build example" ON)

if(PICNIX_BUILD_EXAMPLE)
  add_subdirectory(pic/example)
endif()
