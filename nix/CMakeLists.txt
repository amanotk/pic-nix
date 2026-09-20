cmake_minimum_required(VERSION 3.23)

# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(nix CXX)
  option(BUILD_TESTING "Build tests" OFF)
  include(CTest)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED PICNIX_ENABLE_ADIOS2)
  option(PICNIX_ENABLE_ADIOS2 "Enable ADIOS2 diagnostic output" OFF)
endif()

set(PICNIX_ADIOS2_ROOT "" CACHE PATH "ADIOS2 installation prefix")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# filesystem
find_package(Filesystem REQUIRED)

# external dependencies
include(Dependencies)

# build static library
set(NIX_SOURCES application.cpp balancer.cpp nixio.cpp sfc.cpp chunk.cpp chunkmap.cpp)

if(PICNIX_ENABLE_ADIOS2)
  set(_picnix_adios2_hints)
  if(PICNIX_ADIOS2_ROOT)
    list(APPEND _picnix_adios2_hints "${PICNIX_ADIOS2_ROOT}")
  endif()
  find_package(ADIOS2 2.11 CONFIG REQUIRED HINTS ${_picnix_adios2_hints})

  if(NOT TARGET adios2::cxx_mpi)
    message(FATAL_ERROR "ADIOS2 was found but adios2::cxx_mpi is unavailable")
  endif()

  list(APPEND NIX_SOURCES diag/adios.cpp)
else()
  list(APPEND NIX_SOURCES diag/adios_disabled.cpp)
endif()

add_library(nix STATIC ${NIX_SOURCES})

# include directory
target_include_directories(nix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nix PUBLIC picnix::dependencies)

if(PICNIX_ENABLE_ADIOS2)
  target_link_libraries(nix PUBLIC adios2::cxx_mpi)
endif()

set(NIX_BUILD_TESTING_DEFAULT OFF)
if(DEFINED BUILD_TESTING)
  set(NIX_BUILD_TESTING_DEFAULT ${BUILD_TESTING})
endif()
option(NIX_BUILD_TESTING "Build nix tests" ${NIX_BUILD_TESTING_DEFAULT})

# unittest
if(NIX_BUILD_TESTING)
  add_subdirectory(unittest)
endif()
