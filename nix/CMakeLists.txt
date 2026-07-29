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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# filesystem
find_package(Filesystem REQUIRED)

# external dependencies
include(Dependencies)

# build static library
add_library(nix STATIC application.cpp balancer.cpp nixio.cpp sfc.cpp chunk.cpp chunkmap.cpp)

# include directory
target_include_directories(nix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nix PUBLIC picnix::dependencies)

set(NIX_BUILD_TESTING_DEFAULT OFF)
if(DEFINED BUILD_TESTING)
  set(NIX_BUILD_TESTING_DEFAULT ${BUILD_TESTING})
endif()
option(NIX_BUILD_TESTING "Build nix tests" ${NIX_BUILD_TESTING_DEFAULT})

# unittest
if(NIX_BUILD_TESTING)
  add_subdirectory(unittest)
endif()
