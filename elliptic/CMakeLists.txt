cmake_minimum_required(VERSION 3.14)

# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(elliptic LANGUAGES CXX)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# nix (disable nix tests while configuring elliptic)
set(NIX_BUILD_TESTING OFF CACHE BOOL "Build nix tests")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nix ${CMAKE_CURRENT_BINARY_DIR}/nix_build)

option(BUILD_TESTING "Build tests" OFF)
include(CTest)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

set(ELLIPTIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../nix
  ${CMAKE_CURRENT_SOURCE_DIR}/../nix/thirdparty
)

add_library(elliptic STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_poisson.cpp
)
target_include_directories(elliptic PUBLIC ${ELLIPTIC_INCLUDE_DIRS})
target_link_libraries(elliptic PUBLIC nix PkgConfig::PETSC)

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
