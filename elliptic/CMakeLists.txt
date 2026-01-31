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

# nix headers (no nix build)
set(NIX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nix" CACHE PATH "Path to nix source tree")
if(NOT EXISTS "${NIX_DIR}/nix.hpp")
  message(FATAL_ERROR "NIX_DIR does not contain nix.hpp: ${NIX_DIR}")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(BUILD_TESTING "Build tests" OFF)
  include(CTest)
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

set(ELLIPTIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${NIX_DIR}
  ${NIX_DIR}/thirdparty
)

add_library(elliptic STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_matrix_helpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc_poisson.cpp
)
target_include_directories(elliptic PUBLIC ${ELLIPTIC_INCLUDE_DIRS})
target_link_libraries(elliptic PUBLIC PkgConfig::PETSC)

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
