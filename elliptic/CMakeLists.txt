cmake_minimum_required(VERSION 3.14)

project(elliptic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# nix (disable nix tests while configuring elliptic)
set(_ELLIPTIC_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nix ${CMAKE_CURRENT_BINARY_DIR}/nix_build)
set(BUILD_TESTING ${_ELLIPTIC_BUILD_TESTING})

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

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(unittest)
endif()
