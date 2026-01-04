cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

# project
project(elliptic CXX)

# nix
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nix ${CMAKE_CURRENT_BINARY_DIR}/nix_build)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

enable_testing()
add_subdirectory(unittest)