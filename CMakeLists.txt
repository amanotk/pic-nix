cmake_minimum_required(VERSION 3.20)

set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)

set(CMAKE_CXX_STANDARD 17)

# project
project(pic-nix CXX)

option(BUILD_TESTING "Build tests" OFF)
include(CTest)

option(PICNIX_BUILD_NIX "Build nix module" ON)
option(PICNIX_BUILD_PIC "Build pic module" ON)
option(PICNIX_BUILD_ELLIPTIC "Build elliptic module" OFF)
option(PICNIX_ENABLE_PETSC "Enable PETSc-based solvers" OFF)

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# dependency management
option(PICNIX_USE_SYSTEM_LIBS "Prefer system-installed libraries over FetchContent" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nix/cmake")
include(Dependencies)

# MPI_THREAD_MULTIPLE
option(MPI_THREAD_MULTIPLE "Enable MPI_THREAD_MULTIPLE" ON)
if(MPI_THREAD_MULTIPLE)
    add_compile_definitions(HAS_MPI_THREAD_MULTIPLE)
endif()

# nix
if(PICNIX_BUILD_PIC AND NOT PICNIX_BUILD_NIX)
  set(PICNIX_BUILD_NIX ON CACHE BOOL "Build nix module" FORCE)
endif()

if(PICNIX_BUILD_PIC AND NOT PICNIX_BUILD_ELLIPTIC)
  set(PICNIX_BUILD_ELLIPTIC ON CACHE BOOL "Build elliptic module" FORCE)
endif()

if(PICNIX_BUILD_NIX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
endif()

# Filesystem package
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nix")
find_package(Filesystem REQUIRED)

# elliptic (optional)
if(PICNIX_BUILD_ELLIPTIC)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/elliptic ${CMAKE_CURRENT_BINARY_DIR}/elliptic)
endif()

# explicit PIC library
if(PICNIX_BUILD_PIC)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pic ${CMAKE_CURRENT_BINARY_DIR}/pic)
endif()

# build all executables in pic/example directory
option(PICNIX_BUILD_EXAMPLE "Build example" ON)

if(PICNIX_BUILD_EXAMPLE AND PICNIX_BUILD_PIC)
  add_subdirectory(pic/example)
endif()
