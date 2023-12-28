cmake_minimum_required(VERSION 3.14)
include(common.cmake)

### project
project(pic-nix CXX)

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

### nix as an external project
include(FetchContent)

FetchContent_Declare(
    nix
    GIT_REPOSITORY https://github.com/amanotk/nix.git
    GIT_TAG daa339ef3f6d103e9a49322972c3c854e3a62d9a
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nix
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nix
)
FetchContent_MakeAvailable(nix)

# Filesystem package
set(CMAKE_MODULE_PATH ${nix_SOURCE_DIR})
find_package(Filesystem REQUIRED)

### build all executables in example directory
option(BUILD_EXAMPLE "Build example" ON)
if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif()
