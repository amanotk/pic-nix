cmake_minimum_required(VERSION 3.10)

# define PIC-NIX source tree
if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# read default configuration
include(${PICNIX_DIR}/common.cmake OPTIONAL)

# project
project(pic-nix CXX)

# package
set(CMAKE_MODULE_PATH "${PICNIX_DIR}/nix")
find_package(Filesystem REQUIRED)

# build all executables in project3d
add_subdirectory(project3d)
