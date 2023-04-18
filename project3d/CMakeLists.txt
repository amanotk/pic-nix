# default configuration
cmake_minimum_required(VERSION 3.10)
include(${PICNIX_DIR}/common.cmake OPTIONAL)

# project
project(project3d CXX)

# source directory
add_subdirectory(${PICNIX_DIR}/nix ${CMAKE_BINARY_DIR}/nix)
add_subdirectory(${PICNIX_DIR}/expic ${CMAKE_BINARY_DIR}/expic)

# all the projects
add_subdirectory(thermal)
add_subdirectory(beam)
add_subdirectory(fireball)
