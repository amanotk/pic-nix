# default configuration
cmake_minimum_required(VERSION 3.10)
include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake OPTIONAL)

# project
project(expic_exec CXX)

# add subdirectory
add_subdirectory(nix)
add_subdirectory(expic)

# include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nix)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nix/thirdparty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/expic)

# build executable
add_executable(thermal.out project3d/thermal/main.cpp)
add_executable(beam.out project3d/beam/main.cpp)

# link library
target_link_libraries(thermal.out nix expic)
target_link_libraries(beam.out nix expic)
