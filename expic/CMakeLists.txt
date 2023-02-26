# default configuration
cmake_minimum_required(VERSION 3.10)
include(${CMAKE_CURRENT_SOURCE_DIR}/../common.cmake OPTIONAL)

# project
project(libexpic CXX)

# include directory
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${BASE_DIR})
include_directories(${BASE_DIR}/nix)
include_directories(${BASE_DIR}/nix/thirdparty)

# build static library
add_library(expic STATIC exchunk3d.cpp expic3d.cpp)

# to be included
target_include_directories(expic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
