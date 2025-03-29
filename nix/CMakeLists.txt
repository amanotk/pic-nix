cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

# project
project(nix CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# filesystem
find_package(Filesystem REQUIRED)

# xsimd for xtensor
add_compile_definitions(XTENSOR_USE_XSIMD)

# build static library
add_library(nix STATIC balancer.cpp nixio.cpp sfc.cpp chunk3d.cpp)

# include directory
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(nix PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty)

# unittest
if(ENABLE_TESTING)
	enable_testing()
	add_subdirectory(unittest)
endif()
