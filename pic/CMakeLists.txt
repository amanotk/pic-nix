# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pic CXX)
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

  if(NOT DEFINED PICNIX_DIR)
    set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
  endif()

  set(BUILD_TESTING_PIC ${BUILD_TESTING})
  set(BUILD_TESTING OFF)
  set(NIX_BUILD_TESTING OFF CACHE BOOL "Build nix tests" FORCE)
  add_subdirectory(${PICNIX_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
  set(BUILD_TESTING ${BUILD_TESTING_PIC})
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

option(BUILD_TESTING "Build tests" OFF)
include(CTest)

# build static library
add_library(pic STATIC pic_chunk.cpp pic_application.cpp)

# include directory
target_include_directories(pic PRIVATE ${PICNIX_DIR})
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix)
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix/thirdparty)
target_include_directories(pic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
