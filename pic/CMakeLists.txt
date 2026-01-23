cmake_minimum_required(VERSION 3.14)

# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pic CXX)
  option(BUILD_TESTING "Build tests" OFF)
  include(CTest)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NIX_BUILD_TESTING OFF CACHE BOOL "Build nix tests" FORCE)
  add_subdirectory(${PICNIX_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
endif()

# build static library
add_library(pic STATIC pic_chunk.cpp pic_application.cpp pic_poisson.cpp)

# include directory
target_include_directories(pic PRIVATE ${PICNIX_DIR})
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix)
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix/thirdparty)
target_include_directories(pic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pic PUBLIC elliptic)

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
