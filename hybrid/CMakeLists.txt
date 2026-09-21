cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hybrid CXX)
  option(BUILD_TESTING "Build tests" OFF)
  option(PICNIX_BUILD_HYBRID_EXAMPLE "Build hybrid examples" ON)
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

add_library(hybrid STATIC hybrid_application.cpp hybrid_chunk.cpp)

target_include_directories(hybrid
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PICNIX_DIR}
)
target_link_libraries(hybrid PUBLIC nix)

if(PICNIX_BUILD_HYBRID_EXAMPLE)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
