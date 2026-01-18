# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pic CXX)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# build static library
add_library(pic STATIC pic_chunk.cpp pic_application.cpp)

# include directory
target_include_directories(pic PRIVATE ${PICNIX_DIR})
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix)
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix/thirdparty)
target_include_directories(pic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
