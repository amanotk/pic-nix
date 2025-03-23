# project
project(pic CXX)

# build static library
add_library(pic STATIC pic_chunk.cpp)

# include directory
target_include_directories(pic PRIVATE ${PICNIX_DIR})
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix)
target_include_directories(pic PRIVATE ${PICNIX_DIR}/nix/thirdparty)
target_include_directories(pic PUBLIC ${PROJECT_SOURCE_DIR})
