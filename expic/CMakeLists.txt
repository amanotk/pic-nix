# project
project(expic CXX)

# build static library
add_library(expic STATIC exchunk3d.cpp)

# include directory
target_include_directories(expic PRIVATE ${PICNIX_DIR})
target_include_directories(expic PRIVATE ${PICNIX_DIR}/nix)
target_include_directories(expic PRIVATE ${PICNIX_DIR}/nix/thirdparty)
target_include_directories(expic PUBLIC ${PROJECT_SOURCE_DIR})
