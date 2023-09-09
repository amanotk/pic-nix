# default configuration
cmake_minimum_required(VERSION 3.10)
include(${PICNIX_DIR}/common.cmake OPTIONAL)

set(PROJECT_NAME "thermal")

# project
project(${PROJECT_NAME} CXX)

# order of shape function
option(SHAPE_ORDER "Set SHAPE_ORDER to specify order of shape function" 1)
add_compile_definitions(SHAPE_ORDER=${SHAPE_ORDER})

# executable
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME main.out)

# include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${PICNIX_DIR})

# link library
target_link_libraries(${PROJECT_NAME} nix expic)
