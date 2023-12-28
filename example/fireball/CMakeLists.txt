include(${PICNIX_DIR}/common.cmake OPTIONAL)

set(PROJECT_NAME "fireball")

# project
project(${PROJECT_NAME} CXX)

# order of shape function
set(SHAPE_ORDER 2 CACHE STRING "Order of shape function")
add_compile_definitions(SHAPE_ORDER=${SHAPE_ORDER})

# executable
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME main.out)

# include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${PICNIX_DIR})

# link library
target_link_libraries(${PROJECT_NAME} std::filesystem)
target_link_libraries(${PROJECT_NAME} nix expic)

# copy
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/batch.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
