include(${CMAKE_SOURCE_DIR}/common.cmake OPTIONAL)

set(PROJECT_NAME "mrx")

# project
project(${PROJECT_NAME} CXX)

# executable
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME main.out)

# include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

# link library
target_link_libraries(${PROJECT_NAME} std::filesystem)
target_link_libraries(${PROJECT_NAME} nix expic)
