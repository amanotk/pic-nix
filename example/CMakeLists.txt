include(${CMAKE_SOURCE_DIR}/common.cmake OPTIONAL)

# project
project(example CXX)

# all the projects
add_subdirectory(thermal)
add_subdirectory(beam)
add_subdirectory(mrx)
add_subdirectory(fireball)
