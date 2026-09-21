add_executable(hybrid_beam main.cpp)
set_target_properties(hybrid_beam PROPERTIES OUTPUT_NAME main.out)

target_link_libraries(hybrid_beam PRIVATE hybrid)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config.toml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
