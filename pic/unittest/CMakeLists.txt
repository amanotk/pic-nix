# project
set(PROJECT_NAME "pic_unittest")
project(${PROJECT_NAME} CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Catch2Setup.cmake)

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_LIST_DIR}/../..
  ${CMAKE_CURRENT_LIST_DIR}/..
  ${CMAKE_CURRENT_LIST_DIR}/../../nix
  ${CMAKE_CURRENT_LIST_DIR}/../../nix/thirdparty
  )

set(SERIAL_EXECS
  test_pic_chunk
  )

add_library(obj_serial OBJECT test_serial.cpp)
target_include_directories(obj_serial PRIVATE ${INCLUDE_DIRS})
target_link_libraries(obj_serial PRIVATE Catch2::Catch2)

target_link_libraries(obj_serial PRIVATE pic nix)

foreach(target IN ITEMS ${SERIAL_EXECS})
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE
    Catch2::Catch2
    obj_serial
    pic
    nix
  )
  add_test(NAME ${target} COMMAND ${target})
endforeach()
