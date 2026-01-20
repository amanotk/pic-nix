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

add_library(pic_obj_serial OBJECT test_serial.cpp)
target_include_directories(pic_obj_serial PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pic_obj_serial PRIVATE Catch2::Catch2)

target_link_libraries(pic_obj_serial PRIVATE pic nix)

add_library(pic_obj_parallel OBJECT test_parallel.cpp)
target_include_directories(pic_obj_parallel PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pic_obj_parallel PRIVATE Catch2::Catch2)

target_link_libraries(pic_obj_parallel PRIVATE pic nix)

foreach(target IN ITEMS ${SERIAL_EXECS})
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE
    Catch2::Catch2
    pic_obj_serial
    pic
    nix
  )
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# executable
add_executable(run_pic_application run_pic_application.cpp)
target_include_directories(run_pic_application PRIVATE ${INCLUDE_DIRS})
target_link_libraries(run_pic_application pic nix)


set(MPIEXEC mpiexec)
set(PIC_TMPDIR_NP1 ${CMAKE_CURRENT_BINARY_DIR}/tmp_np1)
set(PIC_TMPDIR_NP8 ${CMAKE_CURRENT_BINARY_DIR}/tmp_np8)
add_test(NAME run_pic_application_np1_prepare
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PIC_TMPDIR_NP1}
)
add_test(NAME run_pic_application_np8_prepare
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PIC_TMPDIR_NP8}
)
add_test(NAME run_pic_application_np1
  COMMAND ${CMAKE_COMMAND} -E env PICNIX_TMPDIR=${PIC_TMPDIR_NP1}
    ${MPIEXEC} -n 1 $<TARGET_FILE:run_pic_application>
    -c ${CMAKE_CURRENT_LIST_DIR}/testdata/run_pic_application_np1.toml
    -t 0.5
)
set_tests_properties(run_pic_application_np1
  PROPERTIES DEPENDS run_pic_application_np1_prepare
)
add_test(NAME run_pic_application_np8
  COMMAND ${CMAKE_COMMAND} -E env PICNIX_TMPDIR=${PIC_TMPDIR_NP8}
    ${MPIEXEC} -n 8 $<TARGET_FILE:run_pic_application>
    -c ${CMAKE_CURRENT_LIST_DIR}/testdata/run_pic_application_np8.toml
    -t 0.5
)
set_tests_properties(run_pic_application_np8
  PROPERTIES DEPENDS run_pic_application_np8_prepare
)

add_test(NAME run_pic_application_np1_cleanup
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${PIC_TMPDIR_NP1}
)
set_tests_properties(run_pic_application_np1_cleanup
  PROPERTIES DEPENDS run_pic_application_np1
)

add_test(NAME run_pic_application_np8_cleanup
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${PIC_TMPDIR_NP8}
)
set_tests_properties(run_pic_application_np8_cleanup
  PROPERTIES DEPENDS run_pic_application_np8
)
