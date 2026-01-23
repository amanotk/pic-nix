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

# helpers
function(add_mpi_test name target nproc tmpdir)
  add_test(NAME ${name}_prepare
    COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir}
  )
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -E env PICNIX_TMPDIR=${tmpdir}
      ${MPIEXEC} -n ${nproc} $<TARGET_FILE:${target}> ${ARGN}
  )
  set_tests_properties(${name}
    PROPERTIES DEPENDS ${name}_prepare
  )
endfunction()

function(add_mpi_test_with_cleanup name target nproc tmpdir)
  add_mpi_test(${name} ${target} ${nproc} ${tmpdir} ${ARGN})
  add_test(NAME ${name}_cleanup
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${tmpdir}
  )
  set_tests_properties(${name}_cleanup
    PROPERTIES DEPENDS ${name}
  )
endfunction()

# common test runners
add_library(pic_obj_serial OBJECT test_serial.cpp)
target_include_directories(pic_obj_serial PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pic_obj_serial PRIVATE Catch2::Catch2)
target_link_libraries(pic_obj_serial PRIVATE pic nix)

add_library(pic_obj_parallel OBJECT test_parallel.cpp)
target_include_directories(pic_obj_parallel PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pic_obj_parallel PRIVATE Catch2::Catch2)
target_link_libraries(pic_obj_parallel PRIVATE pic nix)

# serial tests
set(SERIAL_EXECS
  test_pic_chunk
  )

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

# PicApplicationInterface smoke test (Catch2, MPI)
add_executable(test_pic_application test_pic_application.cpp)
target_include_directories(test_pic_application PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_pic_application PRIVATE
  Catch2::Catch2
  pic_obj_parallel
  pic
  nix
)

set(MPIEXEC mpiexec)
set(PIC_IFACE_TMPDIR_NP1 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_np1)
set(PIC_IFACE_TMPDIR_NP8 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_np8)
add_mpi_test(
  test_pic_application_np1
  test_pic_application
  1
  ${PIC_IFACE_TMPDIR_NP1}
)
add_mpi_test(
  test_pic_application_np8
  test_pic_application
  8
  ${PIC_IFACE_TMPDIR_NP8}
)

# boundary exchange tests (Catch2, MPI)
add_executable(test_pic_boundary test_pic_boundary.cpp)
target_include_directories(test_pic_boundary PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_pic_boundary PRIVATE
  Catch2::Catch2
  pic_obj_parallel
  pic
  nix
)

set(PIC_BOUNDARY_TMPDIR_NP2 ${CMAKE_CURRENT_BINARY_DIR}/tmp_boundary_np2)
set(PIC_BOUNDARY_TMPDIR_NP4 ${CMAKE_CURRENT_BINARY_DIR}/tmp_boundary_np4)
set(PIC_BOUNDARY_TMPDIR_NP8 ${CMAKE_CURRENT_BINARY_DIR}/tmp_boundary_np8)
add_mpi_test(
  test_pic_boundary_np2
  test_pic_boundary
  2
  ${PIC_BOUNDARY_TMPDIR_NP2}
)
add_mpi_test(
  test_pic_boundary_np4
  test_pic_boundary
  4
  ${PIC_BOUNDARY_TMPDIR_NP4}
)
add_mpi_test(
  test_pic_boundary_np8
  test_pic_boundary
  8
  ${PIC_BOUNDARY_TMPDIR_NP8}
)
