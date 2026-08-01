# project
set(PROJECT_NAME "pic_unittest")
project(${PROJECT_NAME} CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../nix/cmake/Catch2Setup.cmake)

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_LIST_DIR}/../..
  ${CMAKE_CURRENT_LIST_DIR}/..
  ${CMAKE_CURRENT_LIST_DIR}/../../nix
  )

# helpers
if(NOT DEFINED MPIEXEC)
  find_program(MPIEXEC NAMES mpiexec mpirun REQUIRED)
endif()

set(MPI_TEST_TIMEOUT 300)

function(add_mpi_test name target nproc tmpdir)
  set(workdir ${tmpdir}_cwd)
  set(prepare_script ${CMAKE_CURRENT_BINARY_DIR}/${name}_prepare.cmake)
  file(WRITE ${prepare_script}
    "file(REMOVE_RECURSE \"${tmpdir}\" \"${workdir}\")\n"
    "file(MAKE_DIRECTORY \"${tmpdir}\" \"${workdir}\")\n"
  )
  add_test(NAME ${name}_prepare
    COMMAND ${CMAKE_COMMAND} -P ${prepare_script}
  )
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND} -E env PICNIX_TMPDIR=${tmpdir}
      ${MPIEXEC} -n ${nproc} $<TARGET_FILE:${target}> ${ARGN}
  )
  set_tests_properties(${name}
    PROPERTIES
      DEPENDS ${name}_prepare
      LABELS MPI
      PROCESSORS ${nproc}
      TIMEOUT ${MPI_TEST_TIMEOUT}
      WORKING_DIRECTORY ${workdir}
  )
endfunction()

function(add_mpi_test_with_cleanup name target nproc tmpdir)
  add_mpi_test(${name} ${target} ${nproc} ${tmpdir} ${ARGN})
  add_test(NAME ${name}_cleanup
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${tmpdir} ${tmpdir}_cwd
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
target_compile_definitions(pic_obj_parallel PRIVATE PICNIX_ENABLE_PETSC=$<BOOL:${PICNIX_ENABLE_PETSC}>)

add_library(pic_obj_parallel_mpi_only OBJECT test_parallel_mpi_only.cpp)
target_include_directories(pic_obj_parallel_mpi_only PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pic_obj_parallel_mpi_only PRIVATE Catch2::Catch2)
target_link_libraries(pic_obj_parallel_mpi_only PRIVATE pic nix)
target_compile_definitions(pic_obj_parallel_mpi_only PRIVATE PICNIX_ENABLE_PETSC=$<BOOL:${PICNIX_ENABLE_PETSC}>)

# serial tests
set(SERIAL_EXECS
  test_pic_chunk
  test_pic_domain_view
  )

if(PICNIX_ENABLE_ASCENT)
  list(APPEND SERIAL_EXECS test_pic_blueprint_builder)
endif()

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

if(PICNIX_ENABLE_ASCENT)
  set_tests_properties(test_pic_blueprint_builder PROPERTIES LABELS ASCENT)

  add_executable(test_pic_ascent_runtime test_pic_ascent_runtime.cpp)
  target_include_directories(test_pic_ascent_runtime PRIVATE ${INCLUDE_DIRS})
  target_compile_definitions(test_pic_ascent_runtime PRIVATE
    PICNIX_ASCENT_ACTIONS_FILE="${CMAKE_CURRENT_SOURCE_DIR}/ascent_empty_actions.yaml"
  )
  target_link_libraries(test_pic_ascent_runtime PRIVATE
    Catch2::Catch2
    pic_obj_parallel
    pic
    nix
  )

  add_mpi_test(
    test_pic_ascent_runtime_np1
    test_pic_ascent_runtime
    1
    ${CMAKE_CURRENT_BINARY_DIR}/tmp_ascent_runtime_np1
  )
  set_tests_properties(test_pic_ascent_runtime_np1_prepare test_pic_ascent_runtime_np1
    PROPERTIES LABELS "MPI;ASCENT"
  )

  if(ASCENT_PYTHON_ENABLED AND CONDUIT_PYTHON_ENABLED)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ascent_two_rank_actions.yaml.in
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_two_rank_actions.yaml
    @ONLY
  )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ascent_two_rank_extract.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_two_rank_extract.py
    @ONLY
  )

  add_executable(test_pic_ascent_extract test_pic_ascent_extract.cpp)
  target_include_directories(test_pic_ascent_extract PRIVATE ${INCLUDE_DIRS})
  target_compile_definitions(test_pic_ascent_extract PRIVATE
    PICNIX_ASCENT_EXTRACT_ACTIONS_FILE="${CMAKE_CURRENT_BINARY_DIR}/ascent_two_rank_actions.yaml"
  )
  target_link_libraries(test_pic_ascent_extract PRIVATE
    Catch2::Catch2
    pic_obj_parallel
    pic
    nix
  )

  add_mpi_test_with_cleanup(
    test_pic_ascent_extract_np2
    test_pic_ascent_extract
    2
    ${CMAKE_CURRENT_BINARY_DIR}/tmp_ascent_extract_np2
  )
  set_tests_properties(test_pic_ascent_extract_np2_prepare test_pic_ascent_extract_np2
    test_pic_ascent_extract_np2_cleanup
    PROPERTIES LABELS "MPI;ASCENT"
  )
  set_tests_properties(test_pic_ascent_extract_np2 PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PYTHONPATH=path_list_prepend:${PICNIX_DIR}/python/src;PYTHONPATH=path_list_prepend:${CONDUIT_PYTHON_MODULE_DIR}"
  )
  endif()

  if(ASCENT_VTKH_ENABLED)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ascent_render_actions.yaml.in
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_render_actions.yaml
    @ONLY
  )

  add_executable(test_pic_ascent_render test_pic_ascent_render.cpp)
  target_include_directories(test_pic_ascent_render PRIVATE ${INCLUDE_DIRS})
  target_compile_definitions(test_pic_ascent_render PRIVATE
    PICNIX_ASCENT_RENDER_ACTIONS_FILE="${CMAKE_CURRENT_BINARY_DIR}/ascent_render_actions.yaml"
  )
  target_link_libraries(test_pic_ascent_render PRIVATE
    Catch2::Catch2
    pic_obj_parallel
    pic
    nix
  )

  add_mpi_test_with_cleanup(
    test_pic_ascent_render_np1
    test_pic_ascent_render
    1
    ${CMAKE_CURRENT_BINARY_DIR}/tmp_ascent_render_np1
  )
  set_tests_properties(test_pic_ascent_render_np1_prepare test_pic_ascent_render_np1
    test_pic_ascent_render_np1_cleanup PROPERTIES LABELS "MPI;ASCENT"
  )
  endif()
endif()

# PicApplicationInterface smoke / application tests (Catch2, MPI)
add_executable(test_pic_application test_pic_application.cpp)
target_include_directories(test_pic_application PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_pic_application PRIVATE
  Catch2::Catch2
  pic_obj_parallel
  pic
  nix
)

set(PIC_IFACE_TMPDIR_NP8 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_np8)
add_mpi_test(
  test_pic_application_np8
  test_pic_application
  8
  ${PIC_IFACE_TMPDIR_NP8}
)

if(PICNIX_ENABLE_ASCENT)
  set(PIC_IFACE_ASCENT_TMPDIR_NP1 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_ascent_np1)
  add_mpi_test(
    test_pic_application_ascent_np1
    test_pic_application
    1
    ${PIC_IFACE_ASCENT_TMPDIR_NP1}
    "PicApplication runs Ascent diagnostic through finalization"
  )
  set_tests_properties(test_pic_application_ascent_np1_prepare test_pic_application_ascent_np1
    PROPERTIES LABELS "MPI;ASCENT"
  )
endif()

if(PICNIX_ENABLE_PETSC)
  set(PIC_IFACE_TMPDIR_NP1 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_np1)
  add_mpi_test(
    test_pic_application_np1
    test_pic_application
    1
    ${PIC_IFACE_TMPDIR_NP1}
  )

  add_executable(test_pic_application_mpi_only test_pic_application.cpp)
  target_include_directories(test_pic_application_mpi_only PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(test_pic_application_mpi_only PRIVATE
    Catch2::Catch2
    pic_obj_parallel_mpi_only
    pic
    nix
  )

  set(PIC_IFACE_MPI_ONLY_TMPDIR_NP1 ${CMAKE_CURRENT_BINARY_DIR}/tmp_iface_mpi_only_np1)
  add_mpi_test(
    test_pic_application_mpi_only_np1
    test_pic_application_mpi_only
    1
    ${PIC_IFACE_MPI_ONLY_TMPDIR_NP1}
    pic_application_interface_smoke
  )
endif()

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

# Basic PicPoisson (fallback) MPI tests always build/run
add_executable(test_pic_poisson_basic test_pic_poisson_basic.cpp)
target_include_directories(test_pic_poisson_basic PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_pic_poisson_basic PRIVATE
  Catch2::Catch2
  pic_obj_parallel
  pic
  nix
)

set(PIC_POISSON_BASIC_TMPDIR_NP2 ${CMAKE_CURRENT_BINARY_DIR}/tmp_pic_poisson_basic_np2)
add_mpi_test(
  test_pic_poisson_basic_np2
  test_pic_poisson_basic
  2
  ${PIC_POISSON_BASIC_TMPDIR_NP2}
)

set(PIC_POISSON_BASIC_TMPDIR_NP4 ${CMAKE_CURRENT_BINARY_DIR}/tmp_pic_poisson_basic_np4)
add_mpi_test(
  test_pic_poisson_basic_np4
  test_pic_poisson_basic
  4
  ${PIC_POISSON_BASIC_TMPDIR_NP4}
)

set(PIC_POISSON_BASIC_TMPDIR_NP8 ${CMAKE_CURRENT_BINARY_DIR}/tmp_pic_poisson_basic_np8)
add_mpi_test(
  test_pic_poisson_basic_np8
  test_pic_poisson_basic
  8
  ${PIC_POISSON_BASIC_TMPDIR_NP8}
)

# PicPoisson MPI tests (Catch2, MPI)
if(PICNIX_ENABLE_PETSC)
  add_executable(test_pic_poisson test_pic_poisson_petsc.cpp)
  target_include_directories(test_pic_poisson PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(test_pic_poisson PRIVATE
    Catch2::Catch2
    pic_obj_parallel
    pic
    nix
  )

  set(PIC_POISSON_TMPDIR_NP2 ${CMAKE_CURRENT_BINARY_DIR}/tmp_pic_poisson_np2)
  add_mpi_test(
    test_pic_poisson_np2
    test_pic_poisson
    2
    ${PIC_POISSON_TMPDIR_NP2}
  )
endif()
