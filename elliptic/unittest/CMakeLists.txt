if(PICNIX_ENABLE_PETSC)
  enable_testing()

  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Catch2Setup.cmake)

  set(UNITTEST_INCLUDE_DIRS
          ${NIX_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

  set(PARALLEL_EXECS
          test_chunk_accessor
          test_petsc_scatter
          test_petsc_interface
          test_petsc_poisson
  )

  if(NOT DEFINED MPIEXEC)
    find_program(MPIEXEC NAMES mpiexec mpirun REQUIRED)
  endif()

  set(MPI_TEST_TIMEOUT 300)

  function(add_mpi_test name target nproc)
    set(workdir ${CMAKE_CURRENT_BINARY_DIR}/tmp_${name})
    file(MAKE_DIRECTORY ${workdir})
    add_test(NAME ${name}
      COMMAND ${MPIEXEC} -n ${nproc} $<TARGET_FILE:${target}> ${ARGN}
    )
    set_tests_properties(${name}
      PROPERTIES
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT ${MPI_TEST_TIMEOUT}
        WORKING_DIRECTORY ${workdir}
    )
  endfunction()

  add_library(elliptic_obj_parallel OBJECT test_parallel.cpp)
  target_include_directories(elliptic_obj_parallel PRIVATE ${UNITTEST_INCLUDE_DIRS})
  target_link_libraries(elliptic_obj_parallel PRIVATE Catch2::Catch2)
  target_link_libraries(elliptic_obj_parallel PRIVATE PkgConfig::PETSC)

  foreach(target IN ITEMS ${PARALLEL_EXECS})
          add_executable(${target} ${target}.cpp)
          target_include_directories(${target} PRIVATE ${UNITTEST_INCLUDE_DIRS})
          target_link_libraries(${target} PRIVATE
                  Catch2::Catch2
                  elliptic_obj_parallel
                  elliptic
                  PkgConfig::PETSC
          )
  endforeach()

  add_mpi_test(test_chunk_accessor_np1 test_chunk_accessor 1 "[np=1]")
  add_mpi_test(test_chunk_accessor_np8 test_chunk_accessor 8 "[np=8]")
  add_mpi_test(test_petsc_scatter_np1 test_petsc_scatter 1 "[np=1]")
  add_mpi_test(test_petsc_scatter_np8 test_petsc_scatter 8 "[np=8]")
  add_mpi_test(test_petsc_interface_np1 test_petsc_interface 1 "[np=1]")
  add_mpi_test(test_petsc_poisson_np1 test_petsc_poisson 1 "[np=1]")
  add_mpi_test(test_petsc_poisson_np8 test_petsc_poisson 8 "[np=8]")
else()
  message(STATUS "Skipping elliptic PETSc tests: PICNIX_ENABLE_PETSC=OFF")
endif()
