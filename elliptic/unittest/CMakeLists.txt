enable_testing()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Catch2Setup.cmake)

set(UNITTEST_INCLUDE_DIRS
        ${NIX_DIR}
        ${NIX_DIR}/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(PARALLEL_EXECS
        test_chunk_accessor
        test_petsc_scatter
        test_petsc_interface
        test_petsc_poisson
)

set(MPIEXEC mpiexec)

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

add_test(NAME test_chunk_accessor_np1
        COMMAND ${MPIEXEC} -n 1 $<TARGET_FILE:test_chunk_accessor> "[np=1]"
)
add_test(NAME test_chunk_accessor_np8
        COMMAND ${MPIEXEC} -n 8 $<TARGET_FILE:test_chunk_accessor> "[np=8]"
)
add_test(NAME test_petsc_scatter_np1
        COMMAND ${MPIEXEC} -n 1 $<TARGET_FILE:test_petsc_scatter> "[np=1]"
)
add_test(NAME test_petsc_scatter_np8
        COMMAND ${MPIEXEC} -n 8 $<TARGET_FILE:test_petsc_scatter> "[np=8]"
)
add_test(NAME test_petsc_interface_np1
        COMMAND ${MPIEXEC} -n 1 $<TARGET_FILE:test_petsc_interface> "[np=1]"
)
add_test(NAME test_petsc_poisson_np1
        COMMAND ${MPIEXEC} -n 1 $<TARGET_FILE:test_petsc_poisson> "[np=1]"
)
add_test(NAME test_petsc_poisson_np8
        COMMAND ${MPIEXEC} -n 8 $<TARGET_FILE:test_petsc_poisson> "[np=8]"
)
