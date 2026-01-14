enable_testing()

set(UNITTEST_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nix
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nix/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(PARALLEL_EXECS
        test_chunk_accessor
        test_petsc_scatter
        test_petsc_solver
)

set(MPIEXEC mpiexec)

add_library(obj_parallel OBJECT test_parallel.cpp)
target_include_directories(obj_parallel PRIVATE ${UNITTEST_INCLUDE_DIRS})
target_link_libraries(obj_parallel PRIVATE PkgConfig::PETSC)

foreach(target IN ITEMS ${PARALLEL_EXECS})
        add_executable(${target} ${target}.cpp)
        target_include_directories(${target} PRIVATE ${UNITTEST_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE
                obj_parallel
                nix
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
add_test(NAME test_petsc_solver_np1
        COMMAND ${MPIEXEC} -n 1 $<TARGET_FILE:test_petsc_solver> "[np=1]"
)
