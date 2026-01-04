enable_testing()

set(INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nix
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nix/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        )

set(PARALLEL_EXECS
        test_petsc_utils
        )

add_library(obj_parallel OBJECT test_parallel.cpp)
target_include_directories(obj_parallel PRIVATE ${INCLUDE_DIRS})
target_link_libraries(obj_parallel PRIVATE PkgConfig::PETSC)

foreach(target IN ITEMS ${PARALLEL_EXECS})
        add_executable(${target} ${target}.cpp)
        target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
        target_link_directories(${target} PRIVATE ${LIBRARY_DIRS})
        target_link_libraries(${target} obj_parallel)
        target_link_libraries(${target} nix)
        target_link_libraries(${target} PkgConfig::PETSC)
endforeach()

add_test(NAME test_petsc_utils_np1
        COMMAND mpiexec -n 1 $<TARGET_FILE:test_petsc_utils> "[np=1]"
        )

add_test(NAME test_petsc_utils_np8
        COMMAND mpiexec -n 8 $<TARGET_FILE:test_petsc_utils> "[np=8]"
        )