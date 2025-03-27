# project
set(PROJECT_NAME "unittest")
project(${PROJECT_NAME} CXX)

set(INCLUDE_DIRS
	${CMAKE_BINARY_DIR}/nix
	${CMAKE_BINARY_DIR}/nix/thirdparty
	)
set(LIBRARY_DIRS
	${CMAKE_BINARY_DIR}/nix
	)

#
# serial testing
#
set(SERIAL_EXECS
	test_balancer
	test_buffer
	test_sfc
	test_random
	test_nixio
	test_chunk
	test_chunkmap
	test_chunkvector
	test_primitives
	test_esirkepov
	test_interp
	test_xtensor_particle
	test_logger
	test_mpistream
	test_argparser
	test_cfgparser
	test_application
	)

add_library(obj_serial OBJECT test_serial.cpp)
target_include_directories(obj_serial PRIVATE ${INCLUDE_DIRS})
target_link_directories(obj_serial PRIVATE ${LIBRARY_DIRS})
target_link_libraries(obj_serial nix)

foreach(target IN ITEMS ${SERIAL_EXECS})
	add_executable(${target} ${target}.cpp)
	target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
	target_link_directories(${target} PRIVATE ${LIBRARY_DIRS})
	target_link_libraries(${target} obj_serial)
	target_link_libraries(${target} nix)
	target_link_libraries(${target} std::filesystem)
	add_test(NAME ${target} COMMAND ${target})
endforeach()

#
# parallel testing
#
set(PARALLEL_EXECS
	test_nixio_parallel
	test_balancer_parallel
	test_mpistream_parallel
	test_statehandler_parallel
	)

add_library(obj_parallel OBJECT test_parallel.cpp)
target_include_directories(obj_parallel PRIVATE ${INCLUDE_DIRS})
target_link_directories(obj_parallel PRIVATE ${LIBRARY_DIRS})
target_link_libraries(obj_parallel nix)

foreach(target IN ITEMS ${PARALLEL_EXECS})
	add_executable(${target} ${target}.cpp)
	target_include_directories(${target} PRIVATE ${INCLUDE_DIRS})
	target_link_directories(${target} PRIVATE ${LIBRARY_DIRS})
	target_link_libraries(${target} obj_parallel)
	target_link_libraries(${target} nix)
	target_link_libraries(${target} std::filesystem)
endforeach()

add_test(NAME test_nixio_parallel
	COMMAND mpiexec -n 8 ./test_nixio_parallel -X 2 -Y 2 -Z 2
	)

add_test(NAME test_balancer_parallel
	COMMAND mpiexec -n 8 ./test_balancer_parallel
	)

add_test(NAME test_mpistream_parallel
	COMMAND mpiexec -n 8 ./test_mpistream_parallel
	)

add_test(NAME test_statehandler_parallel
	COMMAND mpiexec -n 8 ./test_statehandler_parallel
	)
