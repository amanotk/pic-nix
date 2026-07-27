if(NOT DEFINED MPIEXEC)
  find_program(MPIEXEC NAMES mpiexec mpirun REQUIRED)
endif()

include(${PICNIX_DIR}/nix/cmake/Catch2Setup.cmake)

add_library(hybrid_obj_serial OBJECT test_serial.cpp)
target_link_libraries(hybrid_obj_serial PRIVATE Catch2::Catch2 hybrid)

add_library(hybrid_obj_parallel OBJECT test_parallel.cpp)
target_link_libraries(hybrid_obj_parallel PRIVATE Catch2::Catch2 hybrid)

add_executable(test_hybrid_chunk test_hybrid_chunk.cpp)
target_link_libraries(test_hybrid_chunk PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_chunk COMMAND test_hybrid_chunk)

add_executable(test_hybrid_particle test_hybrid_particle.cpp)
target_link_libraries(test_hybrid_particle PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_particle COMMAND test_hybrid_particle)

add_executable(test_hybrid_moment test_hybrid_moment.cpp)
target_link_libraries(test_hybrid_moment PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_moment COMMAND test_hybrid_moment)

add_executable(test_hybrid_fluid test_hybrid_fluid.cpp)
target_link_libraries(test_hybrid_fluid PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_fluid COMMAND test_hybrid_fluid)

add_executable(test_hybrid_field test_hybrid_field.cpp)
target_link_libraries(test_hybrid_field PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_field COMMAND test_hybrid_field)

add_executable(test_hybrid_phasespeed test_hybrid_phasespeed.cpp)
target_link_libraries(test_hybrid_phasespeed PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_phasespeed COMMAND test_hybrid_phasespeed)

add_executable(test_hybrid_mc2 test_hybrid_mc2.cpp)
target_link_libraries(test_hybrid_mc2 PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_mc2 COMMAND test_hybrid_mc2)

add_executable(test_hybrid_stage test_hybrid_stage.cpp)
target_link_libraries(test_hybrid_stage PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_stage COMMAND test_hybrid_stage)

add_executable(test_hybrid_ohm test_hybrid_ohm.cpp)
target_link_libraries(test_hybrid_ohm PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_ohm COMMAND test_hybrid_ohm)

add_executable(test_hybrid_ssor2 test_hybrid_ssor2.cpp)
target_link_libraries(test_hybrid_ssor2 PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_ssor2 COMMAND test_hybrid_ssor2)

add_executable(test_hybrid_pcc2 test_hybrid_pcc2.cpp)
target_link_libraries(test_hybrid_pcc2 PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_pcc2 COMMAND test_hybrid_pcc2)

add_executable(test_hybrid_ohm_bridge test_hybrid_ohm_bridge.cpp)
target_link_libraries(test_hybrid_ohm_bridge PRIVATE Catch2::Catch2 hybrid_obj_serial hybrid)
add_test(NAME test_hybrid_ohm_bridge COMMAND test_hybrid_ohm_bridge)

function(add_hybrid_mpi_test name target nproc)
  set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
  file(MAKE_DIRECTORY ${tmpdir})
  add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
  add_test(NAME ${name} COMMAND ${MPIEXEC} -n ${nproc} $<TARGET_FILE:${target}>)
  set_tests_properties(${name}
    PROPERTIES
      DEPENDS ${name}_prepare
      LABELS MPI
      PROCESSORS ${nproc}
      TIMEOUT 120
      WORKING_DIRECTORY ${tmpdir}
  )
endfunction()

foreach(target IN ITEMS test_hybrid_boundary test_hybrid_particle_transfer test_hybrid_moment_parallel)
  add_executable(${target} ${target}.cpp)
  target_link_libraries(${target} PRIVATE Catch2::Catch2 hybrid_obj_parallel hybrid)
  add_hybrid_mpi_test(${target}_np2 ${target} 2)
  add_hybrid_mpi_test(${target}_np4 ${target} 4)
  add_hybrid_mpi_test(${target}_np8 ${target} 8)
endforeach()

if(TARGET hybrid_beam)
  function(add_hybrid_zero_step_test name nproc)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
    add_test(
      NAME ${name}
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t 0
    )
    set_tests_properties(${name}
      PROPERTIES
        DEPENDS ${name}_prepare
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${tmpdir}
    )
  endfunction()

  add_hybrid_zero_step_test(test_hybrid_beam_zero_step_np1 1)
  add_hybrid_zero_step_test(test_hybrid_beam_zero_step_np2 2)

  function(add_hybrid_one_step_test name nproc)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
    add_test(
      NAME ${name}
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t 0.01
    )
    set_tests_properties(${name}
      PROPERTIES
        DEPENDS ${name}_prepare
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${tmpdir}
    )
  endfunction()

  add_hybrid_one_step_test(test_hybrid_beam_one_step_np1 1)
  add_hybrid_one_step_test(test_hybrid_beam_one_step_np2 2)

  function(add_hybrid_short_run_test name nproc tmax)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
    add_test(
      NAME ${name}
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t ${tmax}
    )
    set_tests_properties(${name}
      PROPERTIES
        DEPENDS ${name}_prepare
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 120
        WORKING_DIRECTORY ${tmpdir}
    )
  endfunction()

  add_hybrid_short_run_test(test_hybrid_beam_short_run_np1 1 0.04)
  add_hybrid_short_run_test(test_hybrid_beam_short_run_np2 2 0.04)

  # --- Restart test ---
  function(add_hybrid_restart_test name nproc)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    set(checkpoint_dir ${tmpdir}/checkpoint)
    set(save_dir ${tmpdir}/save)
    set(resume_dir ${tmpdir}/resume)
    set(direct_dir ${tmpdir}/direct)
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${save_dir} ${resume_dir} ${direct_dir})
    add_test(
      NAME ${name}_save
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t 0.01
        --save ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_resume
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t 0.02
        --load ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_direct
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml -t 0.02
    )
    set_tests_properties(${name}_save
      PROPERTIES
        DEPENDS ${name}_prepare
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${save_dir}
    )
    set_tests_properties(${name}_resume
      PROPERTIES
        DEPENDS ${name}_save
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${resume_dir}
    )
    set_tests_properties(${name}_direct
      PROPERTIES
        DEPENDS ${name}_resume
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${direct_dir}
    )
  endfunction()

  add_hybrid_restart_test(test_hybrid_beam_restart_np1 1)
  add_hybrid_restart_test(test_hybrid_beam_restart_np2 2)

  # --- Rebalance path test (interval=1, exercises communicator rebinding) ---
  function(add_hybrid_rebalance_test name nproc)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    set(rebalance_toml ${tmpdir}/config.toml)
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
    add_test(
      NAME ${name}_write_config
      COMMAND
        ${CMAKE_COMMAND} -E
        copy ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config_rebalance.toml ${rebalance_toml}
    )
    add_test(
      NAME ${name}_run
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${rebalance_toml} -t 0.02
    )
    set_tests_properties(${name}_run
      PROPERTIES
        DEPENDS ${name}_write_config
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${tmpdir}
    )
  endfunction()

  add_hybrid_rebalance_test(test_hybrid_beam_rebalance_np1 1)
  add_hybrid_rebalance_test(test_hybrid_beam_rebalance_np2 2)
endif()
