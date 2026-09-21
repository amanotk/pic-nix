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

add_executable(hybrid_shared_state hybrid_shared_state.cpp)
target_link_libraries(hybrid_shared_state PRIVATE hybrid)

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

foreach(target IN ITEMS test_hybrid_boundary test_hybrid_particle_transfer test_hybrid_moment_parallel
                        test_hybrid_ssor2_parallel)
  add_executable(${target} ${target}.cpp)
  target_link_libraries(${target} PRIVATE Catch2::Catch2 hybrid_obj_parallel hybrid)
  add_hybrid_mpi_test(${target}_np2 ${target} 2)
  add_hybrid_mpi_test(${target}_np4 ${target} 4)
  add_hybrid_mpi_test(${target}_np8 ${target} 8)
endforeach()

if(TARGET hybrid_beam)
  set(shared_state_dir ${CMAKE_CURRENT_BINARY_DIR}/test_hybrid_shared_state)
  file(MAKE_DIRECTORY ${shared_state_dir})
  add_test(
    NAME test_hybrid_shared_state_run
    COMMAND
      ${MPIEXEC} -n 1 $<TARGET_FILE:hybrid_shared_state>
      -c ${CMAKE_CURRENT_LIST_DIR}/config_shared.toml -t 0.01
  )
  set_tests_properties(test_hybrid_shared_state_run PROPERTIES
    LABELS MPI
    PROCESSORS 1
    TIMEOUT 120
    WORKING_DIRECTORY ${shared_state_dir}
  )

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
  function(add_hybrid_restart_test name nproc config)
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
        -c ${config} -t 0.01
        --save ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_resume
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${config} -t 0.02
        --load ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_direct
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${config} -t 0.02
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

  add_hybrid_restart_test(test_hybrid_beam_restart_np1 1 ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml)
  add_hybrid_restart_test(test_hybrid_beam_restart_np2 2 ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config.toml)
  add_hybrid_restart_test(test_hybrid_beam_restart_np4 4 ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config_np4.toml)

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
    set_tests_properties(${name}_write_config PROPERTIES
      DEPENDS ${name}_prepare
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

  # --- Rebalance migration and restart equivalence test ---
  function(add_hybrid_rebalance_restart_test name nproc)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    set(checkpoint_dir ${tmpdir}/checkpoint)
    set(fixed_dir ${tmpdir}/fixed)
    set(save_dir ${tmpdir}/save)
    set(resume_dir ${tmpdir}/resume)
    set(direct_dir ${tmpdir}/direct)
    set(fixed_config ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config_rebalance_skew_fixed.toml)
    set(rebalance_config ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config_rebalance_skew.toml)
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${fixed_dir} ${save_dir} ${resume_dir} ${direct_dir})
    add_test(
      NAME ${name}_fixed
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${fixed_config} -t 0.03
    )
    add_test(
      NAME ${name}_save
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${rebalance_config} -t 0.02
        --save ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_resume
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${rebalance_config} -t 0.03
        --load ${checkpoint_dir}
    )
    add_test(
      NAME ${name}_direct
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${rebalance_config} -t 0.03
    )
    set_tests_properties(${name}_fixed
      PROPERTIES
        DEPENDS ${name}_prepare
        LABELS MPI
        PROCESSORS ${nproc}
        TIMEOUT 60
        WORKING_DIRECTORY ${fixed_dir}
    )
    set_tests_properties(${name}_save
      PROPERTIES
        DEPENDS ${name}_fixed
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

  add_hybrid_rebalance_restart_test(test_hybrid_beam_rebalance_restart_np2 2)

  # --- np4 variants (Cx=2,Cy=2,Cz=1) ---
  function(add_hybrid_np4_test name nproc tmax)
    set(tmpdir ${CMAKE_CURRENT_BINARY_DIR}/${name})
    set(config ${CMAKE_CURRENT_LIST_DIR}/../example/beam/config_np4.toml)
    add_test(NAME ${name}_prepare COMMAND ${CMAKE_COMMAND} -E make_directory ${tmpdir})
    add_test(
      NAME ${name}
      COMMAND
        ${MPIEXEC} -n ${nproc} $<TARGET_FILE:hybrid_beam>
        -c ${config} -t ${tmax}
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

  add_hybrid_np4_test(test_hybrid_beam_zero_step_np4 4 0)
  add_hybrid_np4_test(test_hybrid_beam_one_step_np4 4 0.01)
  add_hybrid_np4_test(test_hybrid_beam_short_run_np4 4 0.04)
  add_hybrid_np4_test(test_hybrid_beam_one_step_4chunk_np1 1 0.01)
  add_hybrid_np4_test(test_hybrid_beam_one_step_4chunk_np2 2 0.01)
endif()
