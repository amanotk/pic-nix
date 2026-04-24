# Graph Report - pic-nix  (2026-04-24)

## Corpus Check
- 142 files · ~129,770 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1009 nodes · 1627 edges · 53 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 355 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]

## God Nodes (most connected - your core abstractions)
1. `DiagHandler` - 26 edges
2. `Run` - 22 edges
3. `Run` - 18 edges
4. `fill_all()` - 18 edges
5. `build_exchange_context()` - 16 edges
6. `push_taskflow()` - 15 edges
7. `main()` - 13 edges
8. `push_openmp()` - 13 edges
9. `doit_job()` - 13 edges
10. `get_extent()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `set_boundary_query()` --calls--> `push_taskflow()`  [INFERRED]
  nix/chunk.cpp → pic/pic_application.cpp
- `copy_chunk_to_src()` --calls--> `pack()`  [INFERRED]
  elliptic/petsc_interface.cpp → pic/pic_chunk.cpp
- `copy_sol_to_chunk()` --calls--> `unpack()`  [INFERRED]
  elliptic/petsc_interface.cpp → pic/pic_chunk.cpp
- `get_interface()` --calls--> `initialize_diagnostic()`  [INFERRED]
  elliptic/elliptic_solver.cpp → pic/pic_application.cpp
- `get_interface()` --calls--> `update_poisson_efield()`  [INFERRED]
  elliptic/elliptic_solver.cpp → pic/pic_application.cpp

## Hyperedges (group relationships)
- **PIC Example Application Family** — example_example_collection, thermal_thermal_example, cherenkov_cherenkov_example, beam_beam_example, mrx_mrx_example, shock_shock_example, fireball_fireball_example [EXTRACTED 1.00]
- **PIC Mixed Serial and MPI Test Runner Pattern** — pic_unittest_suite, pic_serial_test_runner, pic_parallel_test_runner, pic_parallel_mpi_only_test_runner [INFERRED 0.84]
- **NIX Mixed Serial and MPI Test Runner Pattern** — nix_unittest_suite, nix_serial_test_runner, nix_parallel_test_runner [INFERRED 0.86]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (59): get_boundary_margin(), get_mpi_buffer(), get_nb_id(), MpiBuffer::pack(), MpiBuffer::unpack(), pack(), set_boundary_margin(), set_boundary_query() (+51 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (36): begin_bc_exchange(), get_nb_rank(), get_rcvtag(), get_sndtag(), get_xrange(), get_yrange(), get_zrange(), has_xdim() (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (32): match(), doit_job(), convert_tracer_to_hdf5(), is_valid_tracer_hdf5(), remove_tracer_file_after_confirmation(), sort_and_split_particle_id(), Tracer, allocate_memory() (+24 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (39): get_delx(), calculate_moment(), calculate_moment_openmp(), calculate_moment_taskflow(), create_poisson_interface(), exchange_emf_boundaries(), exchange_phi_boundaries(), initialize() (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (43): assert_mpi(), diagnostic(), finalize(), finalize_mpi(), from_json(), get_available_etime(), get_basedir(), get_iomode() (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (26): doit_job(), Run, IntegrationCase, Histogram2D, generate_plots(), _generate_snapshot(), _get_profile_path(), _load_picnix() (+18 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (28): apply_petsc_option(), create_dm(), create_dm1d(), create_dm2d(), create_dm3d(), destroy_petsc_objects(), float_to_string(), int_to_string() (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (22): build_poisson_matrix_1d(), build_poisson_matrix_2d(), build_poisson_matrix_3d(), PetscPoisson, PetscPoisson1D, PetscPoisson2D, PetscPoisson3D, set_matrix() (+14 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (36): BaseCurrent, call_scalar_impl(), call_vector_impl(), global1d(), global2d(), global3d(), local1d(), local2d() (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (8): async_read_time_and_step(), create_handler(), DiagHandler, FieldDiagHandler, LoadDiagHandler, ParticleDiagHandler, read_time_and_step(), TracerDiagHandler

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (20): get_id(), FileSystemEventHandler, doit(), copy_chunk_to_src(), from_json(), to_json(), pack(), load() (+12 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (28): object, sort_particle(), make_boundary_config(), apply_sinusoidal_field(), capture_smoke_state(), compare_smoke(), compare_smoke_diagnostics(), compare_smoke_state() (+20 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (22): add_chunkmap(), add_dataset(), create_chunked_dataset(), create_vds(), doit_parallel(), is_chunked_dataset(), is_external_contiguous(), json2hdf5() (+14 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (21): BasePosition, call_scalar_impl(), call_vector_impl(), push(), ScalarPosition, VectorPosition, BaseVelocity, call_scalar_impl() (+13 more)

### Community 14 - "Community 14"
Cohesion: 0.1
Nodes (10): ChunkAccessor, flatten_index(), get_dims(), get_offset(), bind_chunks(), bind_chunks_impl(), get_accessor(), PicChunkAccessor (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (15): calculate_global_offset(), get_attribute(), get_metadata(), get_size(), put_attribute(), put_metadata(), read_contiguous(), read_contiguous_at() (+7 more)

### Community 16 - "Community 16"
Cohesion: 0.26
Nodes (17): namespace(), analyze_run(), cmd_all(), cmd_analyze(), cmd_build(), cmd_compare(), cmd_plots(), cmd_run() (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (4): test_conservation2d(), test_conservation3d(), test_deposit2d_scalar(), test_deposit3d_scalar()

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (1): BaseMaxwell

### Community 19 - "Community 19"
Cohesion: 0.28
Nodes (11): validate(), check_index(), check_locality3d(), forward_id_2d(), forward_id_3d(), get_map1d(), get_map2d(), get_map3d() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.32
Nodes (7): count_cell_within_fireball(), generate_injection_particle(), is_inside_fireball(), main(), MainApplication, MainChunk, MainInterface

### Community 21 - "Community 21"
Cohesion: 0.24
Nodes (6): copy_chunk_to_src(), copy_sol_to_chunk(), scatter_forward(), scatter_reverse(), solve(), update_mapping()

### Community 22 - "Community 22"
Cohesion: 0.23
Nodes (8): set_id(), get_chunklist(), get_chunkvec(), get_index_and_chunkvec(), get_mpi_size(), get_rank_dims(), MockChunk, MockChunkAccessor

### Community 23 - "Community 23"
Cohesion: 0.21
Nodes (9): analytic_solution(), analytic_source(), cleanup_config_and_tmpdir(), cleanup_tmpdir(), replace_all(), TestApplication, TestChunk, TestInterface (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.18
Nodes (4): MockApplicationInterface, MockChunk, MockChunkMap, TestBalancer

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (1): PicPoissonBasic()

### Community 26 - "Community 26"
Cohesion: 0.35
Nodes (10): apply_operator_1d_primitive(), apply_operator_2d_primitive(), apply_operator_3d_primitive(), flat_index(), preconditioner_backward_1d_primitive(), preconditioner_backward_2d_primitive(), preconditioner_backward_3d_primitive(), preconditioner_forward_1d_primitive() (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.29
Nodes (5): class(), Diag(), format_dirname(), make_sure_directory_exists(), PicDiag

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (4): PickupTracerDiag, PickupTracerPacker, TracerDiag, TracerPacker

### Community 30 - "Community 30"
Cohesion: 0.25
Nodes (3): MockApplication, MockApplicationInterface, MockChunk

### Community 31 - "Community 31"
Cohesion: 0.29
Nodes (3): MpiStream, Singleton, teebuf

### Community 32 - "Community 32"
Cohesion: 0.48
Nodes (4): get_mpi_rank(), get_mpi_size(), main(), require_mpi_size()

### Community 33 - "Community 33"
Cohesion: 0.38
Nodes (3): get_nprocess(), get_thisrank(), parallel_decomposition()

### Community 34 - "Community 34"
Cohesion: 0.29
Nodes (1): MockChunk

### Community 35 - "Community 35"
Cohesion: 0.57
Nodes (5): apply_preconditioner_reference(), is_interior(), primitive_index(), require_operator_result(), require_preconditioner_result()

### Community 36 - "Community 36"
Cohesion: 0.29
Nodes (3): FieldDiag, FieldPacker, MomentPacker

### Community 37 - "Community 37"
Cohesion: 0.29
Nodes (3): MpiioHandler, PicDiagHandler, PosixHandler

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (3): Application, ApplicationInterface, Diag

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (3): ChunkVectorTest, MockChunk, MockChunkMap

### Community 40 - "Community 40"
Cohesion: 0.33
Nodes (5): PicApplication, PicApplicationInterface, PicChunk, PicDiag, PicPacker

### Community 41 - "Community 41"
Cohesion: 0.33
Nodes (3): LoadDiag, LoadPacker, RankPacker

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (3): percentile(), ResourceDiag, statistics()

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (2): ParticleDiag, ParticlePacker

### Community 44 - "Community 44"
Cohesion: 0.5
Nodes (2): Solver, SolverInterface

### Community 47 - "Community 47"
Cohesion: 0.67
Nodes (2): DebugFormatter, DebugPrinter

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (1): TestApplication

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (1): TestChunk

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (1): TestPicPoisson

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (1): HistoryDiag

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (1): ParallelDiag

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): ChunkMapTest

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): TestLogger

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): PicPacker

## Knowledge Gaps
- **51 isolated node(s):** `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file`, `Add dataset to HDF5 file`, `Add chunkmap to HDF5 file` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (15 nodes): `BaseMaxwell`, `compute_efield_from_potential_1d()`, `compute_efield_from_potential_2d()`, `compute_efield_from_potential_3d()`, `get_diverror_1d()`, `get_diverror_2d()`, `get_diverror_3d()`, `init_friedman()`, `push_bfd_1d()`, `push_bfd_2d()`, `push_bfd_3d()`, `push_efd_1d()`, `push_efd_2d()`, `push_efd_3d()`, `maxwell.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (11 nodes): `pic_poisson_basic.cpp`, `pic_poisson_basic.hpp`, `copy_chunk_to_src()`, `copy_sol_to_chunk()`, `Impl()`, `PicPoissonBasic()`, `scatter_forward()`, `scatter_reverse()`, `set_option()`, `solve()`, `update_mapping()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (7 nodes): `test_xtensor_particle.cpp`, `check_sort1d()`, `check_sort2d()`, `check_sort3d()`, `MockChunk`, `.MockChunk()`, `set_random_particle()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (5 nodes): `ParticleDiag`, `.ParticleDiag()`, `ParticlePacker`, `.ParticlePacker()`, `particle.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (4 nodes): `elliptic.hpp`, `Solver`, `SolverInterface`, `.SolverInterface()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (3 nodes): `DebugFormatter`, `DebugPrinter`, `debug.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (3 nodes): `test_application.cpp`, `TestApplication`, `.TestApplication()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (3 nodes): `test_chunk.cpp`, `TestChunk`, `.TestChunk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (3 nodes): `test_pic_poisson_petsc.cpp`, `make_default_option()`, `TestPicPoisson`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (3 nodes): `HistoryDiag`, `.HistoryDiag()`, `history.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (3 nodes): `ParallelDiag`, `.ParallelDiag()`, `parallel.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `test_chunkmap.cpp`, `ChunkMapTest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (2 nodes): `test_logger.cpp`, `TestLogger`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (2 nodes): `PicPacker`, `pic_packer.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Run` connect `Community 5` to `Community 32`, `Community 0`, `Community 2`, `Community 3`, `Community 16`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `load()` connect `Community 10` to `Community 2`, `Community 4`, `Community 9`, `Community 12`, `Community 16`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `unpack()` connect `Community 0` to `Community 1`, `Community 10`, `Community 3`, `Community 12`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `Run` (e.g. with `main()` and `doit_job()`) actually correct?**
  _`Run` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `fill_all()` (e.g. with `test_append_current1d_scalar()` and `test_append_current1d_xsimd()`) actually correct?**
  _`fill_all()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `build_exchange_context()` (e.g. with `get_mpi_size()` and `get_mpi_rank()`) actually correct?**
  _`build_exchange_context()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file` to the rest of the system?**
  _51 weakly-connected nodes found - possible documentation gaps or missing edges._