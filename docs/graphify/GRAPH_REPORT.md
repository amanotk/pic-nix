# Graph Report - pic-nix  (2026-04-24)

## Corpus Check
- 146 files · ~131,200 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1037 nodes · 1648 edges · 64 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 360 edges (avg confidence: 0.8)
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
- [[_COMMUNITY_Community 29|Community 29]]
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
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]

## God Nodes (most connected - your core abstractions)
1. `DiagHandler` - 26 edges
2. `Run` - 22 edges
3. `Run` - 18 edges
4. `fill_all()` - 18 edges
5. `push_taskflow()` - 16 edges
6. `build_exchange_context()` - 16 edges
7. `main()` - 13 edges
8. `push_openmp()` - 13 edges
9. `doit_job()` - 13 edges
10. `get_extent()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `copy_chunk_to_src()` --calls--> `pack()`  [INFERRED]
  elliptic/petsc_interface.cpp → pic/pic_chunk.cpp
- `get_boundary_margin()` --calls--> `build_exchange_context()`  [INFERRED]
  nix/chunk.hpp → pic/unittest/test_pic_boundary.cpp
- `set_boundary_query()` --calls--> `push_taskflow()`  [INFERRED]
  nix/chunk.cpp → pic/pic_application.cpp
- `copy_sol_to_chunk()` --calls--> `unpack()`  [INFERRED]
  elliptic/petsc_interface.cpp → pic/pic_chunk.cpp
- `get_interface()` --calls--> `initialize_diagnostic()`  [INFERRED]
  elliptic/elliptic_solver.cpp → pic/pic_application.cpp

## Hyperedges (group relationships)
- **PIC Simulation Build-Run-Analyze Pipeline** — integration_workflow, example_beam, example_shock, golden_data_system, deterministic_seeding, integration_case_dataclass [EXTRACTED 0.95]
- **Layered Diagnostic Architecture (nix mechanics + module interpretation)** — nix_diag_ownership, pic_diag_ownership, packer_contract, module_extension_model [EXTRACTED 0.95]
- **Three-Module Dependency Chain (pic -> elliptic -> nix)** — pic_module, elliptic_module, nix_module [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (38): begin_bc_exchange(), get_boundary_margin(), get_mpi_buffer(), get_nb_rank(), get_rcvtag(), get_sndtag(), get_xrange(), get_yrange() (+30 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (31): doit_job(), convert_tracer_to_hdf5(), is_valid_tracer_hdf5(), remove_tracer_file_after_confirmation(), sort_and_split_particle_id(), Tracer, allocate_memory(), async_do_read_at() (+23 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (40): get_delx(), set_mpi_communicator(), calculate_moment(), calculate_moment_openmp(), calculate_moment_taskflow(), create_poisson_interface(), exchange_emf_boundaries(), exchange_phi_boundaries() (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (44): assert_mpi(), diagnostic(), finalize(), finalize_mpi(), from_json(), get_available_etime(), get_basedir(), get_iomode() (+36 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (26): doit_job(), Run, IntegrationCase, Histogram2D, generate_plots(), _generate_snapshot(), _get_profile_path(), _load_picnix() (+18 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (34): MpiBuffer::pack(), MpiBuffer::unpack(), pack(), set_boundary_margin(), set_boundary_query(), set_coordinate(), set_global_context(), set_mpi_buffer() (+26 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (28): get_nb_id(), ChunkMap(), get_chunkid(), get_coordinate(), get_neighbor_coord(), get_rank(), get_rank_boundary(), is_chunk_active() (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (26): get_id(), get_chunk_id_range(), is_completed(), require_diagnostic(), test_all(), wait(), wait_all(), write_packed_chunks() (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (36): BaseCurrent, call_scalar_impl(), call_vector_impl(), global1d(), global2d(), global3d(), local1d(), local2d() (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (8): async_read_time_and_step(), create_handler(), DiagHandler, FieldDiagHandler, LoadDiagHandler, ParticleDiagHandler, read_time_and_step(), TracerDiagHandler

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (24): apply_petsc_option(), copy_chunk_to_src(), create_dm(), create_dm1d(), create_dm2d(), create_dm3d(), destroy_petsc_objects(), float_to_string() (+16 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (27): object, sort_particle(), apply_sinusoidal_field(), capture_smoke_state(), compare_smoke(), compare_smoke_diagnostics(), compare_smoke_state(), compute_smoke_field() (+19 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (13): copy_chunk_to_src(), copy_sol_to_chunk(), Impl, PicPoissonPetsc(), PicPoissonPetsc::Impl, scatter_forward(), scatter_reverse(), solve() (+5 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (22): add_chunkmap(), add_dataset(), create_chunked_dataset(), create_vds(), doit_parallel(), is_chunked_dataset(), is_external_contiguous(), json2hdf5() (+14 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (21): BasePosition, call_scalar_impl(), call_vector_impl(), push(), ScalarPosition, VectorPosition, BaseVelocity, call_scalar_impl() (+13 more)

### Community 15 - "Community 15"
Cohesion: 0.1
Nodes (10): ChunkAccessor, flatten_index(), get_dims(), get_offset(), bind_chunks(), bind_chunks_impl(), get_accessor(), PicChunkAccessor (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (15): calculate_global_offset(), get_attribute(), get_metadata(), get_size(), put_attribute(), put_metadata(), read_contiguous(), read_contiguous_at() (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.26
Nodes (17): namespace(), analyze_run(), cmd_all(), cmd_analyze(), cmd_build(), cmd_compare(), cmd_plots(), cmd_run() (+9 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (4): test_conservation2d(), test_conservation3d(), test_deposit2d_scalar(), test_deposit3d_scalar()

### Community 19 - "Community 19"
Cohesion: 0.18
Nodes (9): update_mapping(), get_indexset(), get_indexset_global(), get_indexset_local(), PetscScatter, setup_indexset_global(), setup_indexset_local(), setup_scatter() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (1): BaseMaxwell

### Community 21 - "Community 21"
Cohesion: 0.28
Nodes (11): validate(), check_index(), check_locality3d(), forward_id_2d(), forward_id_3d(), get_map1d(), get_map2d(), get_map3d() (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.32
Nodes (7): count_cell_within_fireball(), generate_injection_particle(), is_inside_fireball(), main(), MainApplication, MainChunk, MainInterface

### Community 23 - "Community 23"
Cohesion: 0.24
Nodes (6): copy_chunk_to_src(), copy_sol_to_chunk(), scatter_forward(), scatter_reverse(), solve(), update_mapping()

### Community 24 - "Community 24"
Cohesion: 0.23
Nodes (8): set_id(), get_chunklist(), get_chunkvec(), get_index_and_chunkvec(), get_mpi_size(), get_rank_dims(), MockChunk, MockChunkAccessor

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (9): analytic_solution(), analytic_source(), cleanup_config_and_tmpdir(), cleanup_tmpdir(), replace_all(), TestApplication, TestChunk, TestInterface (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.21
Nodes (12): CMake Compiler Profiles (cmake/linux-gcc.cmake, cmake/linux-intel.cmake), Deterministic Seeding (seed_type=fixed), Golden Data System (msgpack + json), IntegrationCase Dataclass Registry, PIC Integration Workflow (Python), Python Script Dependencies (numpy, matplotlib, msgpack, h5py, toml, etc.), Open boundaries with particle injection require rebalance, Reduced grid parameters for integration tests (performance vs physics trade-off) (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.18
Nodes (4): MockApplicationInterface, MockChunk, MockChunkMap, TestBalancer

### Community 28 - "Community 28"
Cohesion: 0.18
Nodes (1): PicPoissonBasic()

### Community 29 - "Community 29"
Cohesion: 0.35
Nodes (10): apply_operator_1d_primitive(), apply_operator_2d_primitive(), apply_operator_3d_primitive(), flat_index(), preconditioner_backward_1d_primitive(), preconditioner_backward_2d_primitive(), preconditioner_backward_3d_primitive(), preconditioner_forward_1d_primitive() (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.25
Nodes (3): MockApplication, MockApplicationInterface, MockChunk

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (3): MpiStream, Singleton, teebuf

### Community 33 - "Community 33"
Cohesion: 0.48
Nodes (4): get_mpi_rank(), get_mpi_size(), main(), require_mpi_size()

### Community 34 - "Community 34"
Cohesion: 0.38
Nodes (3): get_nprocess(), get_thisrank(), parallel_decomposition()

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (1): MockChunk

### Community 36 - "Community 36"
Cohesion: 0.57
Nodes (5): apply_preconditioner_reference(), is_interior(), primitive_index(), require_operator_result(), require_preconditioner_result()

### Community 37 - "Community 37"
Cohesion: 0.29
Nodes (3): FieldDiag, FieldPacker, MomentPacker

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (3): Application, ApplicationInterface, Diag

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (3): ChunkVectorTest, MockChunk, MockChunkMap

### Community 40 - "Community 40"
Cohesion: 0.33
Nodes (2): MpiioDiagIoHandler, PosixDiagIoHandler

### Community 41 - "Community 41"
Cohesion: 0.33
Nodes (5): PicApplication, PicApplicationInterface, PicChunk, PicDiag, PicPacker

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (2): MockPacker, MockResourceDiagBase

### Community 43 - "Community 43"
Cohesion: 0.6
Nodes (3): percentile(), savefile(), statistics()

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (2): TracerDiag, TracerPacker

### Community 45 - "Community 45"
Cohesion: 0.4
Nodes (2): TracerPickupDiag, TracerPickupPacker

### Community 46 - "Community 46"
Cohesion: 0.4
Nodes (2): ParticleDiag, ParticlePacker

### Community 47 - "Community 47"
Cohesion: 0.5
Nodes (2): Solver, SolverInterface

### Community 49 - "Community 49"
Cohesion: 0.5
Nodes (2): LoadPacker, RankPacker

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (2): DebugFormatter, DebugPrinter

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (1): TestApplication

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (1): TestChunk

### Community 55 - "Community 55"
Cohesion: 0.67
Nodes (1): PicDiag

### Community 56 - "Community 56"
Cohesion: 0.67
Nodes (1): TestPicPoisson

### Community 57 - "Community 57"
Cohesion: 0.67
Nodes (1): HistoryDiag

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): ChunkMapTest

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): TestLogger

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): PicPacker

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (2): Native Async I/O (Deferred), Async I/O deferred due to MPI buffer lifetime risks and future SENSEI/Ascent alternatives

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (2): Future Hybrid Module, Module Extension Model (nix owns mechanics, modules own interpretation)

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (2): Diagnostic Module Refactor, Diagnostic refactor designed as mechanical move with minimal behavior change

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): PIC-NIX Project

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (1): Catch2 v3 Test Framework

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (1): nix External Repository (github.com/amanotk/nix)

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (1): PIC module auto-enables elliptic and nix modules

## Knowledge Gaps
- **66 isolated node(s):** `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file`, `Add dataset to HDF5 file`, `Add chunkmap to HDF5 file` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 20`** (15 nodes): `BaseMaxwell`, `compute_efield_from_potential_1d()`, `compute_efield_from_potential_2d()`, `compute_efield_from_potential_3d()`, `get_diverror_1d()`, `get_diverror_2d()`, `get_diverror_3d()`, `init_friedman()`, `push_bfd_1d()`, `push_bfd_2d()`, `push_bfd_3d()`, `push_efd_1d()`, `push_efd_2d()`, `push_efd_3d()`, `maxwell.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (11 nodes): `pic_poisson_basic.cpp`, `pic_poisson_basic.hpp`, `copy_chunk_to_src()`, `copy_sol_to_chunk()`, `Impl()`, `PicPoissonBasic()`, `scatter_forward()`, `scatter_reverse()`, `set_option()`, `solve()`, `update_mapping()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (7 nodes): `test_xtensor_particle.cpp`, `check_sort1d()`, `check_sort2d()`, `check_sort3d()`, `MockChunk`, `.MockChunk()`, `set_random_particle()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (6 nodes): `DiagIoHandler()`, `MpiioDiagIoHandler`, `.MpiioDiagIoHandler()`, `PosixDiagIoHandler`, `.PosixDiagIoHandler()`, `io_handler.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (5 nodes): `test_diag_resource.cpp`, `MockPacker`, `.MockPacker()`, `MockResourceDiagBase`, `.MockResourceDiagBase()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (5 nodes): `tracer.hpp`, `TracerDiag`, `.TracerDiag()`, `TracerPacker`, `.TracerPacker()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (5 nodes): `tracer_pickup.hpp`, `TracerPickupDiag`, `.TracerPickupDiag()`, `TracerPickupPacker`, `.TracerPickupPacker()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (5 nodes): `ParticleDiag`, `.ParticleDiag()`, `ParticlePacker`, `.ParticlePacker()`, `particle.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (4 nodes): `elliptic.hpp`, `Solver`, `SolverInterface`, `.SolverInterface()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (4 nodes): `LoadPacker`, `RankPacker`, `.RankPacker()`, `load.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (3 nodes): `DebugFormatter`, `DebugPrinter`, `debug.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (3 nodes): `test_application.cpp`, `TestApplication`, `.TestApplication()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (3 nodes): `test_chunk.cpp`, `TestChunk`, `.TestChunk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (3 nodes): `PicDiag`, `.PicDiag()`, `pic_diag.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (3 nodes): `test_pic_poisson_petsc.cpp`, `make_default_option()`, `TestPicPoisson`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (3 nodes): `HistoryDiag`, `.HistoryDiag()`, `history.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (2 nodes): `test_chunkmap.cpp`, `ChunkMapTest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (2 nodes): `test_logger.cpp`, `TestLogger`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (2 nodes): `PicPacker`, `pic_packer.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (2 nodes): `Native Async I/O (Deferred)`, `Async I/O deferred due to MPI buffer lifetime risks and future SENSEI/Ascent alternatives`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (2 nodes): `Future Hybrid Module`, `Module Extension Model (nix owns mechanics, modules own interpretation)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (2 nodes): `Diagnostic Module Refactor`, `Diagnostic refactor designed as mechanical move with minimal behavior change`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `PIC-NIX Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (1 nodes): `Catch2 v3 Test Framework`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (1 nodes): `nix External Repository (github.com/amanotk/nix)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (1 nodes): `PIC module auto-enables elliptic and nix modules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Run` connect `Community 4` to `Community 33`, `Community 2`, `Community 1`, `Community 6`, `Community 17`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `load()` connect `Community 7` to `Community 1`, `Community 3`, `Community 9`, `Community 13`, `Community 17`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `push_taskflow()` connect `Community 2` to `Community 4`, `Community 5`, `Community 7`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `Run` (e.g. with `main()` and `doit_job()`) actually correct?**
  _`Run` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `fill_all()` (e.g. with `test_append_current1d_scalar()` and `test_append_current1d_xsimd()`) actually correct?**
  _`fill_all()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `push_taskflow()` (e.g. with `reset_load()` and `push_bfd()`) actually correct?**
  _`push_taskflow()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._