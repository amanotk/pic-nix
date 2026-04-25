# Graph Report - pic-nix  (2026-04-25)

## Corpus Check
- 146 files · ~131,481 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1000 nodes · 1631 edges · 61 communities detected
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 388 edges (avg confidence: 0.8)
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
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]

## God Nodes (most connected - your core abstractions)
1. `size()` - 57 edges
2. `data()` - 28 edges
3. `DiagHandler` - 26 edges
4. `Run` - 20 edges
5. `Run` - 18 edges
6. `fill_all()` - 18 edges
7. `build_exchange_context()` - 16 edges
8. `push_openmp()` - 14 edges
9. `main()` - 13 edges
10. `doit_job()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `get_interface()` --calls--> `initialize_diagnostic()`  [INFERRED]
  elliptic/elliptic_solver.cpp → pic/pic_application.cpp
- `setup()` --calls--> `load()`  [INFERRED]
  script/syncdir.py → nix/statehandler.hpp
- `PicChunkAccessor::get_num_chunks()` --calls--> `size()`  [INFERRED]
  pic/pic_poisson.cpp → nix/chunk.hpp
- `get_size_byte()` --calls--> `size()`  [INFERRED]
  pic/pic_chunk.cpp → nix/chunk.hpp
- `create_dm()` --calls--> `size()`  [INFERRED]
  elliptic/petsc_interface.cpp → nix/chunk.hpp

## Hyperedges (group relationships)
- **PIC Simulation Build-Run-Analyze Pipeline** — integration_workflow, example_beam, example_shock, golden_data_system, deterministic_seeding, integration_case_dataclass [EXTRACTED 0.95]
- **Layered Diagnostic Architecture (nix mechanics + module interpretation)** — nix_diag_ownership, pic_diag_ownership, packer_contract, module_extension_model [EXTRACTED 0.95]
- **Three-Module Dependency Chain (pic -> elliptic -> nix)** — pic_module, elliptic_module, nix_module [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (34): async_read_time_and_step(), create_handler(), DiagHandler, FieldDiagHandler, LoadDiagHandler, ParticleDiagHandler, read_time_and_step(), TracerDiagHandler (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (57): DEFINE_MEMBER(), data(), MpiBuffer::get_size_byte(), MpiBuffer::pack(), MpiBuffer::unpack(), pack(), pack_bc_exchange(), set_boundary_query() (+49 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (40): initialize_workload(), setup_chunks_init(), set_coordinate(), set_global_context(), get_chunkid(), get_coordinate(), get_neighbor_coord(), get_rank() (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (30): IntegrationCase, Histogram2D, doit_job(), Run, generate_plots(), _generate_snapshot(), _get_profile_path(), _load_picnix() (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (44): begin_bc_exchange(), probe_bc_exchange(), calculate_moment(), calculate_moment_openmp(), exchange_emf_boundaries(), exchange_phi_boundaries(), initialize_diagnostic(), PicApplication (+36 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (42): assert_mpi(), diagnostic(), finalize(), finalize_mpi(), from_json(), get_available_etime(), get_basedir(), get_iomode() (+34 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (36): BaseCurrent, call_scalar_impl(), call_vector_impl(), global1d(), global2d(), global3d(), local1d(), local2d() (+28 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (18): create_poisson_interface(), initialize(), make_poisson_solver(), copy_chunk_to_src(), copy_sol_to_chunk(), Impl, PicPoissonPetsc(), PicPoissonPetsc::Impl (+10 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (22): fill(), get_energy(), get_order(), get_chunklist(), get_chunkvec(), get_index_and_chunkvec(), get_mpi_size(), get_rank_dims() (+14 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (25): object, apply_sinusoidal_field(), capture_smoke_state(), compare_smoke(), compare_smoke_diagnostics(), compare_smoke_state(), compute_smoke_field(), compute_smoke_particle() (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (19): apply_petsc_option(), create_dm(), create_dm1d(), create_dm2d(), create_dm3d(), destroy_petsc_objects(), float_to_string(), int_to_string() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (22): add_chunkmap(), add_dataset(), create_chunked_dataset(), create_vds(), doit_parallel(), is_chunked_dataset(), is_external_contiguous(), json2hdf5() (+14 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (21): BasePosition, call_scalar_impl(), call_vector_impl(), push(), ScalarPosition, VectorPosition, BaseVelocity, call_scalar_impl() (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.16
Nodes (15): calculate_global_offset(), get_attribute(), get_metadata(), get_size(), put_attribute(), put_metadata(), read_contiguous(), read_contiguous_at() (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.15
Nodes (13): set_mpi_buffer(), allocate_mpi_buffers(), compute_rank_from_chunk_coords(), flatten_chunk_index(), initialize_chunkvec(), make_chunk_grid_dims(), make_rank_coords(), make_test_chunk() (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.26
Nodes (17): namespace(), analyze_run(), cmd_all(), cmd_analyze(), cmd_build(), cmd_compare(), cmd_plots(), cmd_run() (+9 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (9): ChunkAccessor, flatten_index(), bind_chunks(), bind_chunks_impl(), get_accessor(), PicChunkAccessor, PicChunkAccessor::build_global_index(), PicChunkAccessor::get_num_chunks() (+1 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (4): test_conservation2d(), test_conservation3d(), test_deposit2d_scalar(), test_deposit3d_scalar()

### Community 18 - "Community 18"
Cohesion: 0.23
Nodes (12): ChunkMap(), validate(), check_index(), check_locality3d(), forward_id_2d(), forward_id_3d(), get_map1d(), get_map2d() (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (1): BaseMaxwell

### Community 20 - "Community 20"
Cohesion: 0.27
Nodes (5): convert_tracer_to_hdf5(), is_valid_tracer_hdf5(), remove_tracer_file_after_confirmation(), sort_and_split_particle_id(), Tracer

### Community 21 - "Community 21"
Cohesion: 0.32
Nodes (7): count_cell_within_fireball(), generate_injection_particle(), is_inside_fireball(), main(), MainApplication, MainChunk, MainInterface

### Community 22 - "Community 22"
Cohesion: 0.24
Nodes (6): copy_chunk_to_src(), copy_sol_to_chunk(), scatter_forward(), scatter_reverse(), solve(), update_mapping()

### Community 23 - "Community 23"
Cohesion: 0.21
Nodes (9): analytic_solution(), analytic_source(), cleanup_config_and_tmpdir(), cleanup_tmpdir(), replace_all(), TestApplication, TestChunk, TestInterface (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.21
Nodes (12): CMake Compiler Profiles (cmake/linux-gcc.cmake, cmake/linux-intel.cmake), Deterministic Seeding (seed_type=fixed), Golden Data System (msgpack + json), IntegrationCase Dataclass Registry, PIC Integration Workflow (Python), Python Script Dependencies (numpy, matplotlib, msgpack, h5py, toml, etc.), Open boundaries with particle injection require rebalance, Reduced grid parameters for integration tests (performance vs physics trade-off) (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.18
Nodes (4): MockApplicationInterface, MockChunk, MockChunkMap, TestBalancer

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (1): PicPoissonBasic()

### Community 27 - "Community 27"
Cohesion: 0.35
Nodes (10): apply_operator_1d_primitive(), apply_operator_2d_primitive(), apply_operator_3d_primitive(), flat_index(), preconditioner_backward_1d_primitive(), preconditioner_backward_2d_primitive(), preconditioner_backward_3d_primitive(), preconditioner_forward_1d_primitive() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.2
Nodes (3): FileSystemEventHandler, OutputHandler, setup()

### Community 29 - "Community 29"
Cohesion: 0.29
Nodes (7): get_chunk_id_range(), is_completed(), require_diagnostic(), test_all(), wait(), wait_all(), write_packed_chunks()

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (3): MpiStream, Singleton, teebuf

### Community 31 - "Community 31"
Cohesion: 0.38
Nodes (3): get_nprocess(), get_thisrank(), parallel_decomposition()

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (1): MockChunk

### Community 33 - "Community 33"
Cohesion: 0.57
Nodes (5): apply_preconditioner_reference(), is_interior(), primitive_index(), require_operator_result(), require_preconditioner_result()

### Community 34 - "Community 34"
Cohesion: 0.29
Nodes (3): FieldDiag, FieldPacker, MomentPacker

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (3): Application, ApplicationInterface, Diag

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (3): ChunkVectorTest, MockChunk, MockChunkMap

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (2): MpiioDiagIoHandler, PosixDiagIoHandler

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (5): PicApplication, PicApplicationInterface, PicChunk, PicDiag, PicPacker

### Community 39 - "Community 39"
Cohesion: 0.4
Nodes (2): MockPacker, MockResourceDiagBase

### Community 40 - "Community 40"
Cohesion: 0.4
Nodes (2): TracerDiag, TracerPacker

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (2): TracerPickupDiag, TracerPickupPacker

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (2): ParticleDiag, ParticlePacker

### Community 43 - "Community 43"
Cohesion: 0.5
Nodes (2): Solver, SolverInterface

### Community 45 - "Community 45"
Cohesion: 0.5
Nodes (2): LoadPacker, RankPacker

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
Nodes (1): PicDiag

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (1): TestPicPoisson

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (1): HistoryDiag

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): ChunkMapTest

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): TestLogger

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): PicPacker

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (2): Native Async I/O (Deferred), Async I/O deferred due to MPI buffer lifetime risks and future SENSEI/Ascent alternatives

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (2): Future Hybrid Module, Module Extension Model (nix owns mechanics, modules own interpretation)

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (2): Diagnostic Module Refactor, Diagnostic refactor designed as mechanical move with minimal behavior change

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): PIC-NIX Project

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): Catch2 v3 Test Framework

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): nix External Repository (github.com/amanotk/nix)

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): PIC module auto-enables elliptic and nix modules

## Knowledge Gaps
- **66 isolated node(s):** `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file`, `Add dataset to HDF5 file`, `Add chunkmap to HDF5 file` (+61 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 19`** (15 nodes): `BaseMaxwell`, `compute_efield_from_potential_1d()`, `compute_efield_from_potential_2d()`, `compute_efield_from_potential_3d()`, `get_diverror_1d()`, `get_diverror_2d()`, `get_diverror_3d()`, `init_friedman()`, `push_bfd_1d()`, `push_bfd_2d()`, `push_bfd_3d()`, `push_efd_1d()`, `push_efd_2d()`, `push_efd_3d()`, `maxwell.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (11 nodes): `pic_poisson_basic.cpp`, `pic_poisson_basic.hpp`, `copy_chunk_to_src()`, `copy_sol_to_chunk()`, `Impl()`, `PicPoissonBasic()`, `scatter_forward()`, `scatter_reverse()`, `set_option()`, `solve()`, `update_mapping()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (7 nodes): `test_xtensor_particle.cpp`, `check_sort1d()`, `check_sort2d()`, `check_sort3d()`, `MockChunk`, `.MockChunk()`, `set_random_particle()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (6 nodes): `DiagIoHandler()`, `MpiioDiagIoHandler`, `.MpiioDiagIoHandler()`, `PosixDiagIoHandler`, `.PosixDiagIoHandler()`, `io_handler.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (5 nodes): `test_diag_resource.cpp`, `MockPacker`, `.MockPacker()`, `MockResourceDiagBase`, `.MockResourceDiagBase()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (5 nodes): `tracer.hpp`, `TracerDiag`, `.TracerDiag()`, `TracerPacker`, `.TracerPacker()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (5 nodes): `tracer_pickup.hpp`, `TracerPickupDiag`, `.TracerPickupDiag()`, `TracerPickupPacker`, `.TracerPickupPacker()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (5 nodes): `ParticleDiag`, `.ParticleDiag()`, `ParticlePacker`, `.ParticlePacker()`, `particle.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (4 nodes): `elliptic.hpp`, `Solver`, `SolverInterface`, `.SolverInterface()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (4 nodes): `LoadPacker`, `RankPacker`, `.RankPacker()`, `load.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (3 nodes): `DebugFormatter`, `DebugPrinter`, `debug.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (3 nodes): `test_application.cpp`, `TestApplication`, `.TestApplication()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (3 nodes): `test_chunk.cpp`, `TestChunk`, `.TestChunk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (3 nodes): `PicDiag`, `.PicDiag()`, `pic_diag.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (3 nodes): `test_pic_poisson_petsc.cpp`, `make_default_option()`, `TestPicPoisson`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (3 nodes): `HistoryDiag`, `.HistoryDiag()`, `history.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `test_chunkmap.cpp`, `ChunkMapTest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (2 nodes): `test_logger.cpp`, `TestLogger`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (2 nodes): `PicPacker`, `pic_packer.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (2 nodes): `Native Async I/O (Deferred)`, `Async I/O deferred due to MPI buffer lifetime risks and future SENSEI/Ascent alternatives`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (2 nodes): `Future Hybrid Module`, `Module Extension Model (nix owns mechanics, modules own interpretation)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (2 nodes): `Diagnostic Module Refactor`, `Diagnostic refactor designed as mechanical move with minimal behavior change`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `PIC-NIX Project`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `Catch2 v3 Test Framework`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `nix External Repository (github.com/amanotk/nix)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `PIC module auto-enables elliptic and nix modules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `size()` connect `Community 1` to `Community 2`, `Community 4`, `Community 5`, `Community 8`, `Community 9`, `Community 10`, `Community 16`, `Community 18`, `Community 23`, `Community 29`?**
  _High betweenness centrality (0.176) - this node is a cross-community bridge._
- **Why does `load()` connect `Community 1` to `Community 0`, `Community 5`, `Community 11`, `Community 15`, `Community 28`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `Run` connect `Community 0` to `Community 9`, `Community 3`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `size()` (e.g. with `setup_vector_local()` and `setup_indexset_global()`) actually correct?**
  _`size()` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `data()` (e.g. with `setup_vector_local()` and `setup_indexset_global()`) actually correct?**
  _`data()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Run` (e.g. with `main()` and `doit_job()`) actually correct?**
  _`Run` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Solver`, `Parallel execution of json2hdf5 for given list of files`, `Generate HDF5 file from given JSON file` to the rest of the system?**
  _66 weakly-connected nodes found - possible documentation gaps or missing edges._