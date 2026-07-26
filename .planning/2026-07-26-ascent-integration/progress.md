# Progress Log

## Session: 2026-07-26

### Current Status
- **Phase:** 7 - Pointer refresh and visualization integration
- **Started:** 2026-07-26

### Actions Taken
- Restored context from existing root planning files.
- Created isolated plan `2026-07-26-ascent-integration` and made it active.
- Ran Graphify-first architecture queries for diagnostics and field packing.
- Recorded the integration requirements and known local Ascent environment.
- Completed parallel architecture reviews of diagnostic lifecycle, PIC data
  layouts/staggering, build integration, Python packaging, tests, CI, and docs.
- Identified the required generic pre-MPI diagnostic shutdown seam.
- Verified the intended raw zero-copy versus centered-copy ownership model.
- Directly verified main-loop, MPI initialization/finalization, diagnostic
  communicator ownership, and scheduling code.
- Directly verified PIC initialization/finalization ordering and all existing
  compacting/decimation/Yee-centering formulas.
- Directly verified standalone PIC CMake behavior and Python package eager
  import behavior.
- Directly verified `PicChunk::DataContainer` exposure and moment component
  index definitions.
- Wrote the detailed eight-phase implementation plan with data contracts,
  file sets, tests, gates, risks, error policy, and final-report checklist.
- Recorded user decisions for duplicate entries, actions-path resolution,
  publication defaults, and Python import scope.
- Configured and built the Ascent-disabled MPI-wrapper baseline in
   `/tmp/opencode/picnix-build-off-mpi`.
- Added Blueprint field-location metadata and compact E/B/J transformations,
  including tests for nonzero active bounds and reduced dimensions.
- Added non-owning `DomainView` and `BlueprintBuilder` publication layers with
  raw external arrays, centered fields, particle metadata, and Blueprint
  verification coverage.
- Added `AscentRuntime` with lazy MPI communicator duplication, actions-file
  loading, synchronous publish/execute, and idempotent shutdown.
- Retained the configuration-file directory in `CfgParser` and passed it into
  diagnostic metadata so relative Ascent actions paths resolve predictably.
- Added Ascent runtime MPI smoke coverage and rejected duplicate Ascent
  configuration entries.
- Added a fresh Ascent-enabled build in
  `/tmp/opencode/picnix-build-on-tests-v2` using CMake 4.4 and Ascent 0.9.5.
- Added a fresh Ascent-disabled MPI-wrapper build in
  `/tmp/opencode/picnix-build-off-v3`.
- Added `picnix.insitu.Dataset`, domain/field/particle views, raw component
  reconstruction, active interior slicing, particle ID bit reinterpretation,
  and optional MPI reduction helpers.
- Added a Python unit test using a lightweight dictionary representation, so
  the schema adapter is testable without launching Ascent.
- Updated the plan to mark the core build, transformation, domain-view,
  Blueprint, runtime, and basic Python adapter work complete.
- Set the next implementation gate to a real two-rank Ascent Python extract,
  followed by pointer-refresh and deterministic visualization tests.
- Added a configured two-rank Ascent Python extract fixture using
  `Dataset.from_conduit`, explicit Conduit/Ascent Python paths, global domain
  count validation, centered-phi shape reconstruction, and an MPI reduction.
- Made `picnix` top-level analysis imports lazy because the Ascent Python
  environment does not install optional `msgpack` dependencies.
- Added Blueprint pointer-refresh coverage after field and particle storage
  reallocation, plus domain-ID preservation after local-domain reordering.
- Added a deterministic centered-phi pseudocolor render smoke test that checks
  for a nonempty image.
- Added explicit Blueprint physical-bound assertions for render origin,
  spacing, vertex dimensions, and computed upper bounds.
- Added schema-version rejection, explicit-communicator Python reduction,
  missing-actions cleanup coverage, empty/null particle coverage, and the
  `docs/picnix/ascent.md` usage guide.
- Added rank-aware Ascent configuration/path validation and consistent abort
  handling for publish/execute exceptions.
- Implemented the optional Ascent CMake option, MPI/C language discovery,
  `ascent::ascent_mpi` validation/linking, conditional registration, skeletal
  scheduling diagnostic, and generic pre-MPI shutdown hook.
- Added a lifecycle regression test with a fake diagnostic and fixed its test
  fixture to provide a valid `basedir`.
- Configured and built the Ascent-disabled MPI-wrapper baseline in
  `/tmp/opencode/picnix-build-off-mpi`.

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Graphify-first architecture query | Scope implementation areas | Noisy output; followed by scoped source verification | COMPLETE |
| Local Ascent environment | MPI/Python/visualization available | Ascent 0.9.5, Conduit 0.9.5 verified | PASS |
| Source verification | Confirm agent findings | Lifecycle, transforms, CMake, chunk metadata confirmed | PASS |
| Ascent-disabled configure without MPI wrapper | Configure/build | Configure succeeds; build fails because `mpi.h` is unavailable | EXPECTED LIMITATION |
| Ascent-disabled MPI-wrapper build | Build all tests | Successful | PASS |
| Ascent-disabled CTest | 44 tests | 44/44 passed | PASS |
| Ascent-enabled Blueprint/runtime focused CTest | Builder, runtime, and 8-rank lifecycle | 5/5 passed | PASS |
| Ascent-disabled focused CTest | Application, domain view, and 8-rank lifecycle | 5/5 passed | PASS |
| Python in-situ adapter | `ruff check` and focused pytest | 1/1 passed; no lint issues | PASS |
| Plan refresh | Completed phases and next gate recorded | Phase 6 real extract is next | COMPLETE |
| Real Ascent Python extract | Two ranks, local/global domains, phi reduction | Passed | PASS |
| Python regression suite | In-situ plus existing Ohm tests | 64/64 passed | PASS |
| Pointer refresh/domain reorder | Field and particle reallocations plus ID preservation | Passed | PASS |
| Visualization smoke | Centered phi pseudocolor image | Nonempty image produced | PASS |
| Visualization bounds | Blueprint origin, spacing, and upper bounds | Passed | PASS |
| Runtime/config edge cases | Missing actions, invalid schema, empty/null particles | Passed | PASS |
| Documentation | Build, configuration, schema, Python, and test guide | Added | PASS |
| Ascent-enabled focused CTest | Blueprint, runtime, extract, render, and 8-rank app | 10/10 passed | PASS |
| Ascent-disabled focused CTest | Application, domain view, and 8-rank app | 5/5 passed | PASS |
| Ascent-enabled configure/build | External Ascent 0.9.5 | Successful with tests enabled | PASS |
| Lifecycle regression | Shutdown/destruction before MPI finalize | Passed | PASS |
| Ascent-disabled configure without MPI wrapper | Configure/build | Configure succeeds; build fails because `mpi.h` is unavailable | EXPECTED LIMITATION |
| Ascent-disabled MPI-wrapper build | Build all tests | Successful | PASS |
| Ascent-disabled CTest | 44 tests | 44/44 passed | PASS |
| Full enabled CTest | All enabled C++/MPI tests | 53/53 passed | PASS |
| Full disabled CTest | All Ascent-off C++/MPI tests | 45/45 passed | PASS |
| Full Python suite | `python/tests` | 96/96 passed | PASS |
| Render baseline measurement | One-rank, four-cell Ascent render smoke | 0.40 s, 146304 KiB max RSS | RECORDED |

### Errors
| Error | Resolution |
|-------|------------|
| Graphify query results included generated documentation JavaScript | Continued with exact source reads after satisfying Graphify-first requirement |
| Default `/usr/bin/c++` build cannot find `mpi.h` | Reconfigured with project MPI wrapper `mpicxx`; full baseline passed |
| CMake 4.4 rejected legacy dependency policy | Added `CMAKE_POLICY_VERSION_MINIMUM=3.5` to isolated test configurations |
| Conduit 0.9.5 rejects `std::string_view` assignments | Converted Blueprint schema strings to `std::string` |
| Particle test used wrong zero-padding path | Corrected expected species path to `species_000` per schema |
| Python interior helper assumed an object API | Accepted both Domain metadata objects and mapping representations |
| Ascent embedded Python lacked Conduit on `PYTHONPATH` | Added the installed Conduit module directory to the CTest environment |
| Ascent Python environment lacked optional `msgpack` | Made `picnix` analysis-module imports lazy for embedded extracts |
| Blueprint scalar values arrived flat in Python | Reshaped centered fields using published local cell-shape metadata |
| Conduit list nodes returned child iterators | Normalized child nodes to Python lists in the schema adapter |
| Runtime configuration errors needed rank context | Validate paths/types before open and abort consistently after runtime exceptions |
| Lifecycle test exposed empty diagnostic basedir fixture | Added `basedir = "."` to the test configuration |
| Default `/usr/bin/c++` build cannot find `mpi.h` | Reconfigured with project MPI wrapper `mpicxx`; full baseline passed |
