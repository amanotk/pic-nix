# Task Plan: Optional Ascent In-Situ Diagnostics

## Goal

Add an optional MPI-enabled Ascent diagnostic to PIC-NIX that publishes one
Conduit Blueprint domain per local PIC chunk, supports Python extracts and
visualization, preserves raw native PIC layouts through versioned metadata,
and keeps the current build and runtime unchanged when Ascent is disabled.  

## Current Status

- **Planning:** complete  
- **Implementation:** Phases 1-5 in progress
- **Current phase:** Phase 5 - Python API and extract/visualization validation
- **Branch:** `experimental/ascent-integration`  
- **Reference design:** `notes/ascent-integration.md`  
- **Local Ascent:** 0.9.5 with Conduit 0.9.5, MPI, Python, OpenMP, and VTK-m  

## Success Criteria

The integration is complete when all of the following are true:  

- A normal `PICNIX_ENABLE_ASCENT=OFF` build never searches for or links
  Ascent or Conduit and passes all existing tests.  
- `PICNIX_ENABLE_ASCENT=ON` finds an external MPI/Python-enabled Ascent
  installation and links `ascent::ascent_mpi`.  
- One local Blueprint domain is published per currently owned `PicChunk`, with
  globally unique IDs and physical placement derived from chunk metadata.  
- Centered E, B, rho, J, and phi fields contain only owned physical cells and
  pass Blueprint verification in 1-D, 2-D, and 3-D.  
- Raw `uf`, `uj`, `um`, `phi`, and active particle storage have a stable,
  versioned schema with explicit shapes, byte strides, bounds, ordering, and
  physical locations.  
- Raw field/particle data are zero-copy where safe; centered fields and typed
  particle IDs use explicitly documented owned copies.  
- Python extracts can iterate local chunks, reconstruct raw arrays, access
  centered fields and active particles, reinterpret IDs, and perform an MPI
  reduction.  
- Published pointers and domain ownership are refreshed after rebalance and
  particle reallocation.  
- Ascent closes and its communicator is freed before `MPI_Finalize()`.  
- A two-rank extract test and a deterministic visualization smoke test pass.  

## Scope Boundaries

### Included

- Optional CMake integration and diagnostics registration.  
- Generic diagnostic shutdown before MPI finalization.  
- Shared compact Yee/current-to-cell-center transformations.  
- Non-owning chunk/domain views and field-location metadata.  
- Versioned raw PIC-NIX schema and standard centered Blueprint mesh.  
- Ascent runtime lifecycle and actions-file execution.  
- Thin `picnix.insitu` Python API.  
- Unit, MPI, pointer-refresh, extract, and rendering tests.  
- Build, schema, configuration, runtime, and ownership documentation.  

### Excluded Unless Required by a Failing Test

- Adding Ascent to FetchContent or `nix/cmake/Dependencies.cmake`.  
- Supporting Fugaku/AArch64 Python during the initial local implementation.  
- Changing the simulation update order or numerical field definitions.  
- Publishing internal Friedman filter storage as a physical field.  
- Fixing unrelated chunk serialization-size discrepancies.  
- General redesign of diagnostic dispatch, unknown-name handling, or the
  complete Python package import surface.  
- Pixel-perfect rendering comparisons.  

## Architecture

```text
PicApplication / local PicChunk objects
                 |
                 v
        non-owning DomainView
          /              \
         v                v
raw allocated arrays   compact centered transforms
and explicit metadata  (owned per invocation)
          \              /
           v            v
          Conduit Blueprint builder
                    |
                    v
       Ascent runtime (one per MPI rank)
                    |
                    v
          actions / Python extracts
```

### Responsibility Split

| Component | Responsibility | Ascent dependency |
|-----------|----------------|-------------------|
| shared xtensor transform | Compact 1-D/2-D/3-D E/B/J centering and optional decimation | No |
| `field_layout.hpp` | Stable component names, staggering, shape/order metadata | No |
| `domain_view.hpp` | Non-owning snapshot of one current `PicChunk` | No |
| `blueprint_builder.*` | Conduit multi-domain tree and owned centered buffers | Conduit |
| `ascent_runtime.*` | MPI communicator, open/publish/execute/close | Ascent |
| `diag/ascent.*` | Schedule/config validation, actions, orchestration | Ascent |
| `picnix.insitu` | Python-friendly schema access and MPI helpers | NumPy; optional mpi4py |

### Proposed File Layout

```text
pic/
  diag/
    ascent.hpp
    ascent.cpp
  insitu/
    domain_view.hpp
    field_layout.hpp
    blueprint_builder.hpp
    blueprint_builder.cpp
    ascent_runtime.hpp
    ascent_runtime.cpp

python/src/picnix/insitu/
  __init__.py
  dataset.py
  field.py
  particle.py
  mpi.py
```

Use fewer files if direct implementation shows that two responsibilities are
small enough to remain cohesive. Do not add interfaces solely for future use;
the runtime fake is the only currently justified test seam.  

## Data Contracts

### Domain Mapping

- One `PicChunk` equals one Blueprint domain.  
- Domain name: `domain_<global_chunk_id>`.  
- `state/domain_id`: `PicChunk::get_id()`.  
- `state/cycle` and `state/time`: current application step and time.  
- Domain order and owning MPI rank are not part of identity.  
- Physical origin comes from the chunk lower physical limits/global offset,
  never rank or local vector position.  

### Centered Blueprint Mesh

Proposed standard tree per domain:  

```text
state/{domain_id,cycle,time}
coordsets/cell_vertices
topologies/cell_mesh
fields/E/{association,topology,values/{x,y,z}}
fields/B/{association,topology,values/{x,y,z}}
fields/rho/{association,topology,values}
fields/J/{association,topology,values/{x,y,z}}
fields/phi/{association,topology,values}
```

- All standard fields use `association: element`.  
- Standard fields exclude ghosts and use compact owned buffers.  
- Uniform coordset vertex counts are cell counts plus one for each active
  dimension.  
- Proposed dimensional mapping, subject to empirical verification:  
  - 1-D: `dims/i = nx + 1`; omit j/k.  
  - 2-D: `dims/i = nx + 1`, `dims/j = ny + 1`; omit k.  
  - 3-D: add `dims/k = nz + 1`.  
- `E` and `B` reuse the existing numerical Yee-centering definitions.  
- `J` receives equivalent component-aware face-to-center interpolation.  
- `rho` and `phi` copy owned cell-centered interiors directly.  
- Moments are not ordinary named fields until selected and their normalization
  is documented. Raw moments remain available independently.  

### Raw Schema Version 1

Proposed custom subtree per domain:  

```text
picnix/
  schema_version: 1
  mesh/
    dimension
    global_cell_shape: [z,y,x]
    local_cell_shape: [z,y,x]
    allocated_shape: [z,y,x]
    global_offset: [z,y,x]
    active_lower: [z,y,x]
    active_upper: [z,y,x]
    ghost_width
    spacing: [z,y,x]
    physical_origin: [z,y,x]
    layout: "C/zyx-components-last"
  raw/
    uf/{values,shape,strides_bytes,components,component_locations}
    uj/{values,shape,strides_bytes,components,component_locations}
    um/{values,shape,strides_bytes,components,location}
    phi/{values,shape,strides_bytes,location}
  particles/
    species_000/
      values
      shape
      strides_bytes
      np_active
      np_allocated
      particle_width
      charge
      mass
      components
      id_encoding
```

- Raw `values` bind the complete contiguous allocated storage externally.  
- Shape and strides are explicit; Python applies active bounds separately.  
- `uf` components: `Ex,Ey,Ez,Bx,By,Bz`.  
- `uj` components: `rho,Jx,Jy,Jz`.  
- `um` components initially use source-neutral names
  `t,x,y,z,tt,xx,yy,zz,tx,ty,tz,xy,yz,zx`.  
- Locations use normalized cell coordinates and semantic labels:  
  - `Ex/Jx`: `(0.0,0.5,0.5)`, x-face.  
  - `Ey/Jy`: `(0.5,0.0,0.5)`, y-face.  
  - `Ez/Jz`: `(0.5,0.5,0.0)`, z-face.  
  - `Bx`: `(0.5,0.0,0.0)`, x-edge.  
  - `By`: `(0.0,0.5,0.0)`, y-edge.  
  - `Bz`: `(0.0,0.0,0.5)`, z-edge.  
  - `rho`, `phi`, `um`: `(0.5,0.5,0.5)`, cell center.  
- Particle components: `x,y,z,ux,uy,uz,id_bits`.  
- `id_bits` explicitly means signed 64-bit integer bits stored in the seventh
  floating-point slot. Python returns an `int64` bit view, not a numeric cast.  
- All bindings are rebuilt on every scheduled diagnostic call.  

### Ownership and Copy Policy

| Data | Initial policy | Reason |
|------|----------------|--------|
| complete raw `uf/uj/um/phi` | zero-copy external view | allocated arrays are contiguous |
| raw particle rows | zero-copy external view | active rows are a contiguous prefix |
| centered E/B/J | owned compact copy | interpolation and ghost removal required |
| centered rho/phi | owned compact copy initially | simplest Blueprint-safe interior representation |
| standard particle point topology | optional owned copy | positions are interleaved and IDs need typed conversion |
| typed particle IDs | owned `int64` copy/view helper | preserve bit identity safely |

Owned buffers and all external source arrays must remain alive until
`Ascent::execute()` returns. No Conduit tree or data pointer survives into the
next simulation mutation.  

## Proposed Diagnostic Configuration

Initial TOML shape:  

```toml
[[diagnostic]]
name = "ascent"
interval = 100
begin = 0
end = 10000
actions = "ascent_actions.yaml"

[diagnostic.publish]
raw = true
centered = true
particles = false
raw_fields = ["uf", "uj", "um", "phi"]
centered_fields = ["E", "B", "rho", "J", "phi"]
```

Validation requirements:  

- `actions` is required, nonempty, and readable on every rank.  
- `interval > 0`, `begin >= 0`, and `end >= begin`.  
- `publish` is an object and at least one publication mode is enabled.  
- Field names are validated before opening Ascent.  
- Moments are calculated only when `um` or a moment-derived field is selected.  
- Duplicate `ascent` entries are rejected unless multi-instance support is
  explicitly chosen before implementation.  
- Configuration errors are reduced across ranks before entering Ascent.  

## Implementation Phases

### Phase 0: Baseline and Design Lock

**Objective:** establish reproducible disabled/enabled baselines and settle
the few public contracts that affect later code.  

Tasks:  

- [x] Confirm current branch and preserve unrelated worktree changes.  
- [x] Record compiler, MPI, CMake, Ascent, Conduit, and Python versions.  
- [x] Configure/build/test the current default with Ascent absent from package
  search paths.  
- [x] Configure a minimal consumer against local
  `thirdparty/ascent-checkout/lib/cmake/ascent`.  
- [x] Decide duplicate diagnostic behavior, actions-path base, and publication
  defaults.  
- [ ] Verify 1-D and 2-D uniform Blueprint conventions with small standalone
  Conduit nodes before committing the schema.  
- [ ] Confirm Conduit external views observe mutations for flat allocated mesh
  arrays and interleaved particle arrays.  
- [ ] Write a compact schema example and validate naming with the Python API
  design before C++ implementation.  

Tests/gate:  

- Default build/test passes unchanged.  
- Standalone Blueprint verification passes for proposed 1-D/2-D/3-D shapes.  
- Public configuration and schema questions are resolved.  

Likely modified files: planning/findings only.  

**Status:** complete; MPI-wrapper baseline passes 44/44 tests. The direct
`/usr/bin/c++` build remains invalid because MPI headers are unavailable
without an MPI compiler wrapper.  

### Phase 1: Optional Build and Pre-MPI Shutdown Seam

**Objective:** add strictly optional discovery/registration infrastructure and
a generic lifecycle hook, without publishing data yet.  

Tasks:  

- [x] Add `PICNIX_ENABLE_ASCENT` default `OFF` to root and standalone PIC
  configurations.  
- [x] When enabled, require CMake 3.23, enable C, find MPI C/C++, and then call
  `find_package(Ascent CONFIG REQUIRED)`.  
- [x] Validate `ASCENT_MPI_ENABLED` and target `ascent::ascent_mpi`.  
- [x] Append Ascent source files and link target only in the enabled branch.  
- [x] Add a target-scoped Boolean compile definition.  
- [x] Add an idempotent virtual instance hook such as `Diag::shutdown()`.  
- [x] Add `Application::finalize_diagnostic()` that calls shutdown and clears
  `diagvec` while MPI and `Diag::Info` communicators are live.  
- [x] Invoke it immediately before existing `finalize_mpi()`.  
- [x] Add a skeletal Ascent diagnostic only when enabled; do not open Ascent
  in its constructor.  
- [x] Conditionally register the diagnostic in
  `PicApplication::initialize_diagnostic()`.  

Tests/gate:  

- Option-off configure/build/test does not inspect Ascent.  
- Option-on configure links `ascent::ascent_mpi`.  
- Fake diagnostic records shutdown and destruction before `MPI_Finalized()`.  
- Shutdown is safe when called twice and when runtime was never opened.  

Likely files:  

- `CMakeLists.txt`  
- `pic/CMakeLists.txt`  
- `nix/diag.hpp`  
- `nix/application.hpp`  
- `nix/application.cpp`  
- `pic/pic_application.cpp`  
- initial `pic/diag/ascent.*`  
- lifecycle tests and test CMake  

**Status:** complete; disabled MPI-wrapper build and focused lifecycle test
pass, and the Ascent-enabled build with tests compiles successfully.  

### Phase 2: Shared Field Layout and Compact Transformations

**Objective:** establish one tested implementation of spatial metadata and
compact cell centering used by both file output and Ascent.  

Tasks:  

- [ ] Add constexpr/static component names and normalized physical locations
  for `uf`, `uj`, `phi`, and `um`.  
- [ ] Refactor the existing six-component 1-D/2-D/3-D formulas into a compact
  owned `[nz,ny,nx,6]` result containing only active cells.  
- [ ] Preserve `Ub+1` high-side sampling exactly.  
- [ ] Add compact J centering for `[rho,Jx,Jy,Jz]`, leaving rho unchanged.  
- [ ] Keep the shared transformation in `nix` or another lower-level location
  so `nix` never depends on `pic`.  
- [ ] Adapt `XtensorPacker3D::pack_field()` to consume shared compact output
  without changing serialized values.  
- [ ] Keep decimation separate and typed/aligned; repair invalid or
  non-divisible decimation only if required by the refactor, with explicit
  tests and no silent behavior change.  

Tests/gate:  

- Existing packer tests remain passing.  
- New 1-D/2-D/3-D E/B and J tests use manufactured nonconstant values.  
- Tests cover nonzero lower bounds, ghost widths 2 and 3, inactive singleton
  dimensions, compact shape, and absence of ghost sentinels.  
- Old file diagnostic output is byte/numerically equivalent for supported
  decimation values.  
- Component-location metadata tests match solver/deposition formulas.  

Likely files:  

- `nix/xtensor/xtensor_packer3d.hpp`  
- possibly one focused shared transform header under `nix/xtensor/`  
- `pic/insitu/field_layout.hpp` if PIC-specific metadata remains separate  
- `nix/unittest/test_xtensor_packer3d.cpp`  

**Status:** in_progress  

### Phase 3: Non-Owning Domain Views and Raw Schema

**Objective:** extract all metadata and current storage bindings from local
chunks without depending on Conduit.  

Tasks:  

- [ ] Define a lightweight `DomainView` built from `PicChunk`, current cycle,
  time, and species count.  
- [ ] Capture ID, dimension, global/local/allocated shapes, offsets, active
  bounds, ghost width, spacing, origin, and layout.  
- [ ] Capture non-owning array descriptors for `uf`, `uj`, `um`, and `phi`:
  pointer, scalar type, shape, byte strides, components, and locations.  
- [ ] Capture per-species particle descriptors: active/allocated rows, pointer,
  q, m, width, components, and ID encoding.  
- [ ] Handle zero local chunks, zero active particles, and empty species
  without null dereference or invalid pointer arithmetic.  
- [ ] Ensure no view object is cached across a diagnostic invocation.  
- [ ] Document raw schema version 1 in tests as a stable contract.  

Tests/gate:  

- Artificial chunk tests verify every shape, bound, offset, coordinate, and
  species count in 1-D/2-D/3-D.  
- Descriptor pointers alias source storage before mutation.  
- Allocated versus active particle counts are distinct.  
- Particle IDs above `2^53` and negative tracer IDs preserve exact bits.  
- Recreating a view after resize returns the new pointer.  

Likely files:  

- `pic/insitu/domain_view.hpp`  
- `pic/insitu/field_layout.hpp`  
- `pic/unittest/test_pic_domain_view.cpp`  
- `pic/unittest/CMakeLists.txt`  

**Status:** pending  

### Phase 4: Conduit Blueprint Builder

**Objective:** construct valid local multi-domain Conduit data with standard
centered fields and the versioned raw subtree.  

Tasks:  

- [ ] Implement one root child per local `DomainView`.  
- [ ] Build state, uniform coordset, topology, and element-associated fields.  
- [ ] Own compact centered buffers inside a publication object whose lifetime
  extends through Ascent execution.  
- [ ] Bind raw allocated arrays externally as flat values and emit complete
  shape/stride/bounds metadata.  
- [ ] Add publication switches and field-selection filtering.  
- [ ] Add optional per-species Blueprint point topology only after raw particle
  publication works.  
- [ ] Bit-copy typed particle IDs for the standard particle topology.  
- [ ] Run Blueprint verification before publication in tests and optionally in
  debug builds/runtime configuration.  
- [ ] Ensure no standard field includes ghost cells or duplicate halo
  contributions.  

Tests/gate:  

- `conduit::blueprint::mesh::verify` passes for 1-D/2-D/3-D.  
- One and multiple domains have correct IDs, origins, element counts, vertex
  counts, and field associations.  
- Manufactured E/B/J/rho/phi values match expected centered values.  
- Raw pointer alias tests observe source mutation after binding.  
- Disabled raw/centered/particle sections are absent as configured.  
- Empty-rank, empty-species, and zero-active-particle cases pass.  
- Reordered local domains produce equivalent IDs and placement.  

Likely files:  

- `pic/insitu/blueprint_builder.hpp`  
- `pic/insitu/blueprint_builder.cpp`  
- `pic/unittest/test_pic_ascent_blueprint.cpp`  
- conditional test CMake  

**Status:** pending  

### Phase 5: Ascent Runtime and Diagnostic Orchestration

**Objective:** execute configured actions synchronously and safely at scheduled
diagnostic steps.  

Tasks:  

- [ ] Define the smallest runtime seam needed for fake lifecycle tests.  
- [ ] On first required invocation, duplicate `MPI_COMM_WORLD`, pass
  `MPI_Comm_c2f()` through Ascent options, and open once.  
- [ ] Load and cache the actions tree after validating the configured path on
  every rank.  
- [ ] On every required invocation: obtain current local chunks, calculate
  selected moments, rebuild views, rebuild buffers/tree, verify as configured,
  publish, and execute.  
- [ ] Do not retain any publication object after synchronous execution.  
- [ ] Run once per rank outside OpenMP regions.  
- [ ] Catch configuration/Conduit/Ascent/Python exceptions, attach rank and
  actions context, synchronize failure state where possible, then follow the
  repository MPI-abort convention to avoid asymmetric continuation.  
- [ ] On shutdown: close Ascent once, then free the duplicated communicator.  
- [ ] Reject duplicate Ascent diagnostic entries unless multi-instance support
  was selected in Phase 0.  

Tests/gate:  

- Fake runtime verifies skip behavior, open once, publish-before-execute,
  rebuilt tree each call, and close once.  
- Pointer identity test changes source allocation between calls and observes
  the refreshed pointer.  
- Close and runtime destruction occur before MPI finalization.  
- Missing/malformed actions and invalid config produce useful rank-aware
  errors without deadlock.  
- A minimal real serial/rank-local Ascent invocation succeeds.  

Likely files:  

- `pic/insitu/ascent_runtime.hpp`  
- `pic/insitu/ascent_runtime.cpp`  
- `pic/diag/ascent.hpp`  
- `pic/diag/ascent.cpp`  
- runtime/config/lifecycle tests  

**Status:** pending  

### Phase 6: Python Helper API and Examples

**Objective:** let extract scripts consume PIC-NIX data without knowing raw
Conduit paths, shape arithmetic, or ID encoding.  

Tasks:  

- [ ] Add `Dataset.from_conduit(node)` as the primary constructor.  
- [ ] Optionally add a convenience constructor that accepts the injected
  `ascent_data` callable explicitly; do not depend on hidden globals.  
- [ ] Implement local-domain iteration and lookup by domain ID.  
- [ ] Implement mesh metadata, raw field descriptors, component lookup,
  interior slicing, centered field access, and schema-version validation.  
- [ ] Implement active-particle views, q/m access, exact ID reinterpretation,
  and optional relativistic kinetic energy.  
- [ ] Implement small MPI helpers that accept an explicit mpi4py communicator
  or convert `ascent_mpi_comm_id()` supplied by the extract script.  
- [ ] Keep ordinary unit tests independent of Ascent using mappings/fake nodes;
  add conditional real-Conduit tests.  
- [ ] Install `picnix` and its current dependencies into the Ascent Python
  environment for integration tests; defer eager-import cleanup unless it
  blocks execution.  
- [ ] Add a minimal Python extract that computes a local statistic and a global
  MPI reduction.  
- [ ] Add a minimal actions YAML that runs the extract and optionally renders a
  centered scalar field.  

Target usage:  

```python
from picnix.insitu import Dataset

dataset = Dataset.from_conduit(ascent_data())
for chunk in dataset.local_chunks():
    ex = chunk.raw_field("uf").component("Ex")
    electric = chunk.centered_field("E")
    electrons = chunk.particles(species=0)
    ids = electrons.ids
```

Tests/gate:  

- Unit tests cover domain iteration, schema mismatch, shapes/strides, component
  extraction, interior slicing, centered fields, active particles, exact IDs,
  missing values, and zero particles.  
- Tests pass without importing Ascent/Conduit.  
- Conditional tests pass against a real Conduit Node.  
- `ruff format` and `ruff check` pass.  

Likely files:  

- `python/src/picnix/insitu/*`  
- `python/tests/test_insitu_*.py`  
- example actions/extract files under a PIC example or docs example directory  

**Status:** pending  

### Phase 7: MPI, Rebalance, Python, and Visualization Integration

**Objective:** validate the complete runtime under realistic multi-domain MPI
execution and storage changes.  

Tasks:  

- [ ] Add a focused np=2 executable/CTest label rather than extending the
  ordinary PIC application test.  
- [ ] Configure multiple global domains and ensure at least one rank owns
  multiple chunks where practical.  
- [ ] Verify global domain count, unique IDs, physical coverage, and local
  decomposition seen by Python.  
- [ ] Perform an mpi4py global scalar or histogram reduction.  
- [ ] Confirm owned interiors avoid duplicated ghost contributions.  
- [ ] Trigger or simulate chunk ownership movement and verify rebuilt domain
  order and external pointers.  
- [ ] Trigger particle resize/swap and verify active rows and pointers refresh.  
- [ ] Render a deterministic field such as `x + 2*y + 3*z`.  
- [ ] Check successful completion, nonempty image, Blueprint validity, and
  global bounds; do not compare pixels.  
- [ ] Record wall time and centered-buffer memory overhead for a representative
  local multi-chunk case.  

Tests/gate:  

- np=2 extract test completes without deadlock.  
- MPI reduction matches the analytic expected result.  
- Rebalance/reallocation tests prove pointer refresh.  
- Visualization produces a nonempty image and correct global bounds.  
- Ascent closes cleanly on all ranks.  

Likely files:  

- dedicated PIC Ascent MPI test source and CMake registration  
- deterministic test config/actions/extract files  
- optional CTest fixtures/scripts  

**Status:** pending  

### Phase 8: Documentation, CI, and Full Verification

**Objective:** make the feature reproducible for users while preserving the
ordinary project workflow.  

Tasks:  

- [ ] Document Ascent-disabled and enabled configure commands, exact
  `Ascent_DIR`, CMake requirement, runtime paths, and Python environment.  
- [ ] Document diagnostic TOML, actions YAML, MPI behavior, raw schema,
  component locations, copies, lifetimes, and limitations.  
- [ ] Document installing `picnix` into Ascent's Python environment.  
- [ ] Add a dedicated Ascent page and link it from PIC-NIX docs/README.  
- [ ] Keep existing CI jobs Ascent-free.  
- [ ] Add a separate optional/manual/scheduled Ascent workflow using a pinned
  container or versioned cached prefix if practical.  
- [ ] If CI installation remains impractical, document the exact local test
  commands and mark integration coverage as external.  
- [ ] Run formatting, default tests, enabled unit tests, Python tests, MPI tests,
  smoke rendering, and Graphify update.  
- [ ] Produce final architecture/schema/configuration/test/copy report.  

Required final commands, adjusted to actual build directories:  

```sh
clang-format -i <modified-cpp-files>
ruff format python/src/picnix/insitu python/tests/test_insitu_*.py
ruff check python/src/picnix/insitu python/tests/test_insitu_*.py
cmake -S . -B build-default -DBUILD_TESTING=ON -DPICNIX_ENABLE_ASCENT=OFF
cmake --build build-default --parallel
ctest --test-dir build-default --output-on-failure
cmake -S . -B build-ascent -DBUILD_TESTING=ON \
  -DPICNIX_ENABLE_ASCENT=ON \
  -DAscent_DIR=<prefix>/ascent-checkout/lib/cmake/ascent
cmake --build build-ascent --parallel
ctest --test-dir build-ascent -L ASCENT --output-on-failure
python -m pytest python/tests
graphify update .
```

Final gate: all acceptance criteria pass, or every unrun test is explicitly
listed with a reason and a reproducible command.  

**Status:** pending  

## Dependency Order

```text
Phase 0
   |
   v
Phase 1 lifecycle/build
   |
   +-----------> Phase 2 transforms
   |                 |
   |                 v
   +-----------> Phase 3 domain views
                         |
                         v
                    Phase 4 builder
                         |
                         v
                    Phase 5 runtime
                         |
                         v
                    Phase 6 Python
                         |
                         v
                    Phase 7 integration
                         |
                         v
                    Phase 8 delivery
```

Phases 2 and 3 can proceed independently after Phase 1, but Phase 4 requires
both. Keep only one implementation phase active at a time unless work is
explicitly delegated without overlapping files.  

## Testing Matrix

| Layer | Ascent required | MPI required | Runs in normal CI |
|-------|-----------------|--------------|-------------------|
| field layout/transform | No | No | Yes |
| domain view/raw descriptors | No | No | Yes |
| Python helper with fake mapping | No | No | Yes |
| generic diagnostic shutdown | No | Yes | Yes |
| Blueprint builder/verification | Conduit | No | Optional job |
| real runtime lifecycle | Yes | Yes | Optional job |
| Python extract and reduction | Yes | np=2 | Optional job |
| visualization smoke | Yes + VTK-m | np=1 or np=2 | Optional job |
| rebalance/pointer refresh | Prefer real Ascent | np=2 | Optional job |

## Error-Handling Contract

- Validate all noncollective configuration and paths before opening Ascent.  
- Reduce validation success across ranks before any Ascent collective call.  
- Include diagnostic name, cycle, rank, and actions path in errors.  
- Catch Conduit/Ascent/Python exceptions at the diagnostic boundary.  
- Do not let one rank throw into ordinary stack unwinding while peers wait in
  Ascent, MPI, or the application diagnostic barrier.  
- Use the repository's fatal MPI-abort convention after best-effort rank-aware
  reporting when synchronized recovery is not possible.  
- Make shutdown idempotent and safe for never-opened/partially-opened runtime.  

## Performance Checks

- Record bytes externally viewed versus bytes copied per domain.  
- Ensure no global gather is introduced for ordinary visualization.  
- Ensure centered buffers are one compact owned representation, not a full
  ghost-padded temporary plus another compact copy.  
- Do not calculate moments unless selected.  
- Do not copy raw allocated field arrays.  
- Measure diagnostic wall time using existing logging conventions if a small
  hook fits naturally; do not add a profiling framework.  
- Invoke Ascent once per rank and outside OpenMP parallel regions.  

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use external Ascent only | Stack is too large for normal FetchContent workflow |
| Use `ascent::ascent_mpi` | PIC-NIX execution is distributed and local install exports it |
| Require CMake 3.23 only when enabled | Preserve current 3.20 baseline for default builds |
| Lazy-open runtime | Diagnostics are constructed before chunks/species exist |
| Duplicate `MPI_COMM_WORLD` for runtime | Explicit ownership and safe close/free ordering |
| Add generic diagnostic shutdown hook | Ascent must close before MPI finalization |
| One chunk per domain | Preserves load-balanced decomposition without global gather |
| Raw allocated arrays plus bounds | Safest zero-copy representation of ghost-padded storage |
| Owned compact standard fields | Interpolation and ghost exclusion require copies |
| Keep raw Yee fields outside standard Blueprint fields | Components are not colocated |
| Pass Conduit node explicitly to Python | Extract-injected globals are not module globals |
| Separate optional CI job | Avoid multiplying expensive normal matrix |
| Reject duplicate Ascent entries initially | One runtime and actions tree keeps lifecycle/configuration unambiguous |
| Resolve relative actions paths from config directory | Keeps configurations portable across launch directories |
| Default raw and centered on, particles off | Enables analysis/visualization without default particle-volume cost |
| Defer Python eager-import cleanup | Avoid unrelated API changes; install current dependencies in Ascent environment |

## Resolved User Decisions

1. Reject multiple `name = "ascent"` entries initially.  
2. Resolve relative actions paths against the configuration file directory.  
3. Default raw and centered publication on; default particles off.  
4. Install current `picnix` dependencies into the Ascent Python environment
   and defer lazy-import cleanup.  

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Ascent-disabled regression | Strict conditional discovery/sources/link/registration plus default CI |
| Stale pointers after rebalance/resize | Rebuild views/tree every scheduled call; controlled pointer-refresh tests |
| Wrong Yee placement | Source-verified metadata plus manufactured 1-D/2-D/3-D tests |
| Incorrect low-dimensional Blueprint shape | Standalone Phase 0 verification and render tests |
| MPI deadlock on one-rank failure | Rank-wide validation and fatal synchronized error policy |
| Destructor runs after MPI | Generic shutdown and `diagvec.clear()` before `finalize_mpi()` |
| Python ABI mismatch | Use the interpreter/environment against which Ascent was built |
| Centered copy overhead | Compact interior-only transformation, one owned publication lifetime |
| Moment cost surprises | Explicit selection and existing per-step cache |
| Schema drift | Integer `schema_version`, contract tests, documented component metadata |
| Huge CI cost | Separate container/cached optional workflow |
| Fugaku Python cross-build complexity | Postpone platform validation; keep explicit interpreter/toolchain contracts |

## Errors Encountered During Planning

| Error | Resolution |
|-------|------------|
| Initial Graphify queries were polluted by generated documentation JavaScript | Used Graphify first as required, then scoped exact source inspection |

## Final Report Checklist

- [ ] Concise architecture summary.  
- [ ] Modified/added file list.  
- [ ] Final raw schema and component locations.  
- [ ] Final diagnostic configuration.  
- [ ] Build commands with Ascent disabled and enabled.  
- [ ] Test commands/results and unrun tests.  
- [ ] Raw zero-copy and centered-copy accounting.  
- [ ] Verification basis for field positions.  
- [ ] Proof of close-before-MPI-finalize ordering.  
- [ ] Proof of pointer refresh after rebalance/reallocation.  
- [ ] Remaining limitations and follow-up work.  
