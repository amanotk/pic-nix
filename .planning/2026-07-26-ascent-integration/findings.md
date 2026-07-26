# Findings & Decisions

## Requirements
- Implement the design in `notes/ascent-integration.md` incrementally on
  `experimental/ascent-integration`.
- Keep Ascent entirely optional and externally installed; default builds must
  not discover, compile, or link Ascent or Conduit.
- Treat one PIC-NIX chunk as one local Blueprint domain and rebuild all
  external views at each diagnostic invocation.
- Provide both raw native-layout data for Python analysis and compact,
  cell-centered Blueprint fields for ordinary visualization.
- Python extracts are a core requirement, not an optional follow-up.
- Preserve MPI behavior, support future Fugaku cross-compilation, and close
  Ascent before MPI finalization.
- Reuse existing Yee-to-cell-center interpolation rather than duplicating it.
- Keep the Ascent adapter thin and avoid unnecessary abstraction.

## Research Findings
- The repository graph was updated after adding the installer, but initial
  broad Graphify queries were polluted by generated documentation JavaScript
  and did not reliably identify diagnostic or interpolation paths. Continue
  with scoped source inspection after satisfying the Graphify-first rule.
- Ascent 0.9.5, Conduit 0.9.5, MPI/Python bindings, and VTK-m rendering have
  already been built and verified locally under ignored `thirdparty/`.
- The installed Ascent package exports `ascent::ascent` and
  `ascent::ascent_mpi`; its Python environment imports `ascent`, `conduit`, and
  `mpi4py` successfully.

### Diagnostic lifecycle

- `nix::Application` owns one vector of `std::unique_ptr<Diag>` objects and
  dispatches configuration entries by diagnostic name on every step.
- Scheduling is implemented inside each diagnostic through inclusive
  `begin`/`end` and phase-aligned `interval`; `interval <= 0` is currently not
  validated.
- `PicApplication::initialize_diagnostic()` is the registration point. The
  Ascent diagnostic must be registered only when the build option is enabled.
- Diagnostics are constructed before chunks and species are initialized, so
  Ascent must open lazily on its first scheduled invocation.
- Current diagnostic objects survive until application destruction, which can
  occur after `MPI_Finalize()`. Add a generic instance shutdown hook and clear
  diagnostic objects before shared diagnostic communicators and MPI are
  finalized.
- One configured name maps to one registered diagnostic object. Multiple
  `ascent` entries would share runtime state unless explicitly rejected or
  keyed separately.
- Direct source verification confirms the main-loop order is diagnostic,
  push, rebalance, log, increment. Publication therefore observes ownership
  resulting from the previous rebalance and must be rebuilt before the next
  push mutates storage.
- `Application::finalize()` currently performs checkpoint/log work and then
  immediately calls `finalize_mpi()`, where diagnostic communicators are freed
  before optional `MPI_Finalize()`. The new instance shutdown phase belongs
  immediately before this call.

### PIC data and ownership

- Mesh storage order is `[z, y, x, component...]` with right/C ordering.
- One chunk has a globally stable chunk ID, global `[z,y,x]` cell offset,
  local dimensions, spacing, physical limits, active inclusive bounds, and a
  shape-order-dependent ghost margin.
- `uf` is `[nz,ny,nx,6]` with components `Ex,Ey,Ez,Bx,By,Bz`; `uj` is
  `[nz,ny,nx,4]` with `rho,Jx,Jy,Jz`; `um` is
  `[nz,ny,nx,Ns,14]`; `phi` is `[nz,ny,nx]`.
- Verified spatial locations:
  - `Ex/Jx`: x-face; `Ey/Jy`: y-face; `Ez/Jz`: z-face.
  - `Bx`: x-edge; `By`: y-edge; `Bz`: z-edge.
  - `rho`, `phi`, and all `um` components: cell center.
- Existing `XtensorPacker3D` has tested 1-D/2-D/3-D Yee-to-center formulas but
  allocates a full ghost-padded temporary. Refactor to compact interior output
  and preserve the old packer behavior through shared code.
- Current colocation is six-component `uf` specific. Centered current requires
  a new tested four-component `uj` transformation.
- Full allocated mesh arrays are contiguous and can be exposed as raw external
  views with shape/byte-stride metadata and separate active bounds. The
  ghost-free 3-D interior is not one simple flat contiguous range.
- Particle `xu` is `[Np_allocated,7]`; only `[0,Np_active)` is active. The final
  column stores an integer ID as raw bits in a floating slot and must be
  reinterpreted, never numerically converted.
- Rebalance, unpack, particle resize/sort/swap, and ordinary field updates can
  invalidate or change external views. Rebuild the complete Conduit node every
  scheduled invocation and retain it only through synchronous `execute()`.
- Moment calculation is lazy and step-cached but can add substantial work;
  invoke it only when selected publication requires moments.
- Direct source verification confirms PIC diagnostic registration happens
  before species initialization, communicator duplication, solver creation,
  and chunk setup. Runtime construction may occur in the diagnostic object,
  but `Ascent::open()` and all data binding must remain lazy.
- `PicApplication::finalize()` frees PIC boundary communicators and the
  elliptic backend before entering base finalization. The Ascent runtime should
  therefore own/use a duplicate of `MPI_COMM_WORLD`, not any PIC boundary
  communicator.
- Existing 1-D/2-D/3-D formulas read high-side ghost values (`Ub+1`) to center
  face/edge fields. Compact output must preserve this behavior while writing
  only owned interior cells.
- `PicChunk::DataContainer` exposes array references and local numerical
  metadata but not chunk ID, dimensions, or global offset. A domain-view
  builder must accept/retain the `PicChunk` itself rather than only the returned
  data bundle.
- The 14 moment indices are named only `t,x,y,z,tt,xx,yy,zz,tx,ty,tz,xy,yz,zx`
  in source. The raw schema should preserve these neutral names/formulas until
  units and normalization justify stronger physical labels.

### Build, Python, tests, and CI

- Add `PICNIX_ENABLE_ASCENT=OFF` at the root and standalone PIC levels. Keep
  `find_package(Ascent CONFIG REQUIRED)`, source compilation, headers, links,
  and registration entirely inside the enabled branch.
- Ascent 0.9.5 exports `ascent::ascent_mpi`, requires CMake 3.23, and references
  both `MPI::MPI_C` and `MPI::MPI_CXX`; enabled builds must enable C and find
  both MPI components.
- Ascent must remain outside `nix/cmake/Dependencies.cmake` and FetchContent.
- Keep Ascent-independent transformation, domain-view, schema, and Python
  helper tests in ordinary CI. Gate Conduit/Blueprint/runtime tests behind the
  option and isolate the np=2 integration smoke test.
- The existing `picnix` package eagerly imports analysis modules from
  `picnix.__init__`, which may pull heavy dependencies into embedded extracts.
  Decide whether to preserve that behavior initially or make package imports
  lazy as a separately tested cleanup.
- Ascent extract globals such as `ascent_data()` exist in the extract script,
  not automatically in imported module globals. Prefer passing the Conduit
  node explicitly to `Dataset.from_conduit(...)` over hidden global lookup.
- A separate optional/scheduled Ascent CI job or pinned container is preferable
  to multiplying the existing PETSc matrix or rebuilding the stack per job.
- Direct source verification confirms standalone `pic` currently declares a
  C++-only project and conditionally appends PETSc sources. The Ascent branch
  can follow the same source-list pattern but must call `enable_language(C)`
  before importing Ascent's MPI targets.
- Direct source verification confirms `picnix.__init__` performs five eager
  star imports. Keep lazy-import cleanup out of the first C++ milestone; make
  any Python package import change its own reviewed/tested substep.

### Existing issues relevant to implementation

- A malformed diagnostic entry without `name` returns from the whole dispatch
  function; unknown names are silently ignored.
- Existing field decimation can disagree with metadata for non-divisible sizes
  and can divide by zero for invalid values. Do not silently couple this repair
  to Ascent unless the shared transformation refactor requires it; add focused
  regression tests if touched.
- Current packed-buffer writes may use unaligned `double*` casts and `int`
  offsets. New centered storage should use typed, aligned owning arrays.
- PIC chunk packed-size estimation and serialized arrays differ (`um` versus
  `ff`); this is an independent migration risk, not part of the Ascent scope.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use an isolated planning directory | Preserves the completed, unrelated root planning files |
| Inspect source after Graphify | The graph queries returned noisy, insufficiently scoped results |
| Lazy-open Ascent | Diagnostics are constructed before simulation data exists |
| Add generic diagnostic shutdown | Ensures close/destruction occurs while MPI remains valid |
| Raw views expose allocated storage | Enables zero-copy while metadata identifies ghosts and active bounds |
| Centered fields use owned compact buffers | Blueprint physical cells exclude ghosts and buffers remain valid through execute |
| Pass Conduit nodes explicitly to Python helpers | Imported modules cannot safely discover extract-injected globals |
| Keep Ascent outside FetchContent | The dependency stack is large and externally installed |
| Share centering below the PIC adapter | The existing packer is in `nix`; shared formulas must not introduce a `nix` dependency on `pic` |

## Resolved User Decisions

- Reject duplicate `ascent` diagnostic entries initially.
- Resolve relative actions paths against the configuration-file directory.
- Default raw and centered publication on; default particles off.
- Install the existing `picnix` dependency set into Ascent's Python
  environment and defer eager-import cleanup.

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Graphify query relevance was poor | Use exact symbol searches and scoped reads for implementation-level planning |
| Existing finalization order is unsafe for Ascent | Plan an application-level shutdown hook before MPI cleanup |
| Existing colocation does not handle `uj` | Add a separate tested current-centering transformation |

## Resources
- `notes/ascent-integration.md`
- `scripts/install_ascent.sh`
- `.planning/2026-07-26-ascent-integration/`
- `nix/application.*`, `nix/diag.hpp`, `pic/pic_application.*`, `pic/pic_diag.hpp`
- `pic/pic_chunk.*`, `nix/chunk.*`, `nix/xtensor/xtensor_packer3d.hpp`
- `pic/CMakeLists.txt`, `pic/unittest/CMakeLists.txt`, `python/pyproject.toml`
