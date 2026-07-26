# Task Plan: Ascent Integration Remediation

## Goal

Correct the defects found in the branch review so the optional Ascent
integration is safe under MPI, publishes current simulation data, exposes the
documented Python API against real Conduit nodes, and has reliable portable
tests.  Preserve the Ascent-disabled build and existing PIC-NIX behavior.  

## Current Status

- **Planning:** complete  
- **Implementation:** P0/P1 fixes complete; final edge coverage pending  
- **Current phase:** Phase 4 - build and test verification  
- **Branch:** `experimental/ascent-integration`  
- **Review base:** `origin/experimental/ascent-integration`  
- **Review scope:** commits `1c8e370` through `db8e761`  
- **Unrelated worktree file:** `.python-version`; do not modify or commit  

## Priority Summary

| Priority | Work |
|----------|------|
| P0 | Fix MPI rank-asymmetric validation deadlock |
| P0 | Refresh `um` before raw moment publication |
| P0 | Fix raw, vector, and particle Python views for real Conduit nodes |
| P1 | Fix species naming, `from_ascent`, and wildcard-import regressions |
| P1 | Repair test isolation so stale files cannot satisfy assertions |
| P1 | Add an end-to-end `PicApplication` Ascent diagnostic test |
| P1 | Fix standalone/build capability and public dependency propagation |
| P2 | Strengthen lifecycle, empty-rank, decimation, labels, and CI coverage |

## Success Criteria

- Every rank takes the same path before Ascent/MPI collectives, including when
  an actions file is missing or unreadable on only some ranks.  
- Raw `um` is calculated for the current step before publication and does not
  depend on another diagnostic running first.  
- `Dataset` reconstructs real flat Conduit leaves for raw fields, vector
  fields, and particles using published shapes and byte strides.  
- `raw_field("uf").component("Ex")`, `centered_field("E").component("x")`,
  centered vector `.array`, active particles, and particle IDs pass against a
  real Conduit node produced by C++.  
- Species names use the same fixed-width contract in C++ and Python for indices
  0, 9, 10, 99, and 100.  
- Public Python import behavior is either backward compatible or deliberately
  changed with explicit migration documentation and tests.  
- Every Ascent test starts with an empty working directory and removes its own
  outputs.  
- A real `PicApplication` configuration exercises registration, scheduling,
  relative actions paths, publication options, moment refresh, and shutdown.  
- Root and standalone PIC builds work with Ascent enabled; consumers of public
  Ascent-facing headers inherit required dependencies.  
- Capability checks match registered tests for MPI, Python extracts, and
  rendering.  
- `ctest -L ASCENT` selects all Ascent tests and cannot pass with zero tests.  
- Full Ascent-enabled, Ascent-disabled, and Python suites pass from clean test
  directories.  

## Scope Boundaries

### Included

- Corrections required by the detailed branch review.  
- Regression tests that fail against the current branch implementation.  
- Minimal CMake/test restructuring needed for standalone and capability-aware
  builds.  
- Documentation corrections caused by API or test-command fixes.  

### Excluded Unless Required by a Failing Test

- FetchContent installation of Ascent.  
- Pixel-level image comparisons.  
- General diagnostic-dispatch redesign unrelated to Ascent.  
- Full Fugaku Python validation.  
- Performance optimization beyond eliminating correctness-related duplicate
  work or copies.  

## Phase 0: Reproduction Baseline

**Objective:** turn every confirmed review finding into a reproducible failing
test or command before changing behavior.  

### Tasks

- [ ] Preserve current clean branch state and leave `.python-version` alone.  
- [ ] Record the current enabled, disabled, and Python suite results.  
- [ ] Add real-Conduit Python regression fixtures that reproduce flat raw
  leaves, object-valued vector fields, flat particle storage, and null species.  
- [ ] Add import regressions for `Dataset.from_ascent()` and
  `from picnix import *`.  
- [ ] Add species-name cases for indices 9, 10, 99, and 100.  
- [ ] Add a stale-output regression that pre-creates extract/render artifacts
  and proves preparation removes them.  
- [ ] Add or script a standalone PIC configure/build test.  
- [ ] Confirm `ctest -L ASCENT -N` currently selects zero tests.  

### Gate

- Each P0/P1 defect has a deterministic failing test or a documented build
  reproducer.  

**Status:** complete for the documented reproduction commands; some failures
were fixed before dedicated fixtures were added.  

## Phase 1: Real-Conduit Python Correctness

**Objective:** make the documented Python API consume the exact node structure
created by `BlueprintBuilder`.  

### Tasks

- [ ] Add one helper that distinguishes scalar leaves, object children, and
  list children for both mappings and Conduit nodes.  
- [ ] Change `RawField.component()` to index `self.array()` after reconstructing
  published shape and strides.  
- [ ] Reconstruct particle storage from `shape` and `strides_bytes` before
  slicing active rows or interpreting IDs.  
- [ ] Define empty/null particle behavior as a stable `(0, particle_width)`
  array with an empty `int64` ID array.  
- [ ] Handle centered vector field children by name for real Conduit object
  nodes.  
- [ ] Return centered vector `.array` in compact
  `[nz, ny, nx, component]` order.  
- [ ] Reject missing schema versions rather than silently accepting unrelated
  Blueprint data, unless compatibility requirements prove otherwise.  
- [ ] Replace or remove no-argument `Dataset.from_ascent()`; prefer an explicit
  node or injected callable because `ascent_data()` is an extract global.  
- [ ] Restore the previous top-level Python export contract, including wildcard
  imports, without reintroducing heavy eager imports into embedded extracts.  
- [ ] Format with `ruff format` and pass `ruff check`.  

### Tests

- Mapping-backed unit tests remain.  
- Real-Conduit tests cover raw `uf`, centered `E/B/J`, scalar `phi`, particles,
  empty particles, ID bits, and schema errors.  
- The two-rank extract enables raw and particle publication and validates the
  same API used in documentation.  
- Wildcard and direct top-level imports preserve public names such as `Run`,
  `Tracer`, `get_wk_spectrum`, and Ohm helpers.  

### Gate

- The three reproduced Python data-shape failures and both import regressions
  pass against the installed Conduit/Ascent Python environment.  

**Status:** implemented and verified against the installed real-Conduit
environment; broader multi-dimensional fixture coverage remains optional.  

## Phase 2: MPI and Simulation Data Correctness

**Objective:** eliminate deadlocks and ensure every published field represents
the current simulation step.  

### Tasks

- [ ] Validate actions path readability on each rank without entering Ascent.  
- [ ] Reduce validation success and error classification across
  `MPI_COMM_WORLD`; all ranks must return or abort consistently.  
- [ ] Include rank, cycle, and actions path in fatal diagnostics.  
- [ ] Move filesystem operations that can throw inside the diagnostic exception
  boundary.  
- [ ] Validate scheduling values, especially `interval > 0`, before modulo use.  
- [ ] Call `interface->calculate_moment()` exactly when raw `um` publication is
  enabled or a selected derived field needs moments.  
- [ ] Ensure moment calculation happens before creating `DomainView` objects.  
- [ ] Fix C++ species formatting to `species_%03d` semantics for all indices.  
- [ ] Decide and test behavior when a rank owns zero chunks.  

### Tests

- An MPI test simulates rank-asymmetric actions-file visibility and terminates
  without deadlock.  
- An Ascent-only diagnostic configuration proves `um` changes with particle
  data and is independent of diagnostic ordering.  
- An empty-domain rank participates in publish/execute or fails collectively
  with a documented message.  
- Missing/malformed actions and invalid intervals fail consistently.  

### Gate

- No rank-local return can strand peers in `MPI_Comm_dup`, Ascent execution, or
  the application diagnostic barrier.  
- Current-step moment values are verified numerically.  

**Status:** core fixes implemented and the application path is covered; a
rank-asymmetric visibility regression test and numerical moment assertion remain.  

## Phase 3: End-to-End Diagnostic and Lifecycle Coverage

**Objective:** test the actual integration boundary instead of only invoking
`BlueprintBuilder` and `AscentRuntime` directly.  

### Tasks

- [ ] Add a small `PicApplication` Ascent test configuration.  
- [ ] Exercise conditional registration and duplicate-entry rejection.  
- [ ] Verify `begin`, `end`, and `interval` scheduling.  
- [ ] Verify relative actions paths resolve from the configuration directory.  
- [ ] Verify `publish.raw`, `publish.centered`, and `publish.particles`.  
- [ ] Verify shutdown and object destruction occur before `Diag::finalize()`,
  not merely before an externally owned MPI runtime is finalized.  
- [ ] Replace or strengthen the current lifecycle test so reversing
  `finalize_diagnostic()` and `finalize_mpi()` would make it fail.  
- [ ] Add an observable communicator-release test or a minimal injected runtime
  seam; do not add a broad abstraction solely for testing.  

### Gate

- At least one test runs through `Application::diagnostic()` and
  `AscentDiag::operator()`.  
- Lifecycle tests fail under deliberately reversed shutdown ordering.  

**Status:** root and standalone builds pass; 10 Ascent tests are selected and
pass, with capability gating and stale-output cleanup implemented.  

## Phase 4: CMake and Test Reliability

**Objective:** make enabled builds and tests portable, capability-aware, and
incapable of passing from stale artifacts.  

### Tasks

- [ ] Clear `${tmpdir}_cwd` before each MPI test that writes artifacts.  
- [ ] Remove `${tmpdir}_cwd` during cleanup, including extract outputs.  
- [ ] Give every Ascent test the `ASCENT` label; preserve `MPI` where relevant.  
- [ ] Add a guard proving `ctest -L ASCENT` selects at least one test.  
- [ ] Use `${PICNIX_DIR}/python/src` or another standalone-safe source path.  
- [ ] Use CMake-native environment/path-list handling rather than a hard-coded
  `:` separator.  
- [ ] Decide whether Ascent-facing headers are public. If public, propagate
  `ascent::ascent_mpi` usage requirements; if private, stop exposing them as a
  consumer API.  
- [ ] Add a minimal downstream consumer compile test for public headers.  
- [ ] Require Python-enabled Ascent for Python integration, or conditionally
  register extract tests with a clear skip.  
- [ ] Require VTK-h/rendering capability for render tests, or conditionally
  register them with a clear skip.  
- [ ] Preserve an Ascent-disabled configure that never discovers Ascent or
  Conduit.  

### Tests

- Root enabled build.  
- Standalone PIC enabled build.  
- Public-header consumer build.  
- MPI-only Ascent capability configure.  
- MPI+Python without rendering configure.  
- Two consecutive extract/render runs with outputs removed between runs.  

### Gate

- All supported configurations either pass or fail at configure time with a
  precise missing-capability message.  
- Tests cannot reuse artifacts from a previous invocation.  

**Status:** pending  

## Phase 5: Numerical Regression and Edge Coverage

**Objective:** close remaining correctness gaps in transformed output and
published metadata.  

### Tasks

- [ ] Add nontrivial `decimate > 1` equivalence tests for the compact centering
  path.  
- [ ] Cover non-divisible active extents and document fallback semantics.  
- [ ] Verify 1-D, 2-D, and 3-D Blueprint meshes independently.  
- [ ] Verify zero local chunks, zero active particles, null species, and many
  species through both C++ and Python.  
- [ ] Verify external views remain valid through `execute()` and are not
  retained afterward, including exception paths.  
- [ ] Run sanitizer coverage if the toolchain permits.  

### Gate

- Existing packed field output is numerically equivalent for every supported
  decimation case.  
- Edge cases pass without invalid pointer arithmetic or shape ambiguity.  

**Status:** pending  

## Phase 6: Documentation, CI, and Delivery

**Objective:** align user documentation and automation with corrected behavior.  

### Tasks

- [ ] Correct Python examples after finalizing explicit Ascent construction and
  vector/raw/particle APIs.  
- [ ] Document capability requirements and conditional test behavior.  
- [ ] Document the fixed-width species naming contract.  
- [ ] Document moment-refresh cost and when it is incurred.  
- [ ] Add or propose a separate optional Ascent CI workflow using a pinned
  image or cached prefix.  
- [ ] Keep normal CI Ascent-free.  
- [ ] Update the final architecture/schema/configuration/copy report.  
- [ ] Record any unrun platform tests, especially Fugaku and reduced-capability
  Ascent builds, with reproducible commands.  

### Final Verification

```sh
clang-format -i <all modified C++ files>
ruff format python/src/picnix python/tests
ruff check python/src/picnix python/tests
cmake --build <ascent-disabled-build> --parallel
ctest --test-dir <ascent-disabled-build> --output-on-failure
cmake --build <ascent-enabled-build> --parallel
ctest --test-dir <ascent-enabled-build> -L ASCENT --output-on-failure
ctest --test-dir <ascent-enabled-build> --output-on-failure
python -m pytest python/tests
graphify update .
```

### Gate

- All P0/P1 findings are fixed and regression-tested.  
- Full enabled, disabled, standalone, consumer, and Python checks pass.  
- Remaining limitations are explicit and do not contradict documented APIs.  

**Status:** pending  

## Dependency Order

```text
Phase 0 reproduction
        |
        v
Phase 1 Python correctness ----+
        |                       |
        v                       v
Phase 2 MPI/data correctness -> Phase 3 end-to-end
        |                       |
        +----------+------------+
                   v
          Phase 4 build/tests
                   |
                   v
          Phase 5 edge coverage
                   |
                   v
          Phase 6 delivery
```

## Recommended Commit Sequence

1. `test: reproduce Ascent review findings`  
2. `fix: reconstruct Ascent data from Conduit nodes`  
3. `fix: synchronize Ascent validation and refresh moments`  
4. `test: exercise Ascent through PicApplication`  
5. `build: make Ascent tests portable and capability-aware`  
6. `test: close Ascent numerical and edge-case coverage`  
7. `docs: finalize Ascent integration guidance`  

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MPI tests hang while reproducing asymmetric failure | Use short CTest timeouts and an explicit expected-failure harness |
| Python fixes work for mappings but not Conduit | Make real Conduit nodes the authoritative integration fixture |
| Moment refresh adds unexpected cost | Call only when `um` is published and use the existing per-step cache |
| Lazy import repair reintroduces heavy embedded dependencies | Define explicit lazy export metadata and test embedded import separately |
| Public Ascent dependency leaks into disabled builds | Guard all target propagation under `PICNIX_ENABLE_ASCENT` |
| Capability checks overconstrain valid installations | Separate core MPI runtime, Python extract, and rendering requirements |
| Test cleanup removes user data | Restrict cleanup to dedicated CMake-generated test directories |

## Completion Checklist

- [ ] All high-severity findings fixed.  
- [ ] All medium-severity findings fixed or explicitly accepted.  
- [ ] Real producer-to-Python path covers raw/vector/particle data.  
- [ ] MPI asymmetric validation cannot deadlock.  
- [ ] Raw moments are current and order-independent.  
- [ ] Tests are isolated and labeled.  
- [ ] Standalone and downstream-consumer builds pass.  
- [ ] Documentation matches tested behavior.  
- [ ] Full verification matrix passes.  
