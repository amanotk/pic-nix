# Findings: Ascent Branch Review

## Review Scope

- Branch: `experimental/ascent-integration`  
- Base: `origin/experimental/ascent-integration`  
- Commits reviewed: `1c8e370`, `8e92f94`, `7384d65`, `a5cda82`, `db8e761`  
- Untracked `.python-version` is unrelated and must remain untouched.  

## Confirmed High-Severity Findings

### Rank-Asymmetric Actions Validation Can Deadlock

`AscentDiag` checks `std::filesystem::is_regular_file()` and returns locally.
Peers that see the file continue into collective `MPI_Comm_dup()`, while the
returning rank reaches the application's diagnostic barrier. Node-local or
inconsistently mounted configuration directories can therefore deadlock.  

Relevant files:  

- `pic/diag/ascent.cpp`  
- `pic/insitu/ascent_runtime.cpp`  
- `nix/application.cpp`  

### Raw Moments Are Not Refreshed

Raw publication includes `um`, but `AscentDiag` does not call
`interface->calculate_moment()`. Moments are refreshed only by diagnostics such
as history or field output, making Ascent data stale and diagnostic-order
dependent.  

Relevant files:  

- `pic/diag/ascent.cpp`  
- `pic/diag/field.hpp`  
- `pic/pic_application.cpp`  

### Real Conduit Raw Components Are Indexed as Flat Scalars

The C++ publisher binds raw storage as a flat external leaf. Python
`RawField.component()` indexes that leaf before reconstructing shape and
strides. Reproduction against installed Conduit returned scalar `0.0` for a
component that should have been a multidimensional array.  

### Real Conduit Vector Fields Ignore Component Names

Centered vectors are Conduit object nodes with `x`, `y`, and `z` children, not
Python `Mapping` objects. `Field.component("x")` therefore ignores the name and
returns all child arrays with the wrong shape. Reproduction returned `(3, 4)`
instead of `(1, 2, 2)`.  

### Real Conduit Particle Rows Are Never Reconstructed

Particle values are also flat external leaves. Python slices scalar values
instead of `[Np_allocated, 7]` rows, and `ids` raises `IndexError`. Null species
have no `values` leaf and currently produce ambiguous zero-dimensional arrays.  

Relevant Python file for all three data-shape findings:  

- `python/src/picnix/insitu/dataset.py`  

## Confirmed Medium-Severity Findings

### Stale Test Outputs Can Satisfy Assertions

MPI tests write into `${tmpdir}_cwd`, but preparation does not clear that
directory and cleanup removes `${tmpdir}` instead. Existing build directories
retain both the extract result and render image.  

### Species Naming Diverges at Index 10

C++ emits `species_10` for species 10, while Python requests `species_010`.
Indices 10 through 99 cannot be looked up through the documented Python API.  

### `Dataset.from_ascent()` Imports a Nonexistent Symbol

Ascent injects `ascent_data()` into extract globals; it does not export the
function from the `ascent` package. The installed package reproduces the
`ImportError`.  

### Wildcard Import Compatibility Regressed

The lazy package resolver resolves `picnix.__all__` from `ohm.__all__`.
`from picnix import *` now exposes Ohm helpers but drops previous names such as
`Run`, `Tracer`, and `get_wk_spectrum`.  

### Public Headers Do Not Propagate Ascent Usage Requirements

Ascent-facing headers are reachable through PIC's public include directory,
but `ascent::ascent_mpi` is linked privately. Consumers including those headers
do not inherit required include paths and compile definitions.  

### Standalone PIC Uses the Wrong Python Source Path

The test uses `${CMAKE_SOURCE_DIR}/python/src`; standalone PIC makes that
`<repo>/pic/python/src`. The hard-coded colon path separator is also not
CMake-portable.  

### Capability Checks Do Not Match Tests

Configuration requires only MPI-enabled Ascent, while tests unconditionally
require Python/Conduit Python/mpi4py and VTK-h rendering. Valid reduced-capability
installations fail at test time rather than being rejected or conditionally
tested.  

### Lifecycle Test Does Not Prove Ordering

The MPI test harness initializes MPI before `Application`, so `Application`
does not finalize MPI. The test's `MPI_Finalized()` checks remain false even if
diagnostic shutdown is moved after `finalize_mpi()`.  

### No End-to-End `AscentDiag` Test

Current integration tests call `BlueprintBuilder` and `AscentRuntime` directly.
Registration, scheduling, relative paths, options, moment refresh, and shutdown
through `Application::diagnostic()` remain untested.  

### Ascent Tests Lack an `ASCENT` Label

`ctest -L ASCENT -N` selects zero tests, so the planned targeted verification
command succeeds without testing Ascent.  

## Residual Test Gaps

- No rank-asymmetric actions visibility test.  
- No rank with zero local domains.  
- No real-Conduit extract coverage for raw fields, vectors, particles, or IDs.  
- No nontrivial decimation equivalence test.  
- No standalone PIC or public-consumer compile test.  
- No reduced-capability Ascent configure test.  
- No Ascent-enabled CI job.  

## Existing Passing Baselines

These passing suites are useful regression baselines but do not invalidate the
findings because their fixtures do not cover the failing representations and
paths.  

| Suite | Result |
|-------|--------|
| Ascent-enabled CTest | 53/53 passed |
| Ascent-disabled CTest | 45/45 passed |
| Python tests | 96/96 passed |

## Implementation Decisions for Remediation

- Real Conduit nodes are authoritative for the Python adapter.  
- Configuration/path validation must be collective before any Ascent
  collective.  
- Moment calculation is conditional on publishing `um`.  
- Species keys use fixed-width three-digit formatting.  
- Test outputs live only in dedicated directories cleared before each run.  
- Ascent core, Python extract, and rendering capabilities are checked
  separately.  
- Public import compatibility is preserved unless the user approves a breaking
  change.  
