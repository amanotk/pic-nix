# Progress Log: Ascent Integration Remediation

## Session: 2026-07-26

### Current Status

- **Phase:** 4 - build and test verification  
- **State:** P0/P1 implementation complete; edge-case coverage remains  
- **Branch:** `experimental/ascent-integration`  
- **Remote status:** five commits ahead of tracking branch  

### Actions Taken

- Reviewed all five commits ahead of
  `origin/experimental/ascent-integration`.  
- Read complete C++ runtime, diagnostic, Blueprint, CMake, test, and Python
  adapter files.  
- Delegated independent reviews of C++ core, Python API, and build/tests.  
- Verified review findings against installed Conduit and Ascent Python
  bindings.  
- Reproduced raw component scalar output, vector component shape corruption,
  particle `IndexError`, invalid `Dataset.from_ascent()` import, and wildcard
  export regression.  
- Verified `ctest -L ASCENT -N` selects zero tests.  
- Verified extract and render artifacts remain in their test working
  directories after cleanup.  
- Replaced the previous implementation plan with this remediation plan.  
- Reconstructed raw, centered-vector, particle, and empty particle views from
  real Conduit shapes and byte strides.  
- Restored explicit lazy top-level exports and changed `Dataset.from_ascent()`
  to require an injected node or callable.  
- Added collective actions-file validation, current-step moment refresh, and
  fixed-width C++ species naming.  
- Added an end-to-end `PicApplication` Ascent test covering relative actions
  paths and shutdown.  
- Made test preparation remove stale working directories, cleaned both output
  directories, labelled Ascent tests, gated Python/render tests by capability,
  and propagated the Ascent target publicly.  
- Verified root and standalone Ascent-enabled builds with MPI compiler wrappers.

### Reproduction Results

| Reproducer | Actual Result | Status |
|------------|---------------|--------|
| Real Conduit `RawField.component()` | Scalar `0.0` instead of multidimensional component | CONFIRMED |
| Real Conduit centered `Field.component("x")` | Wrong `(3, 4)` shape and ignored name | CONFIRMED |
| Real Conduit particle `ids` | `IndexError` from one-dimensional storage | CONFIRMED |
| `from ascent import ascent_data` | `ImportError` | CONFIRMED |
| `from picnix import *` | Missing `Run`, `Tracer`, and `get_wk_spectrum` | CONFIRMED |
| `ctest -L ASCENT -N` | Zero selected tests | CONFIRMED |
| Render/extract cleanup | Output artifacts remain under `*_cwd` | CONFIRMED |
| Missing Conduit particle node | `ParticleField` raised on empty storage | FIXED |

### Baseline Test Results

| Suite | Result |
|-------|--------|
| Ascent-enabled CTest | 55/55 passed after remediation |
| Ascent-disabled CTest | 45/45 passed before remediation |
| Python tests | 99/99 passed after remediation |
| `ctest -L ASCENT` | 10/10 passed; 10 tests selected |
| Standalone Ascent build | Configure and build passed |

### Errors Encountered

| Error | Resolution |
|-------|------------|
| Initial review subagent invocation from the user was cancelled | Performed a new scoped branch review with supported agents |
| Existing plan marked reviewed paths complete despite reproducible defects | Overwrote active planning files with remediation phases and gates |

### Next Action

Run the remaining optional edge/lifecycle checks, update the user documentation
for explicit `Dataset.from_ascent()` construction and capability-aware tests,
then review the final diff.  Do not modify `.python-version`.  
