# MPI Thread-Mode Performance Validation

This document provides the context and procedure for validating PIC-NIX MPI and
OpenMP performance on systems other than Fugaku.  It is intended to be usable by
both human researchers and an LLM working from a fresh checkout without access
to the investigation history.  

## Scope

The implementation under test is commit `00fcea6` on
`feature/pic-performance-profiler` (draft PR
[#50](https://github.com/amanotk/pic-nix/pull/50)).  It provides explicit
`funneled`, `multiple`, and `auto` MPI thread modes and schema-v2 sampled PIC
performance records.  

The main questions are:  

1. Does FUNNELED execution remove long-tail MPI stalls and final-barrier waits?
2. What throughput or OpenMP-efficiency cost does bulk-synchronous FUNNELED
   execution introduce?
3. Does the system provide a correct and efficient `MPI_THREAD_MULTIPLE`
   implementation?
4. Are observed effects specific to Fugaku MPI, or reproducible with other MPI
   libraries and interconnects?
5. Do both modes preserve numerical and checkpoint behavior?

## Investigation Context

A 1,536-node Fugaku PIC shock run developed severe rank-local timing variability
near step 41000 even with regular field and particle output disabled.  In the
controlled schema-v2 run, rank-local push timing had approximately the following
distribution:  

| Metric | Approximate time |
| --- | ---: |
| Median push | 0.45 s |
| p95 push | 3.28 s |
| Maximum push | 5.02 s |
| Median final-barrier wait | 4.57 s |
| `current_waitall` after slowdown | 3.34 s |
| `field_begin` after slowdown | 1.65 s |

Healthy current and particle initiation remained below approximately 0.01 s.
The key amplification mechanism was an application-wide OpenMP critical section
around MPI calls.  If one worker blocked in an MPI wait, every other worker on
that rank was prevented from entering MPI, producing an MPI-lock convoy.  

Particle traffic or MPI progress may still be the initiating source of a delayed
message.  The thread-mode change removes the lock convoy; it does not assume that
all underlying communication variability disappears.  This distinction is
important when interpreting results.  

## Implementation Under Test

Set the mode under `[application.option]`:  

```toml
[application.option]
mpi_thread_mode = "funneled" # or "multiple" or "auto"
```

The modes behave as follows:  

| Mode | Behavior |
| --- | --- |
| `funneled` | OpenMP workers perform packing, computation, and unpacking; OpenMP thread 0 performs MPI in explicit communication phases. |
| `multiple` | Workers retain dynamic chunk scheduling and may call MPI concurrently. Requires `MPI_THREAD_MULTIPLE`. |
| `auto` | Selects MULTIPLE when the runtime provides it and otherwise selects FUNNELED. |

An explicit mode controls the level requested from `MPI_Init_thread`.  A forced
MULTIPLE run aborts during initialization if the runtime does not provide
`MPI_THREAD_MULTIPLE`.  This is an expected compatibility guard, not a benchmark
failure.  Fugaku currently rejects forced MULTIPLE and must use FUNNELED.  

The `MPI_THREAD_MULTIPLE` CMake option controls the request used by `auto` and is
`ON` by default.  Record its value from the build's `CMakeCache.txt`; do not
infer it from the runtime configuration.  

## Required Correctness Checks

Performance data is valid only after these checks pass:  

1. Build with tests enabled and run the focused application test.
2. Use at least two OpenMP threads with dynamic team sizing disabled.
3. Run a portable build with `MPI_THREAD_MULTIPLE=OFF` to cover FUNNELED on
   systems that provide only SERIALIZED or FUNNELED support.
4. On systems that provide MULTIPLE, run a second build with
   `MPI_THREAD_MULTIPLE=ON` and confirm both mode sections pass.
5. On systems without MULTIPLE, confirm a production executable forced to
   `mpi_thread_mode = "multiple"` exits nonzero during initialization.
6. Run a small representative simulation through setup, moment deposition,
   particle push, field exchange, and finalization.
7. Verify the final checkpoint status is `complete` and records the expected MPI
   process count and final step.
8. Compare physical histories or invariants between modes within an explicitly
   stated numerical tolerance.

Recommended focused test command:  

```sh
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
ctest --test-dir build-funneled -R test_pic_application --output-on-failure
# On a MULTIPLE-capable system, repeat with --test-dir build-multiple.
```

The 8-rank `test_pic_application` path exercises setup exchange, moment exchange,
particle push, Gauss-law checks, and checkpoint handling.  The test runner itself
requests the thread level selected at build time.  A build with
`MPI_THREAD_MULTIPLE=ON` therefore cannot run on an MPI implementation that does
not provide MULTIPLE; use the portable OFF build there.  

## Build Record

Use optimized builds representative of production.  Start from the general
instructions in [`DEVELOPMENT.md`](../../DEVELOPMENT.md).  Configure a portable
FUNNELED test build on every system:  

```sh
cmake -S . -B build-funneled \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DBUILD_TESTING=ON \
  -DMPI_THREAD_MULTIPLE=OFF
cmake --build build-funneled --parallel
```

If the MPI runtime provides `MPI_THREAD_MULTIPLE`, configure a separate MULTIPLE
build:  

```sh
cmake -S . -B build-multiple \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DBUILD_TESTING=ON \
  -DMPI_THREAD_MULTIPLE=ON
cmake --build build-multiple --parallel
```

Record all of the following before running:  

| Category | Required information |
| --- | --- |
| Source | Commit SHA, branch, and whether the worktree is dirty |
| Hardware | CPU model, sockets, cores, NUMA layout, memory, and interconnect |
| Software | Compiler and flags, MPI implementation and version, OpenMP runtime |
| Build | `CMAKE_BUILD_TYPE`, `MPI_THREAD_MULTIPLE`, vectorization flags, optional dependencies |
| Launch | Scheduler, MPI ranks, ranks per node, binding/mapping options |
| OpenMP | `OMP_NUM_THREADS`, `OMP_DYNAMIC`, `OMP_PROC_BIND`, `OMP_PLACES` |
| Workload | Configuration file, grid, chunks, particles per cell, species, restart step |
| Diagnostics | Every enabled diagnostic and its interval |

Do not compare executables built with different optimization, vectorization, or
floating-point settings unless the purpose is specifically to measure those
differences.  

## Benchmark Matrix

Use explicit modes for the primary comparison.  `auto` is a portability check,
not a substitute for identifying which algorithm ran.  

| Case | Source | Build option | Runtime mode | Purpose |
| --- | --- | --- | --- | --- |
| A | `00fcea6` or later | Same for all current cases | `funneled` | Measure the portable lock-free funnel path. |
| B | Same commit as A | Same as A | `multiple` | Measure concurrent MPI when supported. |
| C | Same commit as A | Same as A | `auto` | Confirm platform selection behavior. |
| D, optional | `0f8358a` | `MPI_THREAD_MULTIPLE=OFF` | Not available in old code | Reproduce the former SERIALIZED critical-section baseline. |
| E, optional | `0f8358a` | `MPI_THREAD_MULTIPLE=ON` | Not available in old code | Compare the former worker-driven MULTIPLE path. |

On a MULTIPLE-capable system, use the same `build-multiple` executable for cases
A, B, and C so build differences cannot affect the mode comparison.  Use
`build-funneled` for portable FUNNELED testing on a system without MULTIPLE.  

Use separate build directories or Git worktrees for historical baselines.  Do
not rebuild different commits into the same directory.  

Run at least three repetitions per case when allocation cost permits.  Randomize
or alternate case order if system load varies over time.  A useful workload has
multiple chunks per rank, particle crossings between chunks, and enough steps to
move beyond initialization and warm-up behavior.  

### Establishing the AUTO mode

The current log does not record the MPI-provided thread level or effective AUTO
mode.  Do not label an AUTO result as FUNNELED or MULTIPLE from the TOML value
alone.  For a normal executable that initializes MPI itself, query the provided
level with the same request used by its build:  

```sh
mpicxx -O2 -x c++ -o mpi-thread-level - <<'CPP'
#include <mpi.h>
#include <cstring>
#include <iostream>

int main(int argc, char** argv)
{
  const bool build_multiple = argc < 2 || std::strcmp(argv[1], "serialized") != 0;
  const int requested = build_multiple ? MPI_THREAD_MULTIPLE : MPI_THREAD_SERIALIZED;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided >= MPI_THREAD_MULTIPLE) {
    std::cout << "provided=MPI_THREAD_MULTIPLE effective=multiple\n";
  } else if (provided >= MPI_THREAD_FUNNELED) {
    std::cout << "provided=" << provided << " effective=funneled\n";
  } else {
    std::cout << "provided=" << provided << " effective=unsupported\n";
  }
  MPI_Finalize();
}
CPP

# Match a build with MPI_THREAD_MULTIPLE=ON.
mpiexec -n 1 ./mpi-thread-level multiple

# Match a build with MPI_THREAD_MULTIPLE=OFF.
mpiexec -n 1 ./mpi-thread-level serialized
```

Record both the requested and provided levels.  For primary performance results,
prefer explicit FUNNELED or MULTIPLE configurations so the executed algorithm is
unambiguous.  

## Concrete Run Procedure

The repository shock example provides a communication-heavy PIC workload.  Its
supplied configuration is one-dimensional and exercises communication only in
the x direction.  Its executable is
`<build>/pic/example/shock/main.out`, and its starting configuration is
[`pic/example/shock/config.toml`](../../pic/example/shock/config.toml).  Adapt the
grid and particle count to the available allocation while retaining multiple
chunks per rank in active dimensions.  For multidimensional testing, set `Ny`
and/or `Nz` and the corresponding chunk counts above one, then validate that
adapted workload separately before using its timing results.  

For every mode and repetition:  

1. Create a new empty run directory outside the source tree, for example
   `campaign/<system>/<mode>/run-01/`.
2. Copy the same validated input configuration into each run directory.
3. Use a unique `basedir = "data"` inside each run directory; never share a data
   directory, log, or output checkpoint between cases.
4. Set only the intended `mpi_thread_mode` difference between matched cases.
5. For runs initialized independently, set `seed_type = "fixed"` under
   `[application.option]`.  The default is random and will not reproduce the
   same particles from identical TOML files.  An immutable common checkpoint is
   the preferred alternative for restart comparisons.
6. Add the performance configuration shown below.
7. Remove the `field` and `particle` diagnostic blocks from the copied shock
   configuration, or set their `begin` values beyond the final measured step.
   Retain `history` and `resource` only if their intervals are identical across
   cases.
8. Start every matched run from the same initial condition or immutable input
   checkpoint.  Write each output checkpoint to its own run directory.
9. Launch from the run directory and preserve scheduler stdout/stderr with the
   MessagePack log.

Example launch from one isolated run directory:  

```sh
export OMP_NUM_THREADS=6
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

RANKS=8
EXE=/absolute/path/to/build-funneled/pic/example/shock/main.out
mpiexec -n "$RANKS" "$EXE" \
  -c config.toml \
  -t 50 \
  -e 3600 \
  -s final
```

Here `-t` is maximum physical time and `-e` is maximum elapsed seconds.  Treat
`-e` only as a safety margin and set it high enough for every mode to reach the
same `-t`.  Reject or rerun a case that ends at a different final step or
physical time, even if it writes a complete checkpoint.  For a restart
comparison, add `-l /absolute/path/to/immutable/input-checkpoint` and use a unique
`-s` output prefix.  The example value `-t 50` is illustrative; choose a duration
that includes warm-up followed by a sufficiently long common measurement
window.  

On a system expected not to provide MULTIPLE, copy the validated configuration
to an isolated directory, set `mpi_thread_mode = "multiple"`, and run a minimal
launch.  A nonzero exit with
`` `mpi_thread_mode = multiple` requires MPI_THREAD_MULTIPLE `` is the expected guard
result:  

```sh
mpiexec -n 1 /absolute/path/to/build-funneled/pic/example/shock/main.out \
  -c config-multiple.toml -t 0 -e 60
test "$?" -ne 0
```

## Runtime Controls

Keep these controls identical across compared cases:  

- MPI rank count and rank placement.
- OpenMP thread count, affinity, and dynamic-team setting.
- Grid and chunk decomposition.
- Particle count and random seed.
- Restart checkpoint or initial condition.
- Rebalance configuration.
- Diagnostic and profiler intervals.
- Requested wall-clock and physical-time limits.

Disable large field and particle output during the measured interval unless I/O
is itself under investigation.  Retain only the lightweight diagnostics needed
to validate physics and load balance.  

Pin OpenMP behavior explicitly.  Site-specific values may differ, but record
them verbatim:  

```sh
export OMP_NUM_THREADS=6
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Use the site's recommended MPI process and thread binding.  Verify the actual
placement rather than assuming launcher defaults.  

## Profiler Configuration

Enable sampled schema-v2 profiling under `[application.performance]`:  

```toml
[application.performance]
enabled = true
interval = 100
offset = 0
```

Sampling every step is useful for small validation runs.  For production runs,
an interval such as 100 limits profiler and log overhead while retaining trend
information.  Use the same interval and offset in compared cases.  

The profiler records:  

- Rank-local push time and final `MPI_Barrier` wait.
- Phase wall time, OpenMP efficiency, and maximum chunk time for `advance`,
  `current_field`, `particle_probe`, `particle_exchange`, and `field_exchange`.
- MPI-operation timing for `current_begin`, `particle_begin`,
  `current_waitall`, `field_begin`, `particle_probe`, `particle_waitall`, and
  `field_waitall`.
- Across-rank summary statistics including mean, median, p95, maximum, and the
  rank owning the maximum.

FUNNELED operation totals are recorded on the master thread, whereas MULTIPLE
operation work is distributed among workers.  Compare total push, barrier, and
phase wall time directly.  Interpret operation `thread_max` and OpenMP
efficiency together with the execution mode rather than assuming identical
thread-level distributions.  

## Analysis Commands

Install the Python package as described in
[`DEVELOPMENT.md`](../../DEVELOPMENT.md#python-analysis-package-picnix), then
analyze the MessagePack log:  

```sh
.venv/bin/picnix-log-analyze /path/to/run/data/log.msgpack \
  --csv timing.csv \
  --plot timing.png \
  --no-progress
```

The analyzer accepts either `log.msgpack` or a profile path from which the log
can be resolved.  Preserve the original MessagePack logs; CSV and plots are
derived artifacts.  

Use matched step ranges for comparisons.  Exclude startup, checkpoint, and
diagnostic steps unless they are the subject of the test.  Report at least the
median, p95, and maximum, because the original issue was dominated by tail
latency rather than mean compute growth.  

## What To Analyze

### Overall behavior

- Median, p95, and maximum rank-local push time.
- Final-barrier wait.  If a scalar fraction is needed, compute
  `barrier.mean / (local.mean + barrier.mean)` for each sampled step and report
  its temporal distribution as a ratio of rank means.  Do not divide unrelated
  medians or p95 values.
- Steps per second or particle updates per second.
- Run-to-run variability.

### OpenMP behavior

- Phase wall time and `omp_efficiency`.
- Maximum chunk time and whether it grows with the slow-rank tail.
- CPU utilization and idle time if system profilers are available.
- Whether FUNNELED barriers create unacceptable worker idle time.

### MPI behavior

- `current_waitall`, `particle_waitall`, and `field_waitall` tails.
- `field_begin` inflation, which was a signature of the former lock convoy.
- Particle-probe duration and evidence of head-of-line blocking.
- Rank owning each maximum and whether the same ranks recur.
- MPI progress, message rate, and network counters when site tools expose them.

Schema-v2 operation entries contain `total`, `thread_max`, and `max_call`, and
each is summarized across ranks for every sampled step.  Always name the full
metric path and the temporal aggregation.  For example:  

```text
median over sampled steps of
performance.operation.current_waitall.max_call.p95
```

Here the inner `p95` is across ranks at one sampled step; the outer median is
across sampled steps.  Do not interpret operation `total` as elapsed wall time
when comparing FUNNELED and MULTIPLE: MULTIPLE totals may sum overlapping worker
calls.  Use push and phase wall metrics for elapsed-time comparisons, and use
`max_call` to study operation tails.  

### Load and physics behavior

- Chunk and particle load distribution at matched steps.
- Rebalance events near timing discontinuities.
- Conserved quantities, history diagnostics, and Gauss-law error.
- Checkpoint completeness and restart success.

## Interpretation Guide

| Observation | Likely interpretation |
| --- | --- |
| FUNNELED sharply reduces p95/max push and barrier wait | The former critical-section convoy was a major amplifier. |
| FUNNELED removes `field_begin` inflation but waits remain | The lock convoy is fixed, but delayed messages or MPI progress remain. |
| MULTIPLE is faster with stable tails | The platform has useful `MPI_THREAD_MULTIPLE` support. |
| MULTIPLE has worse tails or CPU overhead | The MPI implementation's thread synchronization or progress path is expensive. |
| FUNNELED has low phase efficiency but stable MPI | Bulk synchronization is correct but may need overlap or task-based optimization. |
| Both modes slow at the same steps | Investigate particle traffic, load migration, topology, or network behavior beyond the removed lock. |
| Barrier wait grows while one rank's push grows | Barrier time is primarily a symptom of rank imbalance, not an independent root cause. |

Do not claim that an MPI operation caused a slowdown solely because its wait time
is large.  A wait may reveal that the matching sender or progress engine was
delayed elsewhere.  Correlate operation timing with rank ownership, phase time,
chunk load, and network data.  

## Result Template

Create one result record per system and test campaign:  

```markdown
# MPI Thread-Mode Results: <system>

## Environment
- Commit:
- Dirty worktree:
- CPU/node:
- Interconnect:
- Compiler:
- MPI implementation/version:
- MPI thread level requested/provided:
- Requested/effective application mode:
- CMake `MPI_THREAD_MULTIPLE`:
- Ranks/nodes/ranks per node:
- OpenMP environment and binding:

## Workload
- Configuration:
- Initial or restart step:
- Grid/chunks/particles:
- Measured step range:
- Repetitions:
- Enabled diagnostics:
- Profiler interval/offset:

## Correctness
- Focused tests:
- Checkpoint status:
- Restart result:
- Physics/invariant comparison and tolerance:

## Performance
| Metric | FUNNELED | MULTIPLE | AUTO | Historical baseline |
| --- | ---: | ---: | ---: | ---: |
| Temporal median of `performance.push.local.median` | | | | |
| Temporal median of `performance.push.local.p95` | | | | |
| Temporal maximum of `performance.push.local.max` | | | | |
| Temporal median of `performance.push.barrier.median` | | | | |
| Temporal median of `performance.operation.current_waitall.max_call.p95` | | | | |
| Temporal median of `performance.operation.field_begin.max_call.p95` | | | | |
| Temporal median of `performance.operation.particle_probe.max_call.p95` | | | | |

## Findings
-

## Raw Artifacts
- Log:
- CSV:
- Plot:
- Scheduler output:
```

## Current Fugaku Status

The small Fugaku FUNNELED validation completed five steps using 8 MPI ranks and
2 OpenMP threads, produced schema-v2 operation records, and wrote a complete
checkpoint.  Mean push time was approximately 31 ms for that functional case.
A forced MULTIPLE validation was rejected by the runtime guard because Fugaku
does not provide `MPI_THREAD_MULTIPLE`.  

The 1,536-node FUNNELED production validation is job `49998836`.  It restarts
from step 38479 with regular field and particle output disabled in the measured
window.  Its results are pending and should be compared with the archived
schema-v1 job `49961148` and controlled schema-v2 job `49969653`.  Job identifiers
and absolute run paths are site-local context, not portable benchmark inputs.  
