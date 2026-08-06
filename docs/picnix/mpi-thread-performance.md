# MPI Communication Performance Validation

This document defines a portable procedure for validating changes to PIC-NIX
MPI and OpenMP communication. It is intended for developers and operators
running on different machines, MPI implementations, and interconnects. The
procedure separates correctness, portability, and performance so that a faster
result is not accepted when it changes simulation behavior.

## Validation Goals

A communication change should answer the following questions:

1. Does it preserve numerical results, boundary exchange, checkpointing, and
   restart behavior?
2. Does it work with the MPI thread levels provided by the target system?
3. Does it improve typical performance or reduce communication tails without an
   unacceptable throughput or OpenMP-efficiency regression?
4. Does it behave consistently as the number of nodes, chunks, and particles
   increases?
5. Are results reproducible across repeated runs and distinguishable from
   system noise?

## Implementations To Compare

Use immutable source revisions and record the full commit SHA for every
executable. For a completion-polling change, compare at least these two
implementations:

| Implementation | Description | Expected profiler schema |
| --- | --- | ---: |
| Ordered FUNNELED reference | Completes each local chunk with blocking waits in container order. | 2 |
| Fair FUNNELED candidate | Polls all incomplete local chunks so a delayed chunk does not prevent progress on later chunks. | 3 |

The completion strategy is selected by the source revision, not by a runtime
option. Build the reference and candidate in separate worktrees and build
directories. Do not rebuild two revisions into the same directory. The test
coordinator must provide both full SHAs in the campaign record before testing
starts; stop and request them if either revision is ambiguous.

The candidate also supports these runtime modes under `[application.option]`:

```toml
[application.option]
mpi_thread_mode = "funneled" # or "multiple" or "auto"
```

| Mode | Behavior |
| --- | --- |
| `funneled` | OpenMP workers perform computation, packing, and unpacking. OpenMP thread 0 performs MPI calls in explicit communication phases. |
| `multiple` | Workers may call MPI concurrently. This requires `MPI_THREAD_MULTIPLE`. |
| `auto` | Selects MULTIPLE when the runtime provides it, selects FUNNELED for any other provided level at least as strong as `MPI_THREAD_FUNNELED`, and rejects lower levels. |

Use explicit modes for performance comparisons. Use `auto` only as a
portability and mode-selection check.

## Required Test Matrix

Run all applicable cases below. A system without `MPI_THREAD_MULTIPLE` may omit
the MULTIPLE performance case, but it must run the rejection check.

| Case | Source revision | Build option | Runtime mode | Purpose |
| --- | --- | --- | --- | --- |
| A | Ordered reference | `MPI_THREAD_MULTIPLE=OFF` | `funneled` | Establish ordered-completion correctness and performance. |
| B | Fair-polling candidate | `MPI_THREAD_MULTIPLE=OFF` | `funneled` | Measure the effect of fair completion polling. |
| C | Fair-polling candidate | `MPI_THREAD_MULTIPLE=ON` | `multiple` | Validate worker-driven MPI on a MULTIPLE-capable system. |
| D1 | Fair-polling candidate | `MPI_THREAD_MULTIPLE=OFF` | `auto` | Confirm AUTO behavior when the build requests SERIALIZED. |
| D2 | Fair-polling candidate | `MPI_THREAD_MULTIPLE=ON` | `auto` | Confirm AUTO behavior when the build requests MULTIPLE. |
| E | Fair-polling candidate | `MPI_THREAD_MULTIPLE=OFF` | forced `multiple` | Confirm clean rejection when MULTIPLE is unavailable. |

Cases A and B are the primary comparison for completion polling. Cases B and C
compare MPI execution modes, not only completion order, and must be reported as
a separate comparison. Cases D1 and D2 are portability checks and must not be
combined unless they selected the same effective mode.

Run at least three repetitions of each performance case when allocation cost
permits. Alternate or randomize case order when system load may vary.

## Build Requirements

Use optimized builds representative of production. Start from the general
instructions in [`DEVELOPMENT.md`](../../DEVELOPMENT.md).

Configure a portable build on every system:

```sh
cmake -S . -B build-funneled \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DBUILD_TESTING=ON \
  -DMPI_THREAD_MULTIPLE=OFF
cmake --build build-funneled --parallel
```

On a system that provides `MPI_THREAD_MULTIPLE`, configure a separate build:

```sh
cmake -S . -B build-multiple \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DBUILD_TESTING=ON \
  -DMPI_THREAD_MULTIPLE=ON
cmake --build build-multiple --parallel
```

The `MPI_THREAD_MULTIPLE` CMake option controls the level requested by the test
runner and by `auto`. Record its value from `CMakeCache.txt`; do not infer it
from the runtime configuration.

The commands use `mpicxx` and `mpiexec` as placeholders. Substitute the target
platform's MPI compiler wrapper and scheduler launcher when they differ.

Keep compiler versions, optimization flags, vectorization settings, optional
dependencies, and floating-point options identical between matched reference
and candidate builds.

## Environment Record

Record the following before running tests:

| Category | Required information |
| --- | --- |
| Source | Full commit SHA, branch or tag, and whether the worktree is dirty |
| Hardware | CPU model, sockets, cores, NUMA layout, memory, and interconnect |
| Software | Compiler, MPI implementation and version, and OpenMP runtime |
| Build | Build type, CMake options, compiler flags, and optional dependencies |
| Launch | Scheduler, node count, rank count, ranks per node, and binding options |
| OpenMP | `OMP_NUM_THREADS`, `OMP_DYNAMIC`, `OMP_PROC_BIND`, and `OMP_PLACES` |
| Workload | Configuration, grid, chunks, particles per cell, species, and initial step |
| Output | Diagnostics, profiler interval, checkpoint settings, and measured steps |

Record the MPI thread level requested by the build and provided by the runtime.
Do not label an `auto` result as FUNNELED or MULTIPLE from its TOML value alone.

The application does not currently write its effective AUTO mode to the
performance log. Use a small probe that requests the same build-time level:

```sh
mpicxx -O2 -x c++ -o mpi-thread-level - <<'CPP'
#include <mpi.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
  const bool request_multiple = argc == 2 && std::atoi(argv[1]) != 0;
  const int requested = request_multiple ? MPI_THREAD_MULTIPLE : MPI_THREAD_SERIALIZED;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, requested, &provided);

  std::cout << "requested=" << requested << " provided=" << provided;
  if (provided >= MPI_THREAD_MULTIPLE) {
    std::cout << " effective=multiple\n";
  } else if (provided >= MPI_THREAD_FUNNELED) {
    std::cout << " effective=funneled\n";
  } else {
    std::cout << " effective=unsupported\n";
  }

  MPI_Finalize();
}
CPP

# Match MPI_THREAD_MULTIPLE=OFF and MPI_THREAD_MULTIPLE=ON builds, respectively.
mpiexec -n 1 ./mpi-thread-level 0
mpiexec -n 1 ./mpi-thread-level 1
```

Run the probe through the same launcher and software environment as the
application. Record the numeric requested/provided values and interpreted
effective mode. The `-x c++ -` compilation syntax is illustrative; use an
equivalent source file and compiler-wrapper syntax when the platform does not
accept GCC-style standard-input compilation.

## Correctness Tests

Performance results are valid only after all applicable correctness checks pass.

1. Run the focused application test with at least two OpenMP threads and dynamic
   team sizing disabled:

   ```sh
   export OMP_NUM_THREADS=2
   export OMP_DYNAMIC=FALSE
   ctest --test-dir build-funneled \
     -R test_pic_application --output-on-failure
   ```

2. Run the boundary and profiler tests under multiple MPI ranks. Include at
   least 2 and 8 ranks when the system permits.
3. On a MULTIPLE-capable system, repeat the application and profiler tests with
   the MULTIPLE build.
4. Run a representative simulation through setup, moment deposition, particle
   exchange, field exchange, diagnostics, and finalization.
5. Verify that the final checkpoint status is `complete` and contains the
   expected MPI process count, final step, and physical time.
6. Restart from the checkpoint and advance at least one additional step.
7. Compare physical histories, conserved quantities, particle counts, and
   Gauss-law error between matched cases. Define numerical tolerances before
   examining results and report them with the comparison.
8. On a system without MULTIPLE, force `mpi_thread_mode = "multiple"` and verify
   that initialization exits nonzero with the expected compatibility error.

The 8-rank `test_pic_application` path exercises setup exchange, moment
exchange, particle push, Gauss-law checks, and checkpoint handling. It does not
replace a representative application run with the target compiler, MPI library,
binding, and workload.

## Workload Design

Use a workload with all of these properties:

- More than one chunk per MPI rank.
- Particle crossings between chunks.
- Communication in every dimension used by the intended production workload.
- Enough particles per rank to represent production compute and message sizes.
- Enough steps to separate initialization and warm-up from steady operation.
- At least one load-balancing event if dynamic load balancing is used in
  production.

The shock example at `pic/example/shock/` is a convenient starting point. Its
default configuration is one-dimensional, so increase `Ny`, `Nz`, `Cy`, and
`Cz` when multidimensional communication is part of the target workload.
Validate any adapted configuration before using its timing results.

Use at least three scales when practical:

| Scale | Purpose |
| --- | --- |
| Functional | Fast correctness, checkpoint, and profiler-schema validation |
| Intermediate | Inter-node communication and repeated-run comparison |
| Representative | Production-like chunks, particles, topology, and node count |

For weak scaling, keep cells, chunks, and particles per rank approximately
constant. For strong scaling, keep the global problem fixed and report the
changing work per rank. State explicitly which scaling method is used.

## Controlled Run Procedure

For every mode and repetition:

1. Create a new empty run directory outside the source tree.
2. Copy the same validated input configuration into every matched directory.
3. Use a unique output directory for every run. Never share logs or checkpoints
   between cases.
4. Set only the intended source revision or `mpi_thread_mode` difference.
5. Use `seed_type = "fixed"` for independently initialized comparisons, or start
   every case from the same immutable checkpoint.
6. Use identical rank placement, OpenMP affinity, rebalance settings, profiler
   settings, diagnostics, wall-clock limit, and final physical time.
7. Disable large field and particle output during the measured interval unless
   I/O is the subject of the test.
8. Preserve the configuration, executable SHA, scheduler script, stdout,
   stderr, MessagePack log, and checkpoint status.

Example OpenMP controls:

```sh
export OMP_NUM_THREADS=6
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Use the site's recommended MPI binding and verify actual placement instead of
assuming launcher defaults. Replace `mpiexec` in the examples with the site's
scheduler launcher when required.

Example launch:

```sh
RANKS=8
EXE=/absolute/path/to/build/pic/example/shock/main.out

mpiexec -n "$RANKS" "$EXE" \
  -c config.toml \
  -t 50 \
  -e 3600 \
  -s final
```

Here `-t` is the maximum physical time and `-e` is an elapsed-time safety limit.
Every matched case must reach the same final step and physical time. Reject or
rerun a truncated case even if it writes a complete checkpoint.

## Profiler Configuration

Enable sampled profiling in every performance run:

```toml
[application.performance]
enabled = true
interval = 100
offset = 0
```

Sampling every step is useful for short validation runs. A larger interval
reduces overhead in long runs. Use the same interval and offset for every
matched case.

The profiler records:

- Rank-local push time and final-barrier wait.
- Phase wall time, OpenMP efficiency, and maximum chunk time.
- MPI initiation, probe, wait, and polling operation timings.
- Across-rank mean, median, p95, maximum, and rank owning the maximum.

Schema 2 records blocking completion under `current_waitall`,
`particle_waitall`, and `field_waitall`. Schema 3 adds `current_poll`,
`particle_poll`, and `field_poll` for fair FUNNELED completion.

Use the analyzer from the candidate revision for both schema-2 and schema-3
logs. It supports both formats. Treat operations that are inactive for an
implementation as `N/A`; do not compare their emitted zero values as timings.

## Required Performance Measurements

Report temporal median, p95, and maximum over a common steady-state step range.
Do not report only whole-run averages.

### Overall

- Rank-local push time.
- Final-barrier wait and its fraction of total push time.
- Steps per second or particle updates per second.
- Run-to-run variation.

### OpenMP

- Phase wall time and OpenMP efficiency.
- Maximum chunk time.
- Worker idle time or CPU utilization when system tools provide it.
- Time spent at internal FUNNELED phase barriers when an external OpenMP
  profiler or an instrumented build can measure it. The built-in profiler does
  not record these internal barriers separately.

### MPI

- Current, particle, and field completion phase wall time.
- Blocking wait tails for the ordered implementation.
- Polling phase and individual-call tails for the fair implementation.
- Particle-probe duration.
- MPI initiation time; inflation here can indicate MPI thread contention.
- Rank owning each maximum and whether slow ranks recur.
- Message rate, progress-engine behavior, and network counters when available.

### Load And Physics

- Chunk, cell, and particle distribution across ranks.
- Rebalance events near timing discontinuities.
- Conserved quantities, history diagnostics, and Gauss-law error.
- Checkpoint completeness and restart success.

## Analysis Procedure

Generate a per-step CSV and preserve the original MessagePack log:

```sh
.venv/bin/picnix-log-analyze /path/to/run/data/log.msgpack \
  --csv timing.csv \
  --plot timing.png \
  --no-progress
```

Use the same inclusive step range for all matched cases. Exclude initialization,
checkpoint, and diagnostic steps unless they are being measured.

The analyzer's text report includes averages across sampled records. Calculate
temporal statistics from the CSV rather than copying an `avg` column from the
text report. Set an inclusive step range and list any diagnostic, checkpoint, or
rebalance steps that must be excluded. This standard-library example prints the
temporal median, p95, and maximum for selected metrics:

```sh
export START_STEP=100
export END_STEP=1000
export EXCLUDE_STEPS=200,400

python - timing.csv <<'PY'
import csv
import math
import os
import statistics
import sys

metrics = (
    "performance.push.local.median",
    "performance.push.local.p95",
    "performance.push.local.max",
    "performance.push.barrier.median",
    "performance.phase.current_field.wall.p95",
    "performance.phase.particle_exchange.wall.p95",
    "performance.phase.field_exchange.wall.p95",
)

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

start = int(os.environ["START_STEP"])
end = int(os.environ["END_STEP"])
excluded = {
    int(value)
    for value in os.environ.get("EXCLUDE_STEPS", "").split(",")
    if value.strip()
}
rows = [
    row
    for row in rows
    if row.get("step") and start <= int(row["step"]) <= end and int(row["step"]) not in excluded
]

if len(rows) >= 2 and not excluded:
    first = rows[0]
    last = rows[-1]
    elapsed = float(last["timestamp_unixtime"]) - float(first["timestamp_unixtime"])
    advanced_steps = int(last["step"]) - int(first["step"])
    print(f"steps_per_second={advanced_steps / elapsed:.9g}")
elif excluded:
    print("steps_per_second=N/A: select a contiguous interval without exclusions")

def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

for metric in metrics:
    values = [float(row[metric]) for row in rows if row.get(metric)]
    if values:
        print(
            f"{metric}: median={statistics.median(values):.9g} "
            f"p95={percentile(values, 0.95):.9g} max={max(values):.9g}"
        )
PY
```

Add schema-specific wait or poll metric paths to `metrics` as needed. Record the
step filter and exact aggregation for every reported value.

End-to-end steps per second requires a contiguous interval with no excluded
intermediate steps. Choose a clean contiguous interval for throughput, or report
throughput as `N/A`; do not subtract selected steps while retaining their wall
time between the endpoints.

An operation path contains two levels of aggregation. For example,
`performance.operation.current_poll.max_call.p95` is the p95 across ranks at
one sampled step. Taking its median across CSV rows produces the temporal median
of that per-step rank p95.

Operation `total` is accumulated time inside MPI calls. It is not necessarily
elapsed wall time when calls overlap across MULTIPLE worker threads. Use push
and phase wall metrics for elapsed-time comparisons; use `max_call` and `total`
to investigate operation behavior.

## Interpretation Rules

| Observation | Interpretation to investigate |
| --- | --- |
| Fair polling reduces completion p95/max and barrier wait | Ordered completion was causing head-of-line blocking or insufficient progress. |
| Fair polling changes poll-call timing but not phase wall time | Polling order changed without improving end-to-end completion. |
| Fair polling increases CPU use or phase wall time | Polling overhead may exceed its progress benefit for this workload. |
| MULTIPLE is faster with stable tails | The MPI implementation provides effective concurrent progress. |
| MULTIPLE has worse tails or CPU overhead | MPI thread synchronization may be expensive on this platform. |
| All implementations slow at the same steps | Investigate load migration, particle traffic, topology, or network behavior. |
| Barrier wait grows with one rank's push time | The barrier is exposing rank imbalance rather than causing it. |

A long MPI wait does not prove that MPI caused the delay. The matching sender,
load imbalance, process scheduling, or the progress engine may be responsible.
Correlate operation timing with phase wall time, slow-rank identity, load, and
system counters.

## Acceptance Criteria

Define project-specific performance thresholds before examining candidate
results. At minimum, a candidate is acceptable only when:

- All required correctness tests pass.
- Checkpoint and restart behavior are valid.
- Physics and invariant differences remain within the predefined tolerances.
- No deadlock, timeout, MPI error, or unbounded memory growth occurs.
- The candidate does not introduce an unexplained regression in median
  throughput, tail latency, or OpenMP efficiency.
- Results are consistent across repetitions and at more than one scale.
- Any unsupported MPI thread mode fails clearly rather than silently selecting
  unsafe behavior.

A candidate that passes functional testing but lacks inter-node or
representative-scale evidence should remain experimental.

## Result Template

```markdown
# MPI Communication Results: <system>

## Environment
- Reference commit:
- Candidate commit:
- Dirty worktrees:
- CPU/node and interconnect:
- Compiler and flags:
- MPI implementation/version:
- MPI thread level requested/provided:
- CMake options:
- Nodes/ranks/ranks per node:
- OpenMP environment and binding:

## Workload
- Configuration and checksum:
- Initial condition or checkpoint:
- Grid/chunks/particles:
- Scaling method:
- Warm-up and measured step ranges:
- Repetitions:
- Diagnostics and profiler interval:

## Correctness
- Focused tests:
- Final checkpoint status:
- Restart result:
- Physics/invariant tolerance and result:

## Performance
| Metric | Ordered FUNNELED | Fair FUNNELED | MULTIPLE | AUTO/OFF | AUTO/ON |
| --- | ---: | ---: | ---: | ---: | ---: |
| `median(performance.push.local.median)` | | | | | |
| `median(performance.push.local.p95)` | | | | | |
| `max(performance.push.local.max)` | | | | | |
| `median(performance.push.barrier.median)` | | | | | |
| `median(performance.phase.current_field.wall.p95)` | | | | | |
| `median(performance.phase.particle_exchange.wall.p95)` | | | | | |
| `median(performance.phase.field_exchange.wall.p95)` | | | | | |
| `median(performance.phase.current_field.omp_efficiency.median)` | | | | | |
| `median(performance.phase.particle_exchange.omp_efficiency.median)` | | | | | |
| `median(performance.phase.field_exchange.omp_efficiency.median)` | | | | | |
| `median(performance.operation.current_waitall.max_call.p95)` | | N/A | | | |
| `median(performance.operation.particle_waitall.max_call.p95)` | | N/A | | | |
| `median(performance.operation.field_waitall.max_call.p95)` | | N/A | | | |
| `median(performance.operation.current_poll.max_call.p95)` | N/A | | N/A | | |
| `median(performance.operation.particle_poll.max_call.p95)` | N/A | | N/A | | |
| `median(performance.operation.field_poll.max_call.p95)` | N/A | | N/A | | |
| `(last_step - first_step) / (last_timestamp - first_timestamp)` over a contiguous interval | | | | | |

## Findings
-

## Raw Artifacts
- Configuration:
- MessagePack log:
- CSV and plot:
- Scheduler stdout/stderr:
- Checkpoint status:
```

For AUTO, fill wait or poll rows according to its recorded effective mode and
source implementation. Mark inactive operations `N/A` rather than zero.
