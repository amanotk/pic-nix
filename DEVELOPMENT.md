# Development

This document describes the local developer workflow.

## Build & Test

### Configure

From the repository root, configure with MPI compiler and enable tests as
follows:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=mpicxx
```

On typical linux systems with GCC, use `cmake/linux-gcc.cmake` for better
optimization:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -C cmake/linux-gcc.cmake
```

To build with PETSc explicitly enabled, add
`-DPICNIX_ENABLE_PETSC=ON` (otherwise PETSc stays disabled by default).
Configuration for Intel oneAPI compilers is also available via
`cmake/linux-intel.cmake`.

### Build

After configuration, build with:

```sh
cmake --build build --parallel
```

For a clean build, add the `--clean-first` option:

```sh
cmake --build build --clean-first --parallel
```

On some systems including WSL2, you may need to limit the number of
parallel jobs for stability:

```sh
cmake --build build --parallel 4
```

### Test

After building, run the tests with:

```sh
ctest --test-dir build --output-on-failure
```

Do **not** use `-j` (parallelism) with `ctest` — many tests spawn MPI
processes and concurrent MPI jobs can interfere or deadlock.

When running MPI tests in a sandboxed environment, use escalated
permissions; otherwise PMIx can fail with `socket()` errors.
For a focused test run, use the `-R` option followed by the test name
pattern.

### PIC Unit Test

The PIC integration test is `test_pic_application` (MPI/Catch2).
It runs with `np=1` and `np=8` and creates a temporary base directory
under `PICNIX_TMPDIR`, which it cleans up inside the test code after
completion.
To run only these tests:

```sh
ctest --test-dir build -R test_pic_application --output-on-failure
```

## Python Analysis Package (`picnix`)

The `python/` directory contains the `picnix` Python package for
analyzing PIC-NIX simulation output (field data, particle data, load
balance diagnostics, etc.).

### Install (editable, for development)

From the repository root:

```sh
uv venv .venv
uv pip install --python .venv -e ./python
```

After this, `import picnix` works from any directory.

For Python tests and optional MPI-enabled tools, install extras:

```sh
uv pip install --python .venv -e "./python[test]"
uv pip install --python .venv -e "./python[mpi]"
```

The HDF5 diagnostic converter is installed as a console command:

```sh
picnix-hdf5-convert --input-dir /path/to/run/data
mpiexec -np 16 picnix-hdf5-convert --input-dir /path/to/run/data
```

### Install from another directory

Point `uv` at the `python/` subdirectory of any local clone:

```sh
uv pip install --python .venv -e /path/to/pic-nix/python
```

### Install from git (no clone needed)

```sh
uv pip install --python .venv "git+https://github.com/amanotk/pic-nix.git#subdirectory=python"
```

## Language Server

Generate the compilation database whenever you configure so `clangd`/your
LSP can resolve MPI headers and `nix/thirdparty` includes:

```sh
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -C cmake/linux-gcc.cmake
```

(or pass `-DCMAKE_CXX_COMPILER=mpicxx` manually) so
`build/compile_commands.json` mirrors the compiler's include paths.

- Ensure your editor points `clangd` at `build/` (for example,
  `--compile-commands-dir=build`).
- If your MPI compiler wrapper is not under a default system path, set
  `clangd` query-driver to include it (for example,
  `--query-driver=/path/to/spack/**/bin/mpicxx,/usr/bin/mpicxx,/usr/bin/mpic++`)
  so headers like `mpi.h` are resolved correctly.
- Point your editor's LSP to the `build/` directory (e.g.
  `clangd.arguments: ["--compile-commands-dir=build"]`).
- Anytime `build/` is deleted or rerun, repeat the configuration command
  above before restarting the language server so its cache stays in sync.

## Catch2 v3 Setup

Prefer an external Catch2 v3 install and point CMake at its config file.
Use the helper script to install Catch2 v3 into a custom prefix:

```sh
script/install_catch2v3.sh "$HOME/usr"
```

Then configure tests with the explicit config path:

```sh
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DPICNIX_CATCH2_CONFIG="$HOME/usr/lib/cmake/Catch2/Catch2Config.cmake"
```

## Smoke Test Golden Data

`pic/unittest/test_pic_chunk.cpp` writes smoke test golden data to
`pic/unittest/testdata/pic_chunk_smoke_*_{field,particle}.msgpack`.
Each msgpack file stores a map keyed by option tags (e.g.,
`vector_o3_Vay_WT`) so multiple option sweeps live in one file.
Regenerate the data by setting `PICNIX_UPDATE_GOLDEN=1` when running
`test_pic_chunk`.

## PIC Integration Workflow

### Overview

The Python workflow under `script/integration/pic/` provides
deterministic build, run, analyze, compare, and plot generation for
multiple PIC test cases.
Each case is defined in its own module under
`script/integration/pic/cases/`.

Entry point:

```sh
uv run python script/integration/pic/main.py <command> [case] [options]
```

### Quick Start

```sh
# Build the target executable
uv run python script/integration/pic/main.py build shock

# Run the simulation
uv run python script/integration/pic/main.py run shock

# Analyze output and compare against golden data
uv run python script/integration/pic/main.py analyze shock
uv run python script/integration/pic/main.py compare shock

# Or do all steps at once
uv run python script/integration/pic/main.py all shock

# Generate PNG plots for manual review
uv run python script/integration/pic/main.py plots shock
```

The default case is `twostream`.
Use `--compiler gcc` to select a compiler profile (currently only
`gcc`).

### Available Cases

| Case | Target | Grid | Species | tmax | nproc | Notes |
|---|---|---|---|---|---|---|
| `twostream` | beam | 256x1x1 / 16 chunks | 3 | 30 | 8 | Two-stream instability, no rebalance |
| `weibel-order{1..4}` | beam | 32x32x1 / 4x4 chunks | 4 | 30 | 8 | Weibel instability, shape-order sweep |
| `shock` | shock | 256x1x1 / 16 chunks | 2 | 160 | 8 | Shock with open boundaries, rebalance |

All cases use `seed_type = 'fixed'`, but fixed seeds should be understood
as reproducible for a given implementation, not as a guarantee of bitwise
portable output across all compilers, standard libraries, CPUs, and math
libraries.

### Golden Comparison And Random Numbers

The integration tests compare simulation summaries against golden data.
This is useful for catching regressions, but golden tests can be fragile
when random-number generation is part of the test case.

The random engine `std::mt19937_64` has a specified deterministic sequence,
but C++ standard-library distributions do **not** guarantee a bitwise-stable
algorithm.  For example, `std::uniform_real_distribution`,
`std::normal_distribution`, `std::poisson_distribution`, and
`std::gamma_distribution` are required to produce values with the requested
statistical distribution, but the exact samples produced from a fixed engine
seed may differ between libstdc++ versions, libc++, operating systems, or
compiler/library updates.

This affects the integration cases differently:

- `twostream` and `weibel-order{1..4}` use random numbers only to build the
  initial particle distribution in `pic/example/beam/main.cpp`.  After
  initialization, the run evolves a fixed particle set.  These cases are
  therefore kept as relatively strict golden-regression tests, but they may
  still need golden regeneration if the CI toolchain or standard library
  changes enough to alter the initial samples.
- `shock` uses stochastic open-boundary particle injection throughout the
  run.  Its boundary condition samples the number of injected particles and
  their velocities during the simulation.  A different Poisson/gamma sample,
  or even one different rejection-sampling decision, can desynchronize the
  random stream and produce a different valid stochastic realization.  The
  shock case is also nonlinear and can amplify small differences at late
  times.  For this reason, shock comparison uses case-specific physical and
  statistical checks rather than strict pointwise snapshot equality.

Do not blindly loosen global tolerances when an integration comparison
fails.  First identify whether the failure is a deterministic regression, a
floating-point/environment difference, or a stochastic-realization difference.
If exact cross-environment stochastic reproducibility is required in the
future, use project-owned portable sampling routines or a deterministic test
mode instead of relying on `std::*_distribution` output.

### CI Workflow

The integration workflow is intentionally not triggered on every push.
It runs on the scheduled workflow and can be launched manually on any branch
that contains the workflow file:

```sh
gh workflow run integration.yml --ref <branch-name>
```

This keeps routine pushes fast while allowing the full integration suite to
be run before merging or when investigating environment-dependent failures.

### Adding New Cases

Each case is an `IntegrationCase` dataclass defined in
`script/integration/pic/cases/`.
Key fields:

- `name` — case identifier (used as CLI argument and directory name)
- `target` — CMake target to build (e.g. `beam`, `shock`)
- `base_config` — path to upstream config TOML
- `config_overrides` — dict deep-merged into the base config
- `config_patch` — optional callable for TOML edits that deep-merge
  cannot handle (e.g. modifying `[[diagnostic]]` array entries)
- `generate_plots` — optional hook for case-specific plot generation
- `snapshot_times` — physical times at which field/particle snapshots
  are captured

Register the case in `cases/__init__.py` to make it available from the
CLI.

### Golden Data

Golden summaries live in `script/integration/pic/golden/<case>/` as
`summary.msgpack`.
To regenerate after intentional physics changes:

```sh
uv run python script/integration/pic/main.py update-golden <case>
```

The `compare` subcommand checks all numeric keys against golden data
with configurable tolerances (`--rtol`, `--atol`).

### Workspace Layout

The integration workspace is created under `run-integration-pic/`:

```
run-integration-pic/
  build-gcc/          # CMake build directory
  <case>/
    config.toml       # generated config (base + overrides)
    data/             # simulation output
    summary.json      # analysis output
    plots/            # generated PNG plots
```

This directory is gitignored.

## Git Hooks

Install the local pre-commit hook (clang-format on staged C/C++) after
cloning:

```sh
script/git-hooks/install.sh
```

## Git Subtree (`nix/`)

The `nix/` directory is a git subtree of the standalone repository
<https://github.com/amanotk/nix>.
Use `script/subtree-nix.sh` to sync `nix/` between the two repos.

### One-time setup

```sh
script/subtree-nix.sh setup
```

This adds a git remote named `nix` pointing to the upstream repository.

### Branch semantics

The `--branch` flag always refers to the **remote** (upstream `nix` repo)
branch. The **local** branch is whatever you are currently on in
`pic-nix` (detected automatically from `git branch --show-current`).

| pic-nix (local) | nix upstream (`--branch`) | Use case |
|---|---|---|
| `develop` | `develop` (default) | Day-to-day sync between integration branches |
| `feature/foo` | `feature/foo` | Push nix/ changes from a feature branch to a matching upstream branch |
| `develop` | `main` | Pull a release from upstream into pic-nix develop |

The default `--branch` value is `develop` (mapping `develop` ↔ `develop`).

### Commands

```sh
# Pull upstream develop into nix/ (most common)
script/subtree-nix.sh pull

# Pull a specific upstream branch
script/subtree-nix.sh pull --branch main

# Push nix/ subtree changes to upstream develop
script/subtree-nix.sh push

# Push to a specific upstream branch
script/subtree-nix.sh push --branch feature/new-balancer

# Fetch upstream refs without merging (for inspection)
script/subtree-nix.sh fetch

# Show upstream log
script/subtree-nix.sh log
```

Pull always uses `--squash` to keep history linear.

### Typical workflow

1. Pull upstream changes: `script/subtree-nix.sh pull`
2. Resolve any conflicts in `nix/`, commit the merge
3. If you modified `nix/` in pic-nix, push back:
   `script/subtree-nix.sh push --branch feature/<name>`,
   then open a PR on the upstream `nix` repo
4. After the upstream PR is merged, pull again to synchronize

## Graphify Snapshot

The project keeps a knowledge graph snapshot at `docs/graphify/`
(`graph.html`, `graph.json`, `GRAPH_REPORT.md`) for architecture
navigation.

### Prerequisites

Install the `graphify` CLI (see the graphify skill for details).

### Update the Snapshot

```sh
script/update-graphify-snapshot.sh
```

The script performs an incremental update:

1. Seeds `graphify-out/` from the existing snapshot (if present).
2. Runs `graphify update .` to re-extract only new or changed files.
3. Copies the outputs (`GRAPH_REPORT.md`, `graph.json`, `graph.html`)
   into `docs/graphify/`.
4. Rewrites absolute paths so the committed snapshot is portable
   across machines.

Commit the changes in `docs/graphify/` when done.

## CI Notes

The GitHub Actions workflow installs Catch2 v3 externally and sets
`PICNIX_CATCH2_CONFIG` for the test builds.
