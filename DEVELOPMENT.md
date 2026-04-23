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
python script/integration/pic/main.py <command> [case] [options]
```

### Quick Start

```sh
# Build the target executable
python script/integration/pic/main.py build shock

# Run the simulation
python script/integration/pic/main.py run shock

# Analyze output and compare against golden data
python script/integration/pic/main.py analyze shock
python script/integration/pic/main.py compare shock

# Or do all steps at once
python script/integration/pic/main.py all shock

# Generate PNG plots for manual review
python script/integration/pic/main.py plots shock
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

All cases use `seed_type = 'fixed'` for deterministic output.

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
`summary.msgpack` and `summary.json`.
To regenerate after intentional physics changes:

```sh
python script/integration/pic/main.py update-golden <case>
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

## CI Notes

The GitHub Actions workflow installs Catch2 v3 externally and sets
`PICNIX_CATCH2_CONFIG` for the test builds.
