# Building

## Requirements

- CMake 3.23 or newer.  
- A C++17 compiler.  
- An MPI implementation and its C++ compiler wrapper.  

CMake can download the ordinary C++ dependencies automatically. For repeated
builds, installing them once into a local prefix is faster.  

## Optional requirements

The default build does not search for PETSc, Ascent, or ADIOS2. Install and
enable only the integrations needed by a simulation:  

| Feature | Requirement | CMake configuration |
| --- | --- | --- |
| PETSc solvers | PETSc discoverable through `pkg-config` | `-DPICNIX_ENABLE_PETSC=ON` |
| Ascent diagnostics | An MPI-enabled Ascent installation providing `ascent::ascent_mpi` | `-DPICNIX_ENABLE_ASCENT=ON -DPICNIX_ASCENT_ROOT=/path/to/ascent` |
| ADIOS2 output | ADIOS2 2.11 or newer providing `adios2::cxx_mpi` | `-DPICNIX_ENABLE_ADIOS2=ON -DPICNIX_ADIOS2_ROOT=/path/to/adios2` |
| C++ tests | Catch2 v3 | `-DBUILD_TESTING=ON` |

`scripts/install_ascent.sh` and `scripts/install_adios2.sh` provide reproducible
source-build helpers. `scripts/install_dependencies.sh` installs the ordinary
C++ dependencies and Catch2 into a reusable prefix.  

## Prepare a build stack

`scripts/prepare_build_stack.sh` prepares a reusable stack in one directory
(typically outside the repository). Give an explicit path; do not move the
directory afterwards because stamps and `env.sh` store absolute paths.  

```sh
scripts/prepare_build_stack.sh "$HOME/picnix-stack" --cache cmake/linux-gcc.cmake
```

By default this creates:

| Path | Contents |
| --- | --- |
| `<stack>/python` | uv virtual environment (system Python preferred) with `mpi4py`, `numpy`, and editable `picnix` |
| `<stack>/deps` | Ordinary C++ dependencies |
| `<stack>/env.sh` | Environment activation script |
| `<stack>/stamp` | Idempotency markers keyed by compiler fingerprint |

Optional components are off by default:

```sh
scripts/prepare_build_stack.sh "$HOME/picnix-stack" \
  --cache cmake/linux-gcc.cmake \
  --with-adios2 \
  --with-ascent
```

| Flag | Effect |
| --- | --- |
| `--with-adios2` | Build ADIOS2 (C++/MPI only; Python bindings off) into `<stack>/adios2` |
| `--with-adios2-python` | Also build ADIOS2 Python bindings into the stack venv |
| `--with-ascent` | Build Ascent **slim** (zlib, Conduit, VTK-m, Ascent only — no HDF5/Silo/ZFP/MFEM/RAJA, no Sphinx/docs/examples) into the stack venv |
| `--ascent-full` | With `--with-ascent`: upstream full third-party set (much slower; needs Cython for ZFP) |
| `--no-deps` | Skip the ordinary C++ dependencies |
| `--no-picnix` | Skip the editable `picnix` install (generic stack) |
| `--check` | Validate an existing stack without building |
| `--force` | Ignore stamps and rebuild |
| `--jobs N` | Parallel build jobs (default **4**; `CMAKE_BUILD_PARALLEL_LEVEL` also works) |

Activate the stack and configure PIC-NIX with the same compiler cache:

```sh
source "$HOME/picnix-stack/env.sh"
cmake -S . -B build -C cmake/linux-gcc.cmake \
  -DCMAKE_PREFIX_PATH="$HOME/picnix-stack/deps" \
  -DPICNIX_USE_SYSTEM_LIBS=ON
```

`env.sh` puts the stack Python on `PATH` and exports `LD_LIBRARY_PATH` /
`CMAKE_PREFIX_PATH` without sourcing venv `activate`, so **your shell prompt
does not change**. Source the same file in **job scripts** as well: the C++
binaries linked to Ascent or ADIOS2 need `LD_LIBRARY_PATH`, and Python
analysis or Ascent extracts need the stack interpreter on `PATH`. If the MPI
launcher does not forward the environment, pass those variables explicitly
(for example `mpiexec -x LD_LIBRARY_PATH ...`).  

The same `--cache` file is forwarded to ADIOS2 (via `-C`) and exported as
`CC`/`CXX`/`CFLAGS`/`CXXFLAGS` for Ascent. This matters for
`cmake/linux-intel.cmake`, where `mpiicpc` only selects `icpx` through
`-cxx=icpx` in `CMAKE_CXX_FLAGS`; without those flags the Intel MPI wrapper
falls back to classic `icpc`.  

By default the stack builds with **4 parallel jobs** (override with
`--jobs N` or `CMAKE_BUILD_PARALLEL_LEVEL`). Full `nproc` on large hosts
often overruns memory during VTK-m / ADIOS2 compiles.  

When `--with-adios2` or `--with-ascent` was used, add
`-DPICNIX_ENABLE_ADIOS2=ON -DPICNIX_ADIOS2_ROOT=...` and/or
`-DPICNIX_ENABLE_ASCENT=ON -DPICNIX_ASCENT_ROOT=...` (the generated `env.sh`
prints a ready-to-use configure line).  

The script resolves `mpicc`/`mpicxx` from `--cache` (or `--mpicc`/`--mpicxx`)
and records a compiler fingerprint. Re-running with the same inputs skips
finished steps; a changed compiler or cache invalidates them. Use
`--check` after environment or module changes.  

Build parallelism defaults to **4** jobs (`--jobs` / `CMAKE_BUILD_PARALLEL_LEVEL`
override). Full `nproc` often overruns memory on Intel `icpx` and VTK-m.  

Cross-compilation caches may build the default deps-only stack.
`--with-adios2` and `--with-ascent` require a native build because the
selected Python interpreter runs during those builds.  

### ADIOS2 Python reader

By default `--with-adios2` builds only the C++ library. Analysis reads BP
datasets with a separate Python package (see [Diagnostics](diagnostics.md)):

```sh
uv pip install --python .venv -e "./python[adios]"
```

Use `--with-adios2-python` only when the reader must match the built ADIOS2
exactly (same version and MPI). Do not install the PyPI `adios2` package into
the stack venv in that mode; bindings are exposed through a path file the
script writes.  

## Standard build

The supplied Linux/GCC initial-cache file selects `mpicxx`, OpenMP, native CPU
instructions, and optimization:  

```sh
cmake -S . -B build -C cmake/linux-gcc.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Because `cmake/linux-gcc.cmake` uses `-march=native`, its binaries are intended
for the machine on which they are built. Configure explicitly when binaries
must run on different CPUs.  

## Preinstalled dependencies

Install the C++ dependencies into a reusable prefix and point CMake to it:  

```sh
scripts/install_dependencies.sh "$HOME/usr"
cmake -S . -B build -C cmake/linux-gcc.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/usr"
cmake --build build --parallel
```

Use the same initial-cache or toolchain options when cross-compiling the
dependencies and PIC-NIX.  

## Common CMake options

| Option | Default | Purpose |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the C++ test suite. |
| `PICNIX_BUILD_EXAMPLE` | `ON` | Build the simulation examples. |
| `PICNIX_USE_SYSTEM_LIBS` | `ON` | Prefer installed dependencies before FetchContent. |
| `PICNIX_ENABLE_PETSC` | `OFF` | Enable PETSc-based elliptic solvers. |
| `PICNIX_ENABLE_ASCENT` | `OFF` | Enable Ascent in-situ diagnostics. |
| `PICNIX_ENABLE_ADIOS2` | `OFF` | Enable ADIOS2 diagnostic output. |
| `MPI_THREAD_MULTIPLE` | `ON` | Compile for MPI thread-multiple support. |

For a developer build with tests:  

```sh
cmake -S . -B build -C cmake/linux-gcc.cmake -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Do not run the CTest suite in parallel: many tests start MPI jobs that can
interfere with one another.  

## Host configurations

The files in `cmake/` are CMake initial-cache files, not universal host
configurations. Inspect and adapt compiler wrappers, optimization flags,
modules, runtime libraries, and scheduler launch commands for each site.  

| Environment | Initial-cache file | Notes |
| --- | --- | --- |
| Linux, GCC, MPI | `cmake/linux-gcc.cmake` | Uses `mpicxx`, OpenMP, and `-march=native`. |
| Linux, Intel oneAPI and Intel MPI | `cmake/linux-intel.cmake` | Uses `mpiicpc` with the `icpx` backend. |
| Fugaku | `cmake/fugaku-*-cross.cmake` | Cross-compilation from a login node. |

The current Fugaku LLVM example is `cmake/fugaku-llvm22-cross.cmake`. It
disables `MPI_THREAD_MULTIPLE` and documents the required compiler module and
compute-node library staging. Site details may change independently of PIC-NIX.  

See [`nix/DEPENDENCIES.md`](https://github.com/amanotk/pic-nix/blob/main/nix/DEPENDENCIES.md)
for dependency versions and the full cross-compilation example.  
