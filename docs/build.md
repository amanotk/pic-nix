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
