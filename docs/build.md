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
| `--with-adios2` | Build ADIOS2 (C++/MPI only; Python bindings off) into `<stack>/adios2`; supports Fugaku aarch64 cross-builds |
| `--with-adios2-python` | Also build ADIOS2 Python bindings; in cross mode uses `--cross-python` with target aarch64 Python 3.11, NumPy, and mpi4py |
| `--with-ascent` | Build the **slim** MPI/Python Ascent profile without VTK-h rendering (zlib, Conduit, and Ascent; no HDF5/Silo/ZFP/MFEM/RAJA, VTK-m, or Sphinx/docs/examples) |
| `--with-ascent-rendering` | Add VTK-m and VTK-h scene/volume rendering to the slim profile; implies `--with-ascent` and works with native or Fugaku aarch64 builds |
| `--with-ascent-full` | Use the upstream full third-party set (native builds only; much slower; needs Cython for ZFP). Rendering remains controlled by `--with-ascent-rendering`. |
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
does not change**. Use it on the login node for builds and analysis. In
**job scripts**, source `<stack>/compute-env.sh` instead: for native builds it
delegates to `env.sh`, and for cross builds it selects the aarch64 Python and
libraries so the C++ binaries and Python extracts run on compute nodes. If the
MPI launcher does not forward the environment, pass those variables explicitly
(for example `mpiexec -x LD_LIBRARY_PATH ...`).  

The same `--cache` file is forwarded to ADIOS2 (via `-C`) and exported as
`CC`/`CXX`/`CFLAGS`/`CXXFLAGS` for Ascent. This matters for
`cmake/linux-intel.cmake`, where `mpiicpc` only selects `icpx` through
`-cxx=icpx` in `CMAKE_CXX_FLAGS`; without those flags the Intel MPI wrapper
falls back to classic `icpc`.  

By default the stack builds with **4 parallel jobs** (override with
`--jobs N` or `CMAKE_BUILD_PARALLEL_LEVEL`). Full `nproc` on large hosts
often overruns memory during VTK-m / ADIOS2 compiles.  

When `--with-adios2`, `--with-ascent`, or `--with-ascent-rendering` was used, add
`-DPICNIX_ENABLE_ADIOS2=ON -DPICNIX_ADIOS2_ROOT=...` and/or
`-DPICNIX_ENABLE_ASCENT=ON -DPICNIX_ASCENT_ROOT=...` (the generated `env.sh`
prints a ready-to-use configure line).  

The script resolves `mpicc`/`mpicxx` from `--cache` (or `--mpicc`/`--mpicxx`)
and records a compiler fingerprint. Re-running with the same inputs skips
finished steps; a changed compiler or cache invalidates them. Use
`--check` after environment or module changes.  

Build parallelism defaults to **4** jobs (`--jobs` / `CMAKE_BUILD_PARALLEL_LEVEL`
override). Full `nproc` often overruns memory on Intel `icpx` and VTK-m.  

Cross-compilation caches may build the default stack. Fugaku aarch64 caches
also support `--with-adios2`: the stack forwards the cache to ADIOS2, supplies
its target-specific FFS float-format result, and limits optional ADIOS2
libraries to those available for the target. The login-node Python environment
does not build target `mpi4py`; it installs a native ADIOS2 Python reader when
the editable `picnix` package is enabled. `--with-adios2-python` cross-builds
the bindings against the target aarch64 Python 3.11, NumPy, and mpi4py from
Fugaku's public Spack installation, while a host x86_64 venv runs build-time
scripts. Ascent's full profile still requires a native build.  
Fugaku aarch64 caches support both Ascent profiles. `--with-ascent`
cross-builds MPI-enabled Conduit and Ascent with target Python 3.11 extracts
without VTK-h rendering. `--with-ascent-rendering` adds VTK-m 2.3.0 and VTK-h
for scene and volume rendering. Both modes use a separate x86_64 Python 3.11
venv for build-time scripts, locate compatible target packages in Fugaku's
public Spack installation, and install target Python modules in
`<stack>/ascent/python-modules`, not in the login-node venv.  

### Native rendering profile

Rendering is an explicit capability, independent of the native build
environment:

```sh
scripts/prepare_build_stack.sh "$HOME/picnix-ascent-rendering" \
  --cache cmake/linux-gcc.cmake \
  --with-ascent-rendering
```

### Fugaku login-node build with LLVM 23

Fugaku login nodes are `x86_64`; the LLVM 23 module supplies `aarch64` MPI
cross-compilers. Load the module before building and use the same cache for
the dependency stack and PIC-NIX:  

```sh
module load LLVM/llvmorg-23.1.0
scripts/prepare_build_stack.sh "$HOME/picnix-llvm23" \
  --cache cmake/fugaku-llvm23-cross.cmake --with-adios2 --jobs 2
source "$HOME/picnix-llvm23/env.sh"
cmake -S . -B build-fugaku-llvm23 -C cmake/fugaku-llvm23-cross.cmake \
  -DCMAKE_PREFIX_PATH="$HOME/picnix-llvm23/deps" \
  -DPICNIX_USE_SYSTEM_LIBS=ON \
  -DPICNIX_ENABLE_ADIOS2=ON \
  -DPICNIX_ADIOS2_ROOT="$HOME/picnix-llvm23/adios2"
cmake --build build-fugaku-llvm23 --parallel 2
```

Use a separate stack and build directory from LLVM 22. Load the matching
module in the job script and follow the site instructions for staging LLVM
shared libraries on compute nodes. The stack's Python venv is built for the
login node and must not be used as a compute-node interpreter.  

The cache also acts as a CMake toolchain file so CMake knows it is
cross-compiling before it configures a project. Passing only target compiler
flags with `-C` does not accomplish this: CMake may otherwise try to run
`aarch64` probes on the login node. Use a **fresh build directory** if one
was previously configured with the older cache.  

For LLVM 22, replace `23` with `22` in the module, cache, stack path, and
build directory above. Keep a separate stack per compiler. The same stack
command builds ADIOS2 on the login node with either LLVM version. The
standalone installer remains available when only ADIOS2 is needed:  

```sh
module load LLVM/llvmorg-23.1.0
MPICC=mpiclang MPICXX=mpiclang++ CMAKE_BUILD_PARALLEL_LEVEL=2 \
  scripts/install_adios2.sh "$HOME/adios2-llvm23" --no-python --cross-build \
  -C cmake/fugaku-llvm23-cross.cmake \
  -DADIOS2_USE_MHS=OFF -DADIOS2_USE_PNG=OFF \
  -DADIOS2_USE_Sodium=OFF -DADIOS2_USE_OpenSSL=OFF \
  -DADIOS2_USE_CURL=OFF -DADIOS2_USE_Campaign=OFF \
  -DADIOS2_USE_Profiling=OFF \
  -DFFS_FLOAT_FORMAT_TEST:STRING=0 \
  -DFFS_FLOAT_FORMAT_TEST__TRYRUN_OUTPUT:STRING=Format_IEEE_754_littleendian
```

Run this from the repository root on the login node. The script clones ADIOS2
2.12.1 into a temporary directory and deletes the source/build tree after
completion; use a manual source build if you need to debug a failure. This
command and the stack-based builds were tested: the installed
`libadios2_cxx_mpi.so` is an `aarch64` library, and its CMake package was
successfully used to link a cross-compiled MPI application. The default
utilities also needed the target GCC 8 `stdc++fs` library (set by the cache)
and the build-tree linker search path supplied by `--cross-build`. Check any
additional libraries detected by CMake to ensure they are compiled for
`aarch64`.  
The ADIOS2 library has **not yet been run on a compute node**. To link PIC-NIX
against it, set `-DPICNIX_ENABLE_ADIOS2=ON` and
`-DPICNIX_ADIOS2_ROOT="$HOME/adios2-llvm23"` during configuration, and make
the installed ADIOS2 `lib` directory available at runtime.  

Ascent's installer drives an upstream superbuild. Native `--with-ascent`
builds install Conduit and Ascent Python extensions into the stack venv
without rendering; `--with-ascent-rendering` adds VTK-m and VTK-h. For
Fugaku cross builds, both Ascent profiles use a host x86_64 Python 3.11 venv
for build-time scripts while extensions compile against target aarch64 Python.
The target modules are exposed through `<stack>/compute-env.sh` rather than
the login-node venv.  

### ADIOS2 Python reader

By default `--with-adios2` builds only the C++ library. Analysis reads BP
datasets with a separate Python package (see [Diagnostics](diagnostics.md)):

```sh
uv pip install --python .venv -e "./python[adios]"
```

Use `--with-adios2-python` only when the reader must match the built ADIOS2
exactly (same version and MPI). In native builds the PyPI `adios2` package is
removed from the stack venv and the built bindings are exposed through a path
file the script writes. Cross builds keep the PyPI reader for login-node
analysis; the target bindings are exposed through `<stack>/compute-env.sh`
instead.  

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

The current Fugaku LLVM example is `cmake/fugaku-llvm23-cross.cmake`. It
disables `MPI_THREAD_MULTIPLE` and documents the required compiler module and
compute-node library staging. Site details may change independently of PIC-NIX.  

See [`nix/DEPENDENCIES.md`](https://github.com/amanotk/pic-nix/blob/main/nix/DEPENDENCIES.md)
for dependency versions and the full cross-compilation example.  
