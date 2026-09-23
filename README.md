# PIC-NIX

A parallel Particle-In-Cell (PIC) simulation code for collisionless space plasmas with dynamic load balancing.  
This is based on the kinetic plasma simulation framework `nix`, which is included as a subtree.  
A separate repository for `nix` can be found [here](https://github.com/amanotk/nix).

## Contents

- [Requirements](#requirements)
- [Build](#build)
  - [Clone](#clone)
  - [Compile (Easiest Way)](#compile-easiest-way)
- [Run](#run)
- [Python Analysis](#python-analysis)
- [Documentation](#documentation)
- [Development](#development)

## Requirements

- C++ compiler supporting C++17 or later
- CMake version 3.23 or later
- MPI library (OpenMPI, MPICH, etc.)

The ordinary C++ dependencies are downloaded and built automatically by CMake.

## Build

### Clone

Clone the repository to a local working directory via:

```sh
git clone git@github.com:amanotk/pic-nix.git
```

### Compile (Easiest Way)

The code can be compiled with `cmake`.  
The easiest way is to use a pre-configured cache file provided in the `cmake` directory:

```sh
cmake -S . -B build -C cmake/linux-gcc.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This uses `mpicxx` with a `g++` backend and OpenMP enabled.  
The `-DCMAKE_BUILD_TYPE=Release` enables optimizations and disables assertions.  
See [Building](https://amanotk.github.io/pic-nix/build/) for optional
dependencies, reusable dependency installations, CMake options, and
host-specific configurations.  

## Run

You can now execute `main.out` using `mpiexec` (or `mpirun`).  
For example, you can run a simulation with default setup in `pic/example/beam/twostream`:

```sh
cd build/pic/example/beam/twostream
export OMP_NUM_THREADS=2
mpiexec -n 8 ../main.out -e 86400 -t 200 -c config.toml
```

In this example, you use 8 MPI processes, each launching 2 threads.  
The simulation parameters are read from `config.toml`, and the example writes
diagnostics below `data/`.  See the
[Diagnostics and I/O](https://amanotk.github.io/pic-nix/diagnostics/)
documentation for output selection and file layouts, and
[Configuration](https://amanotk.github.io/pic-nix/configuration/) for checkpoint
settings.  

## Python Analysis

The `picnix` Python package provides data analysis tools for simulation output.  
Install it from the repository root inside a virtual environment using `uv`:

```sh
uv venv .venv
uv pip install --python .venv -e ./python
```

After finishing the simulation, activate that environment and run the quick-look
script from the example directory:

```sh
. .venv/bin/activate
cd build/pic/example/beam/twostream
python quicklook.py data/profile.msgpack
```

You will now see image files `twostream-XXXXXXXX.png` for each snapshot and `twostream.mp4`, which is a movie file encoded by using `ffmpeg`.

## Documentation

The [PIC-NIX documentation](https://amanotk.github.io/pic-nix/) covers building
the code, configuration and output, equations and units, command-line tools, and
the `picnix` Python module. Documentation for the `develop` branch is available
at <https://amanotk.github.io/pic-nix/develop/>.  

## Development

For build & test instructions, language server setup, PIC integration workflow,  
git hooks, and other development topics, see [DEVELOPMENT.md](DEVELOPMENT.md).
