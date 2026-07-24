# PIC-NIX

A parallel Particle-In-Cell (PIC) simulation code for collisionless space plasmas with dynamic load balancing.  
This is based on the kinetic plasma simulation framework `nix`, which is included as a subtree.  
A separate repository for `nix` can be found [here](https://github.com/amanotk/nix).

## Contents

- [Requirements](#requirements)
- [Build](#build)
  - [Clone](#clone)
  - [Compile](#compile)
- [Run](#run)
- [Post-processing](#post-processing)
  - [Python Analysis Package](#python-analysis-package)
  - [Quick Look](#quick-look)
- [Development](#development)

## Requirements

- C++ compiler supporting C++17 or later
- CMake version 3.20 or later
- MPI library (OpenMPI, MPICH, etc.)

## Build

### Clone

Clone the repository to a local working directory via:

```
$ git clone git@github.com:amanotk/pic-nix.git
```

### Compile

The code can be compiled with `cmake`.  
The simplest way is to use a pre-configured cache file provided in the `cmake` directory:

```
$ cmake -S . -B build -C cmake/linux-gcc.cmake -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```

This uses `mpicxx` with a `g++` backend and OpenMP enabled.  
The `-DCMAKE_BUILD_TYPE=Release` enables optimizations and disables assertions.  
See [DEVELOPMENT.md](DEVELOPMENT.md) for manual configuration and advanced options.  
Please also refer to [CMake Reference Documentation](https://cmake.org/cmake/help/latest/).

## Run

You can now execute `main.out` using `mpiexec` (or `mpirun`).  
For example, you can run a simulation with default setup in `pic/example/beam/twostream`:

```
$ cd build/pic/example/beam/twostream
$ export OMP_NUM_THREADS=2
$ mpiexec -n 8 ../main.out -e 86400 -t 200 -c config.toml
```

In this example, you use 8 MPI processes, each launching 2 threads.  
The simulation parameters will be read from the configuration file `config.toml`.

Available command-line options will be shown with the `--help` option:

```
$ ./main.out --help
usage: ./main.out --config=string [options] ...
options:
  -c, --config     configuration file (string)
  -l, --load       prefix of snapshot to load (string [=])
  -s, --save       prefix of snapshot to save (string [=])
  -t, --tmax       maximum physical time (double [=1.79769e+308])
  -e, --emax       maximum elapsed time [sec] (double [=3600])
  -v, --verbose    verbosity level (int [=0])
  -?, --help       print this message
```

## Post-processing

### Python Analysis Package

The `picnix` Python package provides data analysis tools for simulation output.  
It is recommended to install it inside a virtual environment using `uv`:

```
# Create a virtual environment
$ uv venv .venv

# Install the package
$ uv pip install --python .venv -e ./python
```

You can also install from other locations:

```
# Pointing to a local clone
$ uv pip install --python .venv -e /path/to/pic-nix/python

# Without cloning (install directly from git)
$ uv pip install --python .venv "git+https://github.com/amanotk/pic-nix.git#subdirectory=python"
```

After installation, `import picnix` works from any directory.

See [picnix Python Package](docs/picnix/README.md) for package documentation,
including installed command line tools such as `picnix-hdf5-convert`.

### Quick Look

After finishing the simulation, you can run the following command in the same directory:

```
$ uv run python quicklook.py data/profile.msgpack
```

You will now see image files `twostream-XXXXXXXX.png` for each snapshot and `twostream.mp4`, which is a movie file encoded by using `ffmpeg`.

## Development

For build & test instructions, language server setup, PIC integration workflow,  
git hooks, and other development topics, see [DEVELOPMENT.md](DEVELOPMENT.md).
