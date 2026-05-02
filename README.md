# PIC-NIX
A Particle-In-Cell (PIC) simulation code for collisionless space plasmas.  
This is based on the kinetic plasma simulation framework `nix`, which is included as a subtree.  
A separate repository for `nix` can be found [here](https://github.com/amanotk/nix).

## Contents  
- [Build](#build)  
  - [Clone](#clone)  
  - [Compile](#compile)  
- [Tests](#tests)  
- [Run](#run)  
- [Post-processing](#post-processing)
  - [Python Analysis Package](#python-analysis-package)
  - [Quick Look](#quick-look)

## Build

### Clone
Clone the repository to a local working directory via:
```
$ git clone git@github.com:amanotk/pic-nix.git
```

### Compile
The code can be compiled with `cmake`, to which a proper C++ compiler and its compiler flags should be specified.  
The simplest way is to use a pre-configured cache file provided in `cmake` directory.  
For instance, you can do as follows in the repository's top directory:
```
$ cmake -S . -B build -C cmake/linux-gcc.cmake
$ cmake --build build
```
which means that you are going to use `mpicxx` as a C++ compiler with `g++` backend with OpenMP enabled.

This is equivalent to the following manual build configuration:
```
$ cmake -S . -B build \
	-DCMAKE_CXX_COMPILER=mpicxx \
	-DCMAKE_CXX_FLAGS="-march=native -fopenmp -O3"
$ cmake --build build
```

PETSc-based solvers are optional and disabled by default.  
By default, configure/build will not search for PETSc.  
Enable PETSc explicitly only when needed:  
```
$ cmake -S . -B build -DPICNIX_ENABLE_PETSC=ON
```

Note that this is the so-called out-of-source build, which produces compiled binaries in the `build` directory (in this particular case).
Therefore, you will find executable files `main.out` in, e.g., `build/pic/example/beam`.

See [here](https://github.com/amanotk/pic-nix/wiki/BuildingCode) for details about build configuration.
Please also refer to [CMake Reference Documentation](https://cmake.org/cmake/help/latest/).

## Tests
Tests use Catch2 v3. If Catch2 is installed, it will be used; otherwise CMake will download it during configure.  
To point CMake at a local Catch2 install, set `-DPICNIX_CATCH2_CONFIG=/path/to/Catch2Config.cmake`.  

## Run
You can now execute `main.out` using `mpiexec` (or `mpirun`).  
For example, you can run a simulation with default setup in `pic/example/beam/twostream`.
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

```sh
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install the package
uv pip install -e ./python
```

You can also install from other locations:

```sh
# Pointing to a local clone
uv pip install -e /path/to/pic-nix/python

# Without cloning (install directly from git)
uv pip install "git+https://github.com/amanotk/pic-nix.git#subdirectory=python"
```

After installation, `import picnix` works from any directory.

### Quick Look
After finishing the simulation, you can run the following command in the same directory:
```
$ uv run python quicklook.py data/profile.msgpack
```
You will now see image files `twostream-XXXXXXXX.png` for each snapshot and `twostream.mp4`, which is a movie file encoded by using `ffmpeg`.
