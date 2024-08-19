# PIC-NIX
A Particle-In-Cell (PIC) simulation code for collisionless space plasmas.  
This is based on the kinetic plasma simulation framework `nix`, which can be found [here](https://github.com/amanotk/nix).

# Compiling and Executing the Code

## Environment Variable
```
$ export PICNIX_DIR=/some/where/pic-nix
```
Setting the environment variable `PICNIX_DIR` is optional, but it will be useful for running diagnostic python scripts.

## Clone
Clone the repository to a local working directory via:
```
$ git clone git@github.com:amanotk/pic-nix.git
```

## Compile
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

Note that this is the so-called out-of-source build, which produces compiled binaries in the `build` directory (in this particular case).
Therefore, you will find executable files `main.out` in, e.g., `build/example/beam`.

See [here](https://github.com/amanotk/pic-nix/wiki/BuildingCode) for details about build configuration.
Please also refer to [CMake Reference Documentation](https://cmake.org/cmake/help/latest/).

## Run
You can now execute `main.out` using `mpiexec` (or `mpirun`).  
For example, you can run a simulation with default setup in `example/beam/twostream`.
```
$ cd build/example/beam/twostream
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
  
## Plot
After finishing the simulation, you can run the following command in the same directory:
```
$ python batch.py data/profile.msgpack
```
You will now see image files `twosteam-XXXXXXXX.png` for each snapshot and `twostream.mp4`, which is a movie file encoded by using `ffmpeg`.
