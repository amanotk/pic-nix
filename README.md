# PIC-NIX
A Particle-In-Cell (PIC) simulation code for collisionless space plasmas.

# Compiling and Executing the Code

## Set Environment Variable
```
$ export PICNIX_DIR=${PWD}/pic-nix
```
Setting the environment variable `PICNIX_DIR` is optional but it will be useful for running diagnostic python scripts.

## Clone
Clone the repository to local working directory via:
```
$ git clone git@github.com:amanotk/pic-nix.git
$ cd pic-nix
$ git submodule update --init
```

## Compile
The code can be compiled with `cmake`, to which a proper C++ compiler and its compiler flags should be specified.  
The following example assumes `mpicxx` as a compiler with OpenMP enabled.
```
$ mkdir build
$ cd build
$ cmake .. \
	-DCMAKE_CXX_COMPILER=mpicxx \
	-DCMAKE_CXX_FLAGS="-O3 -fopenmp"
$ make
```
You will now find executable files `main.out` in `project3d/thermal` and `project3d/beam` directories.

## Run
You can now execute `main.out` using `mpiexec` (or `mpirun`).  
For example, you can run a simulation with default setup in `project3d/beam/twostream`.
```
$ cd project3d/beam/twostream
$ export OMP_NUM_THREADS=2
$ mpiexec -n 8 ../main.out -e 86400 -t 200 -c config.json
```
In this example, you use 8 MPI processes each launch 2 threads.

Some command-line options are:
- `-c` or `--config` : configuration file
- `-t` or `--tmax`   : maximum physical time in simulation unit (default 1.79769e+308)
- `-e` or `--emax`   : maximum elapsed time in sec (default 3600)

## Plot
After finishing the simulation, you can run the following command in the same directory:
```
$ python batch.py profile.msgpack
```
You will now see image files `twosteam-XXXXXXXX.png` for each snapshot and `twostream.mp4`, which is a movie file encoded by using `ffmpeg`.
