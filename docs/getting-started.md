# Getting Started

This example assumes a typical Linux system with GCC and an MPI compiler
wrapper. See [Building PIC-NIX](build.md) for other compilers and hosts.  

## Build the example

From the repository root, configure PIC-NIX and build only the `beam` example:  

```sh
cmake -S . -B build -C cmake/linux-gcc.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --target beam --parallel
```

The supplied initial-cache file enables OpenMP and produces a machine-local
optimized build.  

## Install the analysis module

The quick-look script imports the `picnix` Python module. Create and activate a
virtual environment before entering the example directory:  

```sh
uv venv .venv
uv pip install --python .venv -e ./python
. .venv/bin/activate
```

## Run the two-stream simulation

Run with eight MPI processes and two OpenMP threads per process:  

```sh
cd build/pic/example/beam/twostream
export OMP_NUM_THREADS=2
mpiexec -n 8 ../main.out -e 86400 -t 200 -c config.toml
```

The example writes diagnostics below `data/`. The `data/profile.msgpack` file
records the configuration, process layout, and information needed to find the
diagnostic files. See [Diagnostics and I/O](diagnostics.md) before adapting the
run.  

## Inspect the result

Run the included quick-look script after the simulation:  

```sh
python quicklook.py data/profile.msgpack
```

The installed module also provides readers, plotting functions, and analysis
tools. See [Command-Line Tools](cli.md) for the user-facing utilities.  
