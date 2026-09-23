# PIC-NIX

PIC-NIX is a parallel Particle-In-Cell simulation code for collisionless space
plasmas. It combines MPI, OpenMP, and dynamic load balancing through the bundled
[`nix`](https://github.com/amanotk/nix) framework.  

The code provides relativistic electromagnetic PIC simulations, multiple
particle pushers and interpolation schemes, dynamic chunk redistribution, and
parallel diagnostic output. Example applications cover common plasma problems
and provide starting points for new simulations.  

## Requirements

- CMake 3.23 or newer.  
- A C++17 compiler.  
- An MPI implementation and its C++ compiler wrapper.  

Ordinary C++ dependencies can be downloaded automatically during CMake
configuration. Optional integrations such as PETSc, Ascent, and ADIOS2 require
additional packages only when explicitly enabled.  

## Next steps

- [Getting Started](getting-started.md) builds and runs the two-stream example.  
- [Building](build.md) covers dependencies, CMake options, and host configurations.  
- [Configuration](configuration.md) describes simulation input parameters.  
- [Diagnostics and I/O](diagnostics.md) describes output selection and file layouts.  
- [Equations and Units](units.md) defines the electromagnetic unit convention.  
- [Command-Line Tools](cli.md) covers the simulation executable and Python utilities.  

## Documentation versions

Documentation is published for both maintained branches:  

- [Stable documentation](https://amanotk.github.io/pic-nix/) for `main`.  
- [Development documentation](https://amanotk.github.io/pic-nix/develop/) for `develop`.  
