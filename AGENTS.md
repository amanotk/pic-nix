# Project Instructions

## Build/Test: nix module
- From `/home/amano/pic-nix/nix`, configure with MPI compiler and enable tests:
  `cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=mpicxx`
- Build: `cmake --build build`
- Test: `ctest --test-dir build --output-on-failure`
- Clean rebuild: `cmake --build build --clean-first`

## Build/Test: elliptic module
- From `/home/amano/pic-nix/elliptic`, configure with tests:
  `cmake -S . -B build -DBUILD_TESTING=ON`
- Build: `cmake --build build`
- Test: `ctest --test-dir build --output-on-failure`
- Focused run: `ctest -R test_petsc_poisson_np8 -V --test-dir build`
