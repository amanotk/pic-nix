# Project Instructions

## Testing
Tests are in the `unittest` directory of `nix` and `elliptic` modules.  
See the instructions below for building and running tests.

- Configure  
  Configure with MPI compiler and enable tests as follows:
  ```
  cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=mpicxx
  ```
  Of course, a different compiler can be specified.
- Build  
  After configuration, build with:
  ```
  cmake --build build
  ```
  For a clean build, you can add the `--clean-first` option.
  ```
  cmake --build build --clean-first
  ```
- Test  
  After building, run the tests with:
  ```
  ctest --test-dir build --output-on-failure
  ```
  For a focused test run, use the `-R` option followed by the test name pattern.
