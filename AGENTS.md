# Project Instructions

## Coding Style
The coding style of any C++ codes need to follow that specified by .clang-format file in the root directory. The code has to be formatted via the `clang-format` command before making a commit.

## Testing
Tests are in the `unittest` directory of the `nix` and `elliptic` modules.  
When running test of these modules, always run configure/build/test from the module subdirectory, not the root directory.  
See the instructions below for building and running tests.

- Configure  
  From the module subdirectory, configure with MPI compiler and enable tests as follows:
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
