# Project Instructions

## Coding Style
### C++
The style should follow that specified by .clang-format file in the root directory. The code has to be formatted via the `clang-format` command before making a commit.

### Python
The code should be formatted via the `ruff` command before making a commit.

### Markdown
Markdown documents should be human-friendly (i.e., not only for AIs) with proper indentation and line breaks. The line breaks should be placed by double spaces `  ` at the end of the line.
  
## Git Workflow
- Do not create commits unless the user explicitly asks you to.  

## Directory Structure
- `nix/` : Module for dynamic load balancing  
- `pic/` : Module for Particle-in-Cell simulation  
  Examples of PIC simulations are under `pic/example/`.  
- `elliptic/` : Module for elliptic PDE solvers  
- `script/` : Utility scripts  

## Third-Party Code
The code under `nix/thirdparty/` should not be modified unless explicitly instructed.  

## Testing
Tests are in the `unittest` directory of the `nix` and `elliptic` modules, but configure/build/test commands should always run from the repository root so they share the top-level build directory.  
Tests are off by default; enable them with `-DBUILD_TESTING=ON`.  
See the instructions below for building and running tests.

- Configure  
  From the repository root, configure with MPI compiler and enable tests as follows:
  ```
  cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=mpicxx
  ```
  Of course, a different compiler can be specified.  
  On typical linux systems with GCC, use `cmake/linux-gcc.cmake` file for better optimization:
  ```
  cmake -S . -B build -DBUILD_TESTING=ON -C cmake/linux-gcc.cmake
  ```
  Configuration for Intel oneAPI compilers is also available via `cmake/linux-intel.cmake`.
- Build  
  After configuration, build with:
  ```
  cmake --build build --parallel
  ```
  For a clean build, you can add the `--clean-first` option.  
  ```
  cmake --build build --clean-first --parallel
  ```
  On some systems including WSL2, you may need to limit the number of parallel jobs for stability:
  ```
  cmake --build build --parallel 4
  ```
- Test  
  After building, run the tests with:
  ```
  ctest --test-dir build --output-on-failure
  ```
  When running MPI tests in a sandboxed environment, use escalated permissions;  
  otherwise PMIx can fail with `socket()` errors.  
  For a focused test run, use the `-R` option followed by the test name pattern.

### PIC integration test
The PIC integration test is `test_pic_application` (MPI/Catch2).  
It runs with `np=1` and `np=8` and creates a temporary base directory under `PICNIX_TMPDIR`,  
which it cleans up inside the test code after completion.  
To run only these tests:
```
ctest --test-dir build -R test_pic_application --output-on-failure
```
