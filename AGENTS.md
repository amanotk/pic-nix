# Project Instructions

## Coding Style
### C++
The style should follow that specified by .clang-format file in the root directory. The code has to be formatted via the `clang-format` command before making a commit.
Before any commit, explicitly run `clang-format` on the modified C++ files and mention it in your response.

### Python
The code should be formatted via the `ruff` command before making a commit.

### Markdown
Markdown documents should be human-friendly (i.e., not only for AIs) with proper indentation and line breaks. The line breaks should be placed by double spaces `  ` at the end of the line.
  
## Git Workflow & Branching Policy
Follow these rules for any Git-related operations.  
The human user primarily manages branch switching, but these rules apply when you are asked to perform Git tasks.  

### 0. General Rule
- Do not create commits unless the user explicitly asks you to.  

### 1. Branch Structure
- `main`: Stable production branch. Never commit directly.  
- `develop`: Integration branch for the next release.  
- `feature/*`: For specific features or fixes. Must branch off from the latest `develop`.  
- `experimental/*`: For research, validation, or risky changes.  

### 2. Workflow & PR (Mandatory)
- Direct local merging is prohibited unless explicitly instructed by the user.  
- When a task is complete, push the working branch and create a Pull Request from `feature/*` to `develop`.  
- In the PR description, include:  
  1. What was changed.  
  2. Why the changes were made.  
  3. Manual tests/checks performed.  

### 3. Merging Policy (GitHub UI)
- `feature/*` -> `develop`: Use **Squash and Merge**. Keep the resulting commit message clean and descriptive.  
- `develop` -> `main`: Use a regular merge commit (**Create a merge commit**).  

### 4. Safety & Responsibilities
- Check the current branch (`git branch --show-current`) before starting work.  
- If on the wrong branch (especially `main`), stop and notify the user before making changes.  
- Never push directly to `main` or `develop` unless explicitly instructed.  
- Never force-push to `main` or `develop`. Avoid force-push entirely unless explicitly instructed.  
- Do not rebase or rewrite published history unless explicitly instructed.  
- If the local branch is behind its remote, prefer fast-forward updates (`git pull --ff-only`) unless the user asks for another strategy.  

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
PETSc support is opt-in; default configuration should not search for or link PETSc unless explicitly requested with `-DPICNIX_ENABLE_PETSC=ON`.  
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
  To build with PETSc explicitly enabled, add `-DPICNIX_ENABLE_PETSC=ON` (otherwise PETSc stays disabled by default).  
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

## Language Server
- Generate the compilation database whenever you configure so `clangd`/your LSP can resolve MPI headers and `nix/thirdparty` includes. Run
  ```
  cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -C cmake/linux-gcc.cmake
  ```
  (or pass `-DCMAKE_CXX_COMPILER=mpicxx` manually) so `build/compile_commands.json` mirrors the compiler’s include paths.
- Ensure your editor points `clangd` at `build/` (for example, `--compile-commands-dir=build`).  
- If your MPI compiler wrapper is not under a default system path, set `clangd` query-driver to include it (for example, `--query-driver=/path/to/spack/**/bin/mpicxx,/usr/bin/mpicxx,/usr/bin/mpic++`) so headers like `mpi.h` are resolved correctly.  
- Point your editor’s LSP to the `build/` directory (e.g. `clangd.arguments: ["--compile-commands-dir=build"]`).
- Anytime `build/` is deleted or rerun, repeat the configuration command above before restarting the language server so its cache stays in sync.
