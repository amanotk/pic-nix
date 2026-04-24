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
Tests are off by default; enable them with `-DBUILD_TESTING=ON`.  
PETSc support is opt-in; default configuration should not search for or
link PETSc unless explicitly requested with `-DPICNIX_ENABLE_PETSC=ON`.  
For full build/test instructions, language server setup, and the
PIC integration workflow, see DEVELOPMENT.md.

## graphify

This project keeps a graphify knowledge graph snapshot at `docs/graphify/`.

Rules:
- Before architecture/codebase questions, read `docs/graphify/GRAPH_REPORT.md` if it exists.
- For graph traversal, use `graphify query|path|explain` with `--graph docs/graphify/graph.json`.
