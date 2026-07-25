# Project Instructions

## Coding Style
### C++
The style should follow that specified by .clang-format file in the root directory. The code has to be formatted via the `clang-format` command before making a commit.
Before any commit, explicitly run `clang-format` on the modified C++ files and mention it in your response.

### Python
The code should be formatted via the `ruff` command before making a commit.

### Markdown
Markdown documents should be human-friendly (i.e., not only for AIs) with proper indentation and line breaks. The line breaks should be placed by double spaces `  ` at the end of the line.

Math in docstrings and markdown files should use GitHub-compatible syntax:
- **Inline math**: Use `` $`...`$ `` (dollar-backtick) when the expression contains underscores, backticks, or other markdown-sensitive characters; use plain `$...$` for simple expressions like `$x = 1$`.
- **Display math**: Use a fenced code block with the `math` language tag:

  ````markdown
  ```math
  \int f(\boldsymbol{v}) \, d\boldsymbol{v}
  ```
  ````

  Avoid RST-style `.. math::` directives and `:math:` roles — they do not render on GitHub.
- When using the backtick form, LaTeX special characters do not need `\`-escaping from markdown (e.g. write `V_{s}` not `V\_{s}`).
  
## Git Workflow & Branching Policy
Follow these rules for any Git-related operations.  
The human user primarily manages branch switching, but these rules apply when you are asked to perform Git tasks.  

### 0. General Rule
- **Never commit or push without explicit approval.** A clear affirmative
  phrase like "yes", "do it", "commit", or "push" is required before acting.
  - "yes" to a commit proposal means commit only, not push.
  - Push requires a separate explicit request (e.g. "push it", "push now").
  - Do not assume push permission from prior approvals in the session.
- Silence, a description of changes, or "looks good" is NOT approval.
- Before committing, show the planned commit message and ask "Ready to
  commit?", then wait for an explicit yes.
- If you create new files that should be committed, ask the user first.
- Follow the commit message style used in this repository.

### 1. Branch Structure
- `main`: Stable production branch. Never commit directly.  
- `develop`: Integration branch for the next release.  
- `feature/*`: For specific features or fixes. Must branch off from the latest `develop`.  
- `experimental/*`: For research, validation, or risky changes.  

### 2. Workflow & PR (Mandatory)
- Direct local merging is prohibited unless explicitly instructed by the user.
- When a task is complete (and after receiving explicit approval to push),
  push the working branch and create a Pull Request from `feature/*` to
  `develop`.
- In the PR description, include:
  1. What was changed.
  2. Why the changes were made.
  3. Manual tests/checks performed.

### 3. Merging Policy (GitHub UI)
- `feature/*` -> `develop`: Use **Squash and Merge**. Keep the resulting commit message clean and descriptive.  
- `develop` -> `main`: Use a regular merge commit (**Create a merge commit**).  

  After a release PR is merged, synchronize `main` back into `develop` with
  a local merge commit and push it directly to `develop`.  This back-merge
  does **not** go through a PR — it is the only routine direct push to
  `develop` and should contain the release merge history only, not new
  feature or fix work.  

### 4. Safety & Responsibilities
- Check the current branch (`git branch --show-current`) before starting work.  
- If on the wrong branch (especially `main`), stop and notify the user before making changes.  
- Never push directly to `main` or `develop` unless explicitly instructed
  (the back-merge from `main` into `develop` after a release is the only
  routine exception — see Merging Policy above).  
- Never force-push to `main` or `develop`. Avoid force-push entirely unless explicitly instructed.  
- Do not rebase or rewrite published history unless explicitly instructed.  
- If the local branch is behind its remote, prefer fast-forward updates (`git pull --ff-only`) unless the user asks for another strategy.  

## Directory Structure
- `nix/` : Module for dynamic load balancing  
- `pic/` : Module for Particle-in-Cell simulation  
  Examples of PIC simulations are under `pic/example/`.  
- `elliptic/` : Module for elliptic PDE solvers  
- `scripts/` : Repository maintenance and developer helper scripts  

## Third-Party Code
The vendored single-header `nix/cmdline.hpp` should not be modified unless explicitly instructed.  

## Testing
Tests are off by default; enable them with `-DBUILD_TESTING=ON`.  
PETSc support is opt-in; default configuration should not search for or
link PETSc unless explicitly requested with `-DPICNIX_ENABLE_PETSC=ON`.  
For full build/test instructions, language server setup, and the
PIC integration workflow, see DEVELOPMENT.md.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

### Tool preference (mandatory)
For any question about the codebase, **graphify tools take priority over grep/glob/file reads** because they return scoped, structured results and require fewer follow-up tool calls.

| Graphify tool (`graphify <cmd>`) | When to use | Example |
|---|---|---|
| `query "<question>"` | Understanding concepts, architecture, functionality | "how does the Maxwell solver work?" |
| `path "<A>" "<B>"` | Finding relationships between components | "how does chunkmap connect to the balancer?" |
| `explain "<concept>"` | Focused explanation of a specific symbol or module | "what is PicChunk?" |
| `update .` | After any code modification | `graphify update .` |

Use `grep`/`glob` for **exact text/pattern tasks** (e.g. "find all callsites of `solve_poisson`", "rename `Foo` to `Bar`", "which files have `#include <petsc.h>`"). After graphify identifies relevant files or symbols, read the scoped source files to verify implementation details, especially for debugging, code review, behavioral questions, or uncommitted changes that may not yet be reflected in the graph. For broad discovery of concepts, architecture, or file relationships, graphify is always preferred over raw source browsing.

### Rules
- For codebase questions, **first** run `graphify query "<question>"` when graphify-out/graph.json exists. These return a scoped subgraph, much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
