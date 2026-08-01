# picnix Python Package

`picnix` is the Python package for reading, analyzing, and post-processing
PIC-NIX simulation output.

The package lives under `python/` and is installable with `uv`.  After
installation, analysis modules can be imported from Python and several command
line tools are available from the environment's `bin/` directory.

## Start Here

- [Install](install.md): how to install the package for development or from git.
- [Command Line Tools](cli.md): installed `picnix-*` commands.
- [HDF5 Converter](hdf5-converter.md): converting `.json`/`.data` diagnostics to HDF5.
- [syncdir](syncdir.md): experimental directory synchronizer for node-local output.
- [Ohm's Law Solver](ohms-law.md): technical note for the generalized Ohm's law solver.
- [Ascent In-Situ Diagnostics](ascent.md): optional MPI-enabled publication,
  Python extracts, and visualization.

## Documentation Scope

These pages are intentionally lightweight.  They are meant to capture the
current working knowledge in plain Markdown before introducing a full website
or documentation generator.

Keep user-facing Python package notes here.  Keep repository maintenance
workflow in `DEVELOPMENT.md` or `scripts/`.
