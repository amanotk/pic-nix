# Installing picnix

The `picnix` Python package is in the `python/` subdirectory of this
repository.  Use `uv` to install it into a virtual environment.

## Editable Install

From the repository root:

```sh
uv venv .venv
uv pip install --python .venv -e ./python
```

After this, `import picnix` works from any directory when using that virtual

## Optional Extras

Install test dependencies:

```sh
uv pip install --python .venv -e "./python[test]"
```

Install optional MPI support for MPI-aware tools such as the HDF5 converter:

```sh
uv pip install --python .venv -e "./python[mpi]"
```

The MPI extra installs `mpi4py`.  On systems with multiple MPI installations,
make sure the Python environment can find the same MPI runtime used by
`mpiexec` or the scheduler launcher.

## Install From Another Clone

Point `uv` at the `python/` subdirectory of any local clone:

```sh
uv pip install --python .venv -e /path/to/pic-nix/python
```

## Install From Git

Install directly from the repository without a local clone:

```sh
uv pip install --python .venv "git+https://github.com/amanotk/pic-nix.git#subdirectory=python"
```

## Installed Commands

The package exposes command line tools through `pyproject.toml`.  See
[Command Line Tools](cli.md) for the current list.
