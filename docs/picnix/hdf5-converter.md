# HDF5 Converter

`picnix-hdf5-convert` converts PIC-NIX `.json`/`.data` diagnostics into HDF5
step files and lightweight VDS index files.

The converter treats the original diagnostic directory as read-only during
conversion.  Original files are removed only by the explicit `remove-original`
command after verification.

## Install

Install `picnix` with MPI support if you plan to run conversion in parallel:

```sh
uv pip install --python .venv -e "./python[mpi]"
```

See [Install](install.md) for other install modes.

## Basic Usage

Convert a run directory:

```sh
picnix-hdf5-convert --input-dir /path/to/run/data
```

Run with MPI for larger runs:

```sh
mpiexec -np 16 picnix-hdf5-convert --input-dir /path/to/run/data
```

On a scheduler-managed machine, use the launcher recommended for that system.
For example, some Slurm environments prefer `srun`, while others work better
with the MPI launcher installed with `mpi4py`.

## Commands

`convert` is the default command:

```sh
picnix-hdf5-convert convert --input-dir /path/to/run/data
```

It converts diagnostics to HDF5 step files and VDS indexes, then runs
verification automatically.  Use `--no-verify` to skip automatic verification.

`verify` checks existing HDF5 output against the original diagnostics:

```sh
picnix-hdf5-convert verify --input-dir /path/to/run/data
```

The default verification level is `fast`, which performs structural HDF5
checks and sampled comparisons against the original data.  Use
`--verify-level full` to scan all original metadata.

`remove-original` deletes original `.json` and `.data` files that are covered
by verified HDF5 output:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data
```

Run `remove-original` in a normal interactive shell, not under MPI or a batch
scheduler.  It asks for confirmation before deleting files.

## Typical Workflow

Convert under MPI or a batch scheduler:

```sh
mpiexec -np 16 picnix-hdf5-convert --input-dir /path/to/run/data
```

Verify separately if needed:

```sh
picnix-hdf5-convert verify --input-dir /path/to/run/data
```

Preview what would be removed:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data --dry-run
```

Remove original files interactively:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data
```

For noninteractive use, pass `--yes`:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data --yes
```

## What remove-original Preserves

`remove-original` deletes only verified original `.json` and `.data` files for
the selected prefixes.

It preserves:

- `hdf5/`
- `profile.msgpack`
- `log.msgpack`
- configuration files
- unrelated files

For POSIX output, it promotes `node000000/history.txt` to `history.txt` in the
input directory when possible.  It does not overwrite a different existing
destination file.  After deletion, it removes empty prefix directories and
empty `nodeXXXXXX` directories.

## Resuming Failed Runs

If a conversion is interrupted after creating partial output, rerun with
`--resume`:

```sh
picnix-hdf5-convert --input-dir /path/to/run/data --resume
```

Use `--overwrite` only when you intentionally want to remove existing HDF5
output and start over.
