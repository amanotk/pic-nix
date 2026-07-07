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
checks and sampled comparisons against the original data.  Fast verification
does not scan or store a complete original-file fingerprint.  Use
`--verify-level full` to scan all original metadata and enable strict cleanup.

`remove-original` deletes original `.json` and `.data` files that are covered
by verified HDF5 output:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data
```

Run `remove-original` in a normal interactive shell, not under MPI or a batch
scheduler.  It asks for confirmation before deleting files.

By default, `remove-original` requires a full verification manifest.  It uses
the exact original file list recorded by `verify --verify-level full` in
`hdf5/manifest.json`, checks the recorded size and modification time for each
listed file, then deletes only those verified files.

For large POSIX runs where metadata scans are expensive, pass
`--trust-manifest` after a passed fast verification.  This is an explicit
low-metadata cleanup mode: it confirms the selected HDF5/VDS outputs exist,
generates deterministic original `.json` and `.data` paths from the manifest,
and unlinks those paths without per-file stat checks.

## Typical Workflow

Convert under MPI or a batch scheduler:

```sh
mpiexec -np 16 picnix-hdf5-convert --input-dir /path/to/run/data
```

Verify separately if needed:

```sh
picnix-hdf5-convert verify --input-dir /path/to/run/data
```

Use full verification if you want strict default removal:

```sh
picnix-hdf5-convert verify --input-dir /path/to/run/data --verify-level full
```

Preview what would be removed:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data --dry-run --trust-manifest
```

Remove original files interactively:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data --trust-manifest
```

For noninteractive use, pass `--yes`:

```sh
picnix-hdf5-convert remove-original --input-dir /path/to/run/data --yes --trust-manifest
```

Omit `--trust-manifest` only after `verify --verify-level full` if you want the
strict pre-delete file-size and modification-time checks.

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
