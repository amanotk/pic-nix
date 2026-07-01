# Command Line Tools

The `picnix` package installs command line tools into the active Python
environment.  These commands are the supported interface for Python utilities
that used to live under `scripts/`.

## Commands

| Command | Purpose |
|---------|---------|
| `picnix-hdf5-convert` | Convert PIC-NIX diagnostics to HDF5 and VDS files. |
| `picnix-loadchecker` | Plot load balance diagnostics from a run profile. |
| `picnix-memory-estimator` | Estimate memory usage from a PIC-NIX config file. |
| `picnix-msgpack-printer` | Pretty-print MessagePack files as JSON. |
| `picnix-syncdir` | Experimental directory synchronizer for node-local output. |

Run any command with `--help` for the exact options:

```sh
picnix-memory-estimator --help
```

## Running Without Installing Globally

From the `python/` directory, `uv run` can run the package command in the
project environment:

```sh
uv run picnix-memory-estimator --help
```

For regular use, prefer installing the package into a virtual environment as
described in [Install](install.md).

## HDF5 Converter

The HDF5 converter is the most developed command line tool.  See
[HDF5 Converter](hdf5-converter.md) for the recommended workflow.

## Experimental Commands

`picnix-syncdir` exists because node-local output can be useful on large
systems, but it has not seen much production use.  Treat it as experimental
and see [syncdir](syncdir.md) before relying on it.
