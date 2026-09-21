# Command-Line Tools

## Simulation executable

Each example builds a `main.out` executable. Run it through the MPI launcher
appropriate for the host:  

```sh
export OMP_NUM_THREADS=2
mpiexec -n 8 ./main.out -c config.toml -t 200 -e 86400
```

| Option | Meaning |
| --- | --- |
| `-c`, `--config FILE` | Read a TOML or JSON configuration file. The source default is `config.json`; examples pass `config.toml` explicitly. |
| `-l`, `--load PREFIX` | Load a saved checkpoint before running. |
| `-s`, `--save PREFIX` | Save a checkpoint at normal termination. |
| `-t`, `--tmax TIME` | Stop at the given physical simulation time. |
| `-e`, `--emax SECONDS` | Stop after the given wall-clock time; the default is one hour. |
| `-v`, `--verbose LEVEL` | Set the runtime verbosity level. |
| `-?`, `--help` | Print the executable's current options. |

Command-line limits and the configured checkpoint interval can be combined so a
batch job periodically saves restart state and exits before its scheduler time
limit. See [Checkpoints and restart](configuration.md#checkpoints-and-restart).  

## `picnix` Python module

The `picnix` module reads simulation profiles, fields, particles, load-balance
records, and other diagnostics. Install it from the repository root:  

```sh
uv venv .venv
uv pip install --python .venv -e ./python
```

Install `"./python[mpi]"` instead when a tool needs `mpi4py`, or
`"./python[adios]"` to read ADIOS2 output.  

## Python commands

The main user-facing commands are:  

| Command | Purpose |
| --- | --- |
| `picnix-hdf5-convert` | Convert raw field and particle diagnostics to HDF5 and VDS files. |
| `picnix-loadchecker` | Plot load-balance diagnostics from a run profile. |
| `picnix-log-analyze` | Summarize timing and runtime log records. |
| `picnix-memory-estimator` | Estimate memory use from a PIC-NIX configuration. |
| `picnix-msgpack-printer` | Print MessagePack data in a readable form. |

Run any command with `--help` for its current arguments:  

```sh
picnix-memory-estimator --help
```

See [HDF5 Converter](picnix/hdf5-converter.md) for the conversion workflow.  
