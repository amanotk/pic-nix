# Configuration

PIC-NIX accepts TOML and JSON configuration files. TOML is recommended for
normal use. A configuration has three required top-level sections:
`application`, `diagnostic`, and `parameter`.  

Start from the configuration shipped with an example, such as
[`pic/example/beam/twostream/config.toml`](https://github.com/amanotk/pic-nix/blob/main/pic/example/beam/twostream/config.toml),
rather than writing one from scratch.  

## Basic structure

```toml
[application]
  basedir = "data"
  iomode = "mpiio"

  [application.log]
    interval = 5000

  [application.rebalance]
    interval = 10
    loglevel = 1

  [application.option]
    vectorization = "vector"
    order = 2

[[diagnostic]]
  name = "history"
  interval = 100

[[diagnostic]]
  name = "field"
  interval = 100

[parameter]
  Nx = 512
  Ny = 1
  Nz = 1
  Cx = 64
  Cy = 1
  Cz = 1
  delt = 0.01
  delh = 1.0
  # The remaining parameters depend on the selected example.
```

`Nx`, `Ny`, and `Nz` are the global cell counts. `Cx`, `Cy`, and `Cz` are the
chunk counts, and each cell count must be divisible by its corresponding chunk
count. `delt` is the time step and `delh` is the uniform cell size.  

The remaining `[parameter]` keys define the initial condition and are specific
to each example executable. Consult its `config.toml` and `main.cpp`.  

The `[[diagnostic]]` entries select simulation output. See
[Diagnostics and I/O](diagnostics.md) for scheduling, diagnostic-specific
parameters, and output layouts.  

## Output location and mode

Two keys in `[application]` provide the top-level output configuration:  

| Key | Default | Meaning |
| --- | --- | --- |
| `basedir` | `"."` | Root directory for the run profile, log, diagnostics, and checkpoints. |
| `iomode` | `"mpiio"` | Diagnostic storage backend: `"mpiio"`, `"posix"`, or `"adios"`. |

Most examples set `basedir = "data"`. The I/O mode changes the diagnostic file
layout, while `"adios"` additionally requires an ADIOS2-enabled build. See
[Diagnostics and I/O](diagnostics.md#output-location-and-io-mode) for the
resulting paths and backend details.  

## Application options

`application.option` controls simulation implementation choices. Common keys
are:  

| Key | Typical values | Default |
| --- | --- | --- |
| `vectorization` | `"scalar"`, `"vector"`, or a per-kernel table | `"scalar"` |
| `order` | Shape-function order from 1 through 4 | `2` |
| `pusher` | `"Boris"`, `"Vay"`, `"HigueraCary"` | `"Boris"` |
| `interpolation` | `"MC"`, `"WT"` | `"MC"` |
| `seed_type` | `"random"`, `"fixed"` | `"random"` |
| `friedman` | Friedman-filter coefficient | `0.0` |
| `cell_load` | Relative cell cost for load balancing | `1.0` |

`application.rebalance.interval` controls how often dynamic load balancing is
considered. Its implementation default is 100 steps; examples commonly use a
shorter interval.  

## Checkpoints and restart

Use `application.checkpoint` for periodic wall-clock checkpoints:  

```toml
[application.checkpoint]
  interval = 3600.0
  prefix = "checkpoint"
```

The interval is in seconds. Zero or an omitted section disables periodic
checkpointing. PIC-NIX rotates between `<prefix>.0` and `<prefix>.1`; loading
the logical prefix selects the latest complete slot:  

```sh
mpiexec -n 8 ./main.out -c config.toml -l checkpoint
```

Use `-s PREFIX` to save a checkpoint at normal termination. When periodic
checkpointing is enabled, the final prefix must differ from the periodic prefix
and both rotating slot names.  
