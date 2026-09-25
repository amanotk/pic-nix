# Diagnostics and I/O

Diagnostics control what PIC-NIX records while a simulation runs. Each
`[[diagnostic]]` entry answers three questions: what to record, when to record
it, and, for some diagnostics, how much data to retain.  

## Scheduling output

Every diagnostic requires `name`. The scheduling keys are shared:  

| Key | Default | Meaning |
| --- | --- | --- |
| `interval` | `1` | Write every this many steps. It must be positive. |
| `begin` | `0` | Do not write before this step. |
| `end` | No practical limit | Do not write after this step. |
| `prefix` | Diagnostic-dependent | Name used for field, particle, load, and tracker datasets. |

For example, this entry writes at steps 1000, 1100, ..., 10000:  

```toml
[[diagnostic]]
  name = "field"
  prefix = "field-lowres"
  interval = 100
  begin = 1000
  end = 10000
  decimate = 2
```

`prefix` allows multiple outputs of the same type, such as frequent low-resolution
fields and less frequent full-resolution fields. It does not rename the fixed
`history.txt` and `resource.msgpack` files, and `pickup_tracker` writes no file.  

## History

`history` is a compact text diagnostic for monitoring a run. It records the
simulation step and time, RMS divergence errors, electric and magnetic energy,
and kinetic energy for each particle species in `history.txt`. It has no
diagnostic-specific parameters.  

```toml
[[diagnostic]]
  name = "history"
  interval = 100
```

Use this diagnostic for conservation checks and quick progress inspection.  

## Fields and moments

`field` writes the six electromagnetic-field components together with 14
particle moments per species. The optional `decimate` parameter reduces the
spatial resolution of both arrays by block averaging. A value of `1` preserves
the full grid; `2` averages blocks of two cells along each active dimension.
Use a positive factor that divides each active local chunk dimension; otherwise
the writer may retain the original dimension.  

```toml
[[diagnostic]]
  name = "field"
  interval = 100
  decimate = 1
```

Full-resolution field output is usually the largest grid-based diagnostic.
Increase `decimate` or `interval` when only large-scale evolution is needed.  

## Particle samples

`particle` randomly samples every species. `fraction` is the fraction retained
on each output step: `0.01` writes approximately one percent, while `1.0` writes
all particles. The default is `0.01`; use a value between `0.0` and `1.0`.  

```toml
[[diagnostic]]
  name = "particle"
  interval = 1000
  fraction = 0.01
```

Particle output can grow much faster than field output. Use `fraction = 1.0`
only for small runs or when every particle is required.  

## Load distribution

`load` records the estimated work associated with each chunk and the MPI rank
that owns it. It has no diagnostic-specific parameters. Use it to inspect the
effect of dynamic load balancing.  

```toml
[[diagnostic]]
  name = "load"
  interval = 1000
```

## Resource use

`resource` records chunk count, estimated memory, resident memory, and load.
The `node` and `rank` parameters independently control the amount of data:  

| Value | Result |
| --- | --- |
| Omitted | Do not include that aggregation level. |
| `"stats"` | Store summary statistics only. |
| `"full"` | Store summary statistics and every node or rank value. |

```toml
[[diagnostic]]
  name = "resource"
  interval = 1000
  node = "stats"
  rank = "full"
```

Results are appended to `resource.msgpack`. Prefer `"stats"` for large jobs;
rank-level `"full"` output grows with the MPI process count.  

## Particle tracking

Tracking has two stages. `pickup_tracker` marks particles that satisfy a
spatial selection, and `tracker` writes the marked particles on later scheduled
steps. Both entries must use the same `species` index.  

`pickup_tracker` accepts:  

| Key | Default | Meaning |
| --- | --- | --- |
| `species` | `0` | Particle species to inspect. |
| `xmin`, `xmax` | Full x range | Inclusive x selection. |
| `ymin`, `ymax` | Full y range | Inclusive y selection. |
| `zmin`, `zmax` | Full z range | Inclusive z selection. |
| `fraction` | `0.0` | Probability of marking a particle inside the spatial selection. |

Because the default fraction is zero, selection requires an explicit positive
value:  

```toml
[[diagnostic]]
  name = "pickup_tracker"
  interval = 1000
  species = 0
  xmin = 0.0
  xmax = 100.0
  fraction = 0.001

[[diagnostic]]
  name = "tracker"
  interval = 100
  species = 0
```

Once marked, a particle remains marked. The `tracker` entry accepts only the
common scheduling keys and `species`.  

## Ascent in-situ processing

The optional `ascent` diagnostic publishes simulation data to an Ascent actions
file instead of writing the normal field or particle format. `actions` is
required and is resolved relative to the simulation configuration file. Boolean
publication switches select electric field, magnetic field, moments, raw fields,
and raw particles.  

See [Ascent In-Situ Diagnostics](picnix/ascent.md) for the complete options and
runtime requirements.  

## Output location and I/O mode

`application.basedir` selects the output root and defaults to the current
directory. `application.iomode` controls how prefixed field, particle, load, and
tracker data is stored:  

| Mode | Layout |
| --- | --- |
| `mpiio` | Shared files below `<basedir>/<prefix>/`. This is the default. |
| `posix` | Per-node files below `<basedir>/nodeXXXXXX/<prefix>/`. |
| `adios` | Numbered BP5 datasets at `<basedir>/<prefix>/0000.bp`; restarts add `0001.bp`, `0002.bp`, and so on. |

Each `<prefix>` directory contains BP5 dataset directories managed by ADIOS2.  
On restart, the PIC-NIX reader combines numbered datasets into one logical time  
series, with a later segment replacing earlier output from its restart step  
onward.  

Every new run writes `<basedir>/profile.msgpack`. It contains the configuration,
process count, chunk mapping, and metadata used by `picnix.Run` to discover the
other outputs. Runtime logging defaults to `<basedir>/log.msgpack` and can be
relocated with `application.log.path` and `application.log.prefix`.  

## ADIOS2 output

ADIOS2 requires a build with `PICNIX_ENABLE_ADIOS2=ON`. Select it at runtime and
optionally pass scalar BP5 parameters directly under `application.adios`:  

```toml
[application]
  basedir = "data"
  iomode = "adios"

  [application.adios]
    AsyncWrite = true
```

The engine is always BP5. Install the optional Python reader with:  

```sh
uv pip install --python .venv -e "./python[adios]"
```

ADIOS2 segments remain open as `.bp.tmp` directories while a run is active.  
Segments are finalized every 100 completed ADIOS2 diagnostic steps by default.  
Override this with `steps_per_segment`, or set it to `0` to disable rotation:  

```toml
[application.adios]
  AsyncWrite = true
  steps_per_segment = 100
```

At a rotation boundary PIC-NIX closes the current engine, renames the temporary  
directory to `.bp`, and opens the next segment only when the next diagnostic  
step is written. On restart, recoverable temporary segments are promoted, while  
invalid `.bp.tmp` directories remain ignored and their indices are not reused.  
Segment indices may contain gaps, and readers use the record from the highest  
segment index when simulation steps overlap. Each segment retains its  
`segment_index` and `restart_step` metadata.

Raw MPI-I/O and POSIX field or particle diagnostics can instead be converted to
HDF5; see [HDF5 Converter](picnix/hdf5-converter.md).  
