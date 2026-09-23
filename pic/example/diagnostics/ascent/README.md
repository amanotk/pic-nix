# PIC Ascent Diagnostic Examples

These examples demonstrate the PIC-specific data published through the Ascent
diagnostic. They can be attached to an existing PIC physics problem; they do not
define a simulation by themselves.  

## Configuration

Add one Ascent diagnostic to a PIC configuration. The following options publish
the data used by both examples:  

```toml
[[diagnostic]]
name = "ascent"
interval = 50
actions = "/path/to/actions.yaml"
publish_magnetic_field = true
publish_mass_current = true
publish_raw_fields = true
publish_raw_particles = true
```

The `actions` path is resolved relative to the simulation configuration file.
For example, a configuration in `pic/example/beam/weibel/` can refer to the
rendering example with:  

```toml
actions = "../../diagnostics/ascent/render.yaml"
```

See [`docs/picnix/ascent.md`](../../../../docs/picnix/ascent.md) for build and
runtime requirements and the complete list of publication options.  

## Native Rendering

[`render.yaml`](render.yaml) uses native Ascent pipelines and scenes to render:  

* the magnitude of the cell-centered magnetic field `B`;
* the species-00 mass density `um00_M0`.

It writes cycle-numbered PNG images in the process working directory. Each
render uses a wide canvas and places its color bar in the right margin.  

## Python Inspection

[`inspect.yaml`](inspect.yaml) runs [`inspect.py`](inspect.py) as a Python
extract. Launch the simulation from the repository root so the extract's file
path resolves correctly.  

The extract selects the first local chunk on each MPI rank and demonstrates:  

```python
dataset = Dataset.from_conduit(ascent_data())
chunk = next(dataset.local_chunks())

magnetic = chunk.B.array
mass_density = chunk.um00.M0
raw_field = chunk.uf.array
particles = chunk.particle00.array
```

Each rank prints its sampled domain and local extrema. MPI reductions compute
the minimum and maximum magnetic-field magnitude and species-00 mass density
across the sampled chunks. Rank 0 writes the result to:  

```text
ascent_inspect_<cycle>.json
```

The JSON file also records the sampled domain IDs, array shapes, and active
species-00 particle counts. This example intentionally samples one local chunk
per rank; analyses requiring all domains should iterate over
`dataset.local_chunks()` before performing MPI reductions.  
