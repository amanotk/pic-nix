# Ascent In-Situ Diagnostics

Ascent support is optional and disabled by default.  The integration publishes
one Blueprint domain per local `PicChunk`, with compact centered fields and a
versioned raw PIC-NIX subtree for Python analysis.  

## Build

Use the externally installed Ascent package and an MPI compiler wrapper:  

```sh
cmake -S . -B build-ascent \
  -DBUILD_TESTING=ON \
  -DPICNIX_ENABLE_ASCENT=ON \
  -DAscent_DIR=/path/to/ascent-checkout/lib/cmake/ascent \
  -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build-ascent --parallel
```

Ascent builds require CMake 3.23 or newer and an MPI-enabled Ascent package.
Normal builds use `-DPICNIX_ENABLE_ASCENT=OFF` and do not search for Ascent.  

## Configuration

Add one diagnostic entry to the simulation TOML file:  

```toml
[[diagnostic]]
name = "ascent"
interval = 100
actions = "ascent_actions.yaml"

[diagnostic.publish]
raw = true
centered = true
particles = false
```

The actions path is resolved relative to the configuration file.  Raw and
centered publication are enabled by default; particle publication is disabled
by default.  Only one Ascent diagnostic entry is supported.  

## Actions

Ascent actions can run Python extracts or visualization scenes:  

```yaml
- action: "add_extracts"
  extracts:
    statistics:
      type: "python"
      params:
        file: "ascent_extract.py"
- action: "add_scenes"
  scenes:
    rho:
      plots:
        value:
          type: "pseudocolor"
          field: "rho"
      image_name: "picnix_rho"
```

The runtime opens lazily on the first scheduled invocation, publishes and
executes synchronously, and closes before MPI finalization.  

## Python API

Ascent injects `ascent_data()` and `ascent_mpi_comm_id()` into Python extract
scripts.  Pass the Conduit node explicitly to the helper API:  

```python
from mpi4py import MPI

from picnix.insitu import Dataset

comm = MPI.Comm.f2py(ascent_mpi_comm_id())
dataset = Dataset.from_conduit(ascent_data())
for chunk in dataset.local_chunks():
    electric = chunk.centered_field("E").array
    raw_ex = chunk.raw_field("uf").component("Ex")
```

`Dataset.from_ascent(node_or_callable)` is also available when the caller
wants to make the injected node source explicit; it does not import Ascent
globals implicitly.  

`picnix.insitu` reconstructs raw shapes and byte strides, applies active-cell
metadata, exposes centered fields, and reinterprets particle ID bits as
`int64`.  Install the existing `picnix` dependencies into Ascent's Python
environment, or expose `python/src` with `PYTHONPATH`.  The package's heavy
analysis imports are lazy so lightweight extracts do not require `msgpack`.  

## Schema

Each domain contains `picnix/schema_version = 1`.  Raw arrays are under
`picnix/raw/{uf,uj,um,phi}` with values, shapes, byte strides, component names,
and physical locations.  Centered fields under `fields/` exclude ghost cells
and use owned buffers valid through synchronous Ascent execution.  

## Tests

With Ascent enabled, the focused integration checks are:  

```sh
ctest --test-dir build-ascent -L ASCENT --output-on-failure
python -m pytest python/tests
```

The current tests cover two-rank Python extraction, MPI reduction, pointer
refresh after reallocation, Blueprint verification, and deterministic image
creation.  Performance and memory measurements are not part of the automated
gate yet.  
