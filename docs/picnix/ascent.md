# Ascent In-Situ Diagnostics

PIC-NIX can publish each local `PicChunk` as an Ascent Blueprint domain.  
Ascent support is optional, is disabled in normal builds, and requires an
MPI-enabled Ascent installation.  

## Build and test

Configure with CMake 3.23 or newer, an MPI compiler wrapper, and the CMake
package directory from the Ascent installation:  

```sh
cmake -S . -B build-ascent \
  -DBUILD_TESTING=ON \
  -DPICNIX_ENABLE_ASCENT=ON \
  -DAscent_DIR=/path/to/ascent/lib/cmake/ascent \
  -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build-ascent --parallel
```

The Ascent package must provide `ascent::ascent_mpi`. Python extract tests are
enabled when Ascent and Conduit have Python support; rendering tests are enabled
when Ascent has VTK-h support. Without `-DPICNIX_ENABLE_ASCENT=ON`, PIC-NIX does
not search for or link Ascent.  

Run the Ascent-focused C++ and MPI tests with:  

```sh
ctest --test-dir build-ascent -L ASCENT --output-on-failure
```

Run the Python helper tests from the repository root with:  

```sh
python -m pytest python/tests/test_insitu.py
```

## Configuration

Add one Ascent diagnostic to the simulation configuration:  

```toml
[[diagnostic]]
name = "ascent"
interval = 100
actions = "ascent_actions.yaml"

[diagnostic.publish]
centered = true
raw = false
particles = false
```

The publish table and its individual keys are optional. The defaults are:  

| Option | Default | Published data |
| --- | --- | --- |
| `centered` | `true` | Owned-cell mesh, collocated `E` and `B`, and one 14-component moment field per species |
| `raw` | `false` | Storage-index mesh, zero-copy `uf` and `uj` components, and domain neighbors |
| `particles` | `false` | Zero-copy active `xu` rows for each nonempty species |

Particle publication is independent of raw publication, so particles may be
published while `raw=false` as long as the centered mesh remains enabled. At
least one of `centered` or `raw` must be true so every domain is a valid
Blueprint mesh. Moment calculation is performed only when centered publication
is enabled.  

`interval` must be a positive integer when present. `actions` is required and
must name a nonempty, readable file. A relative actions path is resolved against
the simulation configuration directory on every MPI rank. Only one Ascent
diagnostic entry is supported.  

The runtime duplicates `MPI_COMM_WORLD`, opens lazily on the first scheduled
diagnostic, and loads the actions once. Publication and action execution are
synchronous. The runtime closes and frees its communicator before MPI
finalization.  

## Published domains

The top-level Conduit node contains one child named `domain_<domain-id>` for
each local PIC chunk. Every domain contains:  

```text
state/domain_id
state/cycle
state/time
```

The following tree shows every optional group and its exact path:  

```text
domain_<domain-id>/
|-- state/{domain_id,cycle,time}
|-- coordsets/cell_coords                         # centered=true
|-- topologies/cell_mesh                          # centered=true
|-- fields/E/values/{x,y,z}                       # centered=true
|-- fields/B/values/{x,y,z}                       # centered=true
|-- fields/um00/values/{m00,...,m13}              # centered=true
|-- fields/um01/values/{m00,...,m13}              # centered=true
|-- coordsets/raw_storage_coords                  # raw=true
|-- topologies/raw_storage_mesh                   # raw=true
|-- fields/uf/values/{Ex,Ey,Ez,Bx,By,Bz}          # raw=true
|-- fields/uj/values/{rho,Jx,Jy,Jz}               # raw=true
|-- pic/neighbors/domain_ids                      # raw=true
|-- pic/neighbors/neighbor_ranks                  # raw=true
|-- pic/particles/particle00/xu                   # particles=true, if active
|-- pic/particles/particle01/xu                   # particles=true, if active
|-- pic/schema_version                            # metadata owner only
|-- pic/boundary_margin                           # metadata owner only
`-- pic/config                                    # metadata owner only
```

Species numbers use at least two zero-padded digits: `um00`, `um01`,
`particle00`, and `particle01`.  

## Centered mesh and fields

`coordsets/cell_coords` is a uniform coordset covering only the chunk's owned
physical cells. Its origin is the chunk's physical lower bound and its spacing
is the PIC grid spacing. `topologies/cell_mesh` uses that coordset. The
Blueprint topology has the actual simulation dimension: only `i` in 1D, `i`
and `j` in 2D, and `i`, `j`, and `k` in 3D. Coordset dimensions are the owned
cell counts plus one vertex on each active axis.  

All centered fields have `association: element` and `topology: cell_mesh`. `E`
and `B` are collocated at cell centers from the staggered `uf` storage and have
canonical component order `(x, y, z)`. Each `umNN` field copies the owned part
of that species' moment array and has canonical component order
`(m00, m01, ..., m13)`. No ghost cells are included.  

The centered component buffers are compact, publication-owned copies. The
moment arrays must therefore be current before publication; the diagnostic
calls the PIC moment calculation when `centered=true`.  

## Raw storage

Raw publication adds `raw_storage_coords` and `raw_storage_mesh`. This uniform
mesh has origin zero and unit spacing: it describes storage indices, not the
physical locations of staggered quantities. `uf` and `uj` are element-associated
fields on this mesh. Their component leaves are external, strided views into the
original interleaved `float64` arrays, so no component reorder or field copy is
performed.  

On active axes, the raw shape includes the complete allocated storage extent,
including boundary cells. On an inactive axis, publication selects the single
plane at `boundary_margin` and reports an extent of one. Consequently, all
dimensions expose a normalized `(z, y, x)` spatial shape even though the
Blueprint topology omits inactive axes:  

```text
1D spatial shape: (1, 1, nx_raw)
2D spatial shape: (1, ny_raw, nx_raw)
3D spatial shape: (nz_raw, ny_raw, nx_raw)
```

The component axis is appended in Python, giving `uf` shape
`(nz_raw, ny_raw, nx_raw, 6)` and `uj` shape
`(nz_raw, ny_raw, nx_raw, 4)`. The raw mesh is not a collocated physical vector
mesh: the electromagnetic and current-storage components retain their native
PIC staggering. Do not apply generic mesh differentiation, recentering, or
vector filters to these raw fields. Use the centered fields for ordinary Ascent
visualization and interpret raw storage with PIC-aware analysis.  

The Python owned view removes `boundary_margin` cells from both ends of every
non-singleton raw axis. Singleton inactive axes are unchanged.  

## Neighbors

With `raw=true`, every domain publishes the paired integer arrays
`pic/neighbors/domain_ids` and `pic/neighbors/neighbor_ranks`. Both have shape
`(27,)` and use the native PIC-NIX ordering:  

```text
index = 9 * (dz + 1) + 3 * (dy + 1) + (dx + 1)
dx, dy, dz in {-1, 0, 1}
```

Index 13 is the center and identifies the current domain and MPI rank. Missing
neighbors retain the values supplied by PIC-NIX, normally `-1` for a missing
domain ID and `MPI_PROC_NULL` for its rank. Neighbor data are omitted when raw
publication is disabled.  

## Active particles

For each species with at least one active particle, `particles=true` publishes
the first `Np_active` rows of the original `xu` allocation at
`pic/particles/particleNN/xu`. Empty or unallocated species are omitted. The
unused allocated tail is never published.  

Each array has shape `(Np_active, 7)`, `float64` storage, and component order:  

```text
(x, y, z, ux, uy, uz, id)
```

The ID column remains the original `float64` value; the publisher and Python
helper do not reinterpret or convert it. Particle arrays are custom `pic` data,
not Blueprint point meshes or fields. Python extracts that need them must read
the direct published input returned by `ascent_data()`, not a downstream
mesh-only pipeline result.  

## Shared metadata

Shared metadata is always present, regardless of the publish switches, and has
exactly one local owner: the first `PicChunk` in the local publication order.
This is not necessarily the domain with the lowest ID. Only that domain has:  

```text
pic/schema_version
pic/boundary_margin
pic/config
```

`pic/schema_version` is currently integer `1`. `pic/boundary_margin` is the
common non-negative boundary margin; publication rejects local chunks with
different margins. `pic/config` is the complete parsed simulation configuration,
currently serialized by C++ as a JSON string. The Python reader also accepts an
equivalent Conduit object tree and exposes either representation as a `dict`.
Other local domains do not duplicate these values.  

## Python extracts

Ascent provides `ascent_data()` and `ascent_mpi_comm_id()` to Python extract
scripts. Construct the PIC-NIX wrapper explicitly:  

```python
from mpi4py import MPI

from picnix.insitu import Dataset

comm = MPI.Comm.f2py(ascent_mpi_comm_id())
dataset = Dataset.from_conduit(ascent_data())

for chunk in dataset.local_chunks():
    electric = chunk.E.array
    ex = chunk.E.component("x")
    moments = chunk.um00.array
```

`Dataset.from_ascent(node_or_callable)` is an equivalent explicit constructor;
it calls its argument when the argument is callable. The helper does not import
Ascent globals implicitly. `dataset.domain(domain_id)` finds a local domain by
its `state/domain_id`. Shared values are available as `dataset.schema_version`,
`dataset.boundary_margin`, and `dataset.config`. Dataset construction verifies
that all three values have one and the same local owner and rejects unsupported
schema versions.  

Centered accessors are `chunk.E`, `chunk.B`, `chunk.umNN`, and
`chunk.centered_field(name)`. Raw accessors are `chunk.uf`, `chunk.uj`,
`chunk.raw_field(name)`, `chunk.uf_owned`, and `chunk.uj_owned`; a raw field also
provides `.owned` and `.interior()`. Neighbor arrays are
`chunk.neighbor_domain_ids` and `chunk.neighbor_ranks`. Particle accessors are
`chunk.particleNN` and `chunk.particles(species)`. Requesting an unpublished
particle species returns an empty `(0, 7)` `float64` array.  

Python always restores three spatial axes in `(z, y, x)` order:  

```text
E.array, B.array:                 (nz, ny, nx, 3)
umNN.array:                       (nz, ny, nx, 14)
uf.array:                         (nz_raw, ny_raw, nx_raw, 6)
uj.array:                         (nz_raw, ny_raw, nx_raw, 4)
particleNN.array:                 (Np_active, 7)
neighbor_domain_ids, ranks:       (27,)
```

Inactive spatial axes have extent one. Canonical component order is independent
of Conduit child insertion order. `Field.component(name)` restores a normalized
NumPy view of one component and preserves an external source view where Conduit
permits it. `.array` stacks the canonical components along a new final axis and
therefore materializes a NumPy array. Particle `.array` reshapes the directly
published storage.  

Install `picnix`, NumPy, and, for MPI extracts, `mpi4py` in Ascent's Python
`PYTHONPATH`; the Conduit Python module must also be importable.  

## Lifetime requirements

Conduit external views do not own raw field or particle memory. Compact centered
buffers are owned by the publication object, but they are also attached to
Conduit as external data. All source and publication buffers must remain valid
until synchronous `Ascent::execute()` returns.  

Between building the node and completion of execution, do not resize, swap,
reallocate, overwrite, or destroy published `uf`, `uj`, particle, or centered
buffers, and do not migrate their chunks. Complete simulation updates, boundary
exchange, and moment processing first; build and publish the node; execute all
actions; then resume updates and release temporary storage.  

## Non-goals

This interface does not publish `ascent_ghosts`, Blueprint `adjsets` or
`nestsets`, particle coordsets or point topologies, raw moment arrays, or
additional per-domain copies of shared configuration. It does not turn the raw
storage mesh into a physically collocated mesh, preserve custom particle data
through arbitrary mesh-only pipelines, or provide performance and memory
benchmarks as an automated test gate.  
