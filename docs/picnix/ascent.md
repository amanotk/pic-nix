# Ascent In-Situ Diagnostics

PIC-NIX can publish each local `PicChunk` as an Ascent Blueprint domain.  
Ascent support is optional, is disabled in normal builds, and requires an
MPI-enabled Ascent installation.  

## Build and test

Configure with CMake 3.23 or newer, an MPI compiler wrapper, and the Ascent
superbuild installation prefix:  

```sh
cmake -S . -B build-ascent \
  -DBUILD_TESTING=ON \
  -DPICNIX_ENABLE_ASCENT=ON \
  -DPICNIX_ASCENT_ROOT=/path/to/ascent-prefix \
  -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build-ascent --parallel
```

When Ascent is installed by `scripts/install_ascent.sh`, use the prefix passed
to the script. PIC-NIX automatically looks for Ascent in the upstream superbuild
layout at `<prefix>/ascent-checkout`. Passing the superbuild prefix through
`CMAKE_PREFIX_PATH` also works:  

```sh
cmake -S . -B build-ascent \
  -DBUILD_TESTING=ON \
  -DPICNIX_ENABLE_ASCENT=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ascent-prefix \
  -DCMAKE_CXX_COMPILER=mpicxx
```

Advanced users may still pass `-DAscent_DIR=/path/to/lib/cmake/ascent`
directly.  

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
python -m pytest python/tests/test_ascent.py
```

Use Ascent's Python environment when testing Python extracts from an Ascent
superbuild installation:  

```sh
/path/to/ascent-prefix/python-venv/bin/python -m pytest python/tests/test_ascent.py
```

Runnable rendering and Python-extract examples are available under
[`pic/example/diagnostics/ascent/`](../../pic/example/diagnostics/ascent/README.md).  

## Runtime environment

PIC-NIX executables linked to Ascent may contain embedded runtime search paths,
but batch environments can change library and Python lookup behavior. For job
scripts, set the Ascent superbuild prefix and prepend the relevant runtime
paths explicitly:  

```sh
export ASCENT_ROOT=/path/to/ascent-prefix
export LD_LIBRARY_PATH="$ASCENT_ROOT/ascent-checkout/lib:$ASCENT_ROOT/conduit-v0.9.5/lib:${LD_LIBRARY_PATH:-}"
export PYTHONHOME=/path/to/python-prefix:/path/to/python-exec-prefix
export PYTHONPATH="/path/to/pic-nix/python/src:$ASCENT_ROOT/python-venv/lib/pythonX.Y/site-packages:${PYTHONPATH:-}"
```

Depending on how Ascent was built, `LD_LIBRARY_PATH` may also need the MPI,
VTK-m, RAJA, Umpire, Camp, MFEM, Silo, HDF5, or other dependency library
directories. Site modules often provide these paths; otherwise use the same
dependency prefixes recorded by the Ascent installation.  

Python extracts run inside Ascent's Python support. Their Python environment
must be able to import `picnix`, `conduit`, and `mpi4py` when MPI-aware extracts
are used. Installing `picnix` and `mpi4py` into Ascent's `python-venv` is usually
cleaner than maintaining a long `PYTHONPATH`. Embedded Python may also need
`PYTHONHOME` so every MPI rank can find the standard library and
platform-dependent `lib-dynload` directory. For Python virtual environments,
derive these paths from the Python used to build Ascent:  

```sh
$ASCENT_ROOT/python-venv/bin/python - <<'PY'
import sys
print(sys.base_prefix)
print(sys.base_exec_prefix)
PY
```

Use the two printed paths as `PYTHONHOME=<base_prefix>:<base_exec_prefix>`. In
many MPI environments, exporting these variables before launch is enough. If the
launcher or scheduler does not propagate the shell environment, pass them
explicitly; for example, OpenMPI accepts:  

```sh
mpiexec -x PYTHONHOME -x PYTHONPATH -x LD_LIBRARY_PATH -n 16 ./main.out -c config.toml
```

Python-enabled Ascent builds are native-build oriented because the selected
Python executable is used during the Ascent/Conduit build. Cross-build systems
where login and compute nodes use different architectures, such as an x86_64
login node targeting aarch64 compute nodes, may not be able to build usable
Python extension modules with a normal host Python. In that case, build the
C++/MPI Ascent path first and enable Python extracts only with a site-provided
target Python/sysroot/toolchain recipe.  

## Configuration

Add one Ascent diagnostic to the simulation configuration:  

```toml
[[diagnostic]]
name = "ascent"
interval = 100
actions = "ascent_actions.yaml"
publish_electric_field = true
publish_magnetic_field = true
publish_mass_current = true
publish_energy_momentum = false
publish_raw_fields = false
publish_raw_particles = false
```

The publish keys are optional boolean flags. The defaults are:  

| Option | Default | Published data |
| --- | --- | --- |
| `publish_electric_field` | `true` | Collocated three-component `E` field |
| `publish_magnetic_field` | `true` | Collocated three-component `B` field |
| `publish_mass_current` | `true` | Four scalar mass-current moments per species |
| `publish_energy_momentum` | `false` | Ten scalar energy-momentum components per species |
| `publish_raw_fields` | `false` | Zero-copy interleaved `uf` and `uj` arrays, their shape, and domain neighbors |
| `publish_raw_particles` | `false` | Zero-copy active `xu` rows for each nonempty species |

The owned-cell `cell_mesh` is always published because Ascent requires every
domain to contain a valid Blueprint mesh. All six data switches are independent,
so a publication may contain only the mesh and selected custom raw data. Moment
calculation runs only when mass-current or energy-momentum publication is
enabled. The current deposition implementation calculates all 14 moments
together even when only one group is published.  

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
|-- coordsets/cell_coords
|-- topologies/cell_mesh
|-- fields/E/values/{x,y,z}                       # electric field enabled
|-- fields/B/values/{x,y,z}                       # magnetic field enabled
|-- fields/um00_{M0,Mx,My,Mz}/values              # mass current enabled
|-- fields/um00_{Ttt,...,Tzx}/values              # energy momentum enabled
|-- fields/um01_{M0,Mx,My,Mz}/values              # mass current enabled
|-- fields/um01_{Ttt,...,Tzx}/values              # energy momentum enabled
|-- pic/raw/shape                                 # raw fields enabled
|-- pic/raw/uf                                    # raw fields enabled
|-- pic/raw/uj                                    # raw fields enabled
|-- pic/neighbors/domain_ids                      # raw fields enabled
|-- pic/neighbors/neighbor_ranks                  # raw fields enabled
|-- pic/particles/particle00/xu                   # raw particles enabled, if active
|-- pic/particles/particle01/xu                   # raw particles enabled, if active
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
canonical component order `(x, y, z)`. Each particle moment is an independent
scalar field prefixed by its species name. No ghost cells are included.  

The default mass-current fields are:  

```text
umNN_M0
umNN_Mx
umNN_My
umNN_Mz
```

`M0` is the deposited mass density. `Mx`, `My`, and `Mz` are its spatial mass
fluxes. For particle mass $`m_s`$, interpolation weight $`W_p`$, and velocity
$`v_i`$, their definitions are:  

```math
M_0 = \sum_p m_s W_p, \qquad M_i = \sum_p m_s v_i W_p.
```

Optional energy-momentum publication adds:  

```text
umNN_Ttt  umNN_Txx  umNN_Tyy  umNN_Tzz
umNN_Ttx  umNN_Tty  umNN_Ttz
umNN_Txy  umNN_Tyz  umNN_Tzx
```

The `xy`, `yz`, `zx` sequence preserves the cyclic ordering used by the moment
deposition. The spatial block is the complete momentum flux, including bulk
transport and random or thermal pressure; it is not a bulk-subtracted stress
tensor. These names follow the PIC-NIX normalization:  

```math
T_{tt} = \sum_p m_s \gamma_p c W_p, \qquad
T_{ti} = \sum_p m_s u_i W_p, \qquad
T_{ij} = \sum_p m_s \frac{u_i u_j}{\gamma_p} W_p.
```

The centered component buffers are compact, publication-owned copies. The
moment arrays must therefore be current before publication; the diagnostic
calls the PIC moment calculation when `publish_mass_current=true` or
`publish_energy_momentum=true`.  

## Raw storage

Raw-field publication adds `pic/raw/shape`, `pic/raw/uf`, and `pic/raw/uj`.
`shape` is a three-integer `(z, y, x)` spatial shape. `uf` and `uj` are flat
external views of the original interleaved `float64` allocations, so no
component reorder or field copy is performed. They are custom Conduit data, not
Blueprint fields, and must be read from the direct input returned by
`ascent_data()`.  

On active axes, the raw shape includes the complete allocated storage extent,
including boundary cells. On an inactive axis, publication selects the single
plane at `boundary_margin` and reports an extent of one. Consequently, all
dimensions expose a normalized `(z, y, x)` spatial shape:  

```text
1D spatial shape: (1, 1, nx_raw)
2D spatial shape: (1, ny_raw, nx_raw)
3D spatial shape: (nz_raw, ny_raw, nx_raw)
```

The component axis is appended in Python, giving `uf` shape
`(nz_raw, ny_raw, nx_raw, 6)` and `uj` shape
`(nz_raw, ny_raw, nx_raw, 4)`. `uf` uses component order
`(Ex, Ey, Ez, Bx, By, Bz)` and `uj` uses `(rho, Jx, Jy, Jz)`. The flat leaves
therefore contain `product(shape) * 6` and `product(shape) * 4` values,
respectively. These components retain their native PIC staggering. They are
unavailable to generic Ascent mesh filters; use centered fields for
visualization and interpret raw storage with PIC-aware analysis.  

The Python owned view removes `boundary_margin` cells from both ends of every
non-singleton raw axis. Singleton inactive axes are unchanged.  

## Neighbors

With `publish_raw_fields=true`, every domain publishes the paired integer arrays
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

For each species with at least one active particle,
`publish_raw_particles=true` publishes the first `Np_active` rows of the original
`xu` allocation at
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
pic/boundary_margin
pic/config
```

`pic/boundary_margin` is the common non-negative boundary margin; publication
rejects local chunks with different margins. `pic/config` is the complete parsed
simulation configuration, currently serialized by C++ as a JSON string. The
Python reader also accepts an equivalent Conduit object tree and exposes either
representation as a `dict`. Other local domains do not duplicate these values.  

## Python extracts

Ascent provides `ascent_data()` and `ascent_mpi_comm_id()` to Python extract
scripts. Construct the PIC-NIX wrapper explicitly:  

```python
from mpi4py import MPI

from picnix.ascent import Dataset

comm = MPI.Comm.f2py(ascent_mpi_comm_id())
dataset = Dataset.from_conduit(ascent_data())

for chunk in dataset.local_chunks():
    electric = chunk.E.array
    ex = chunk.E.component("x")
    mass_density = chunk.um00.M0
    mass_flux_x = chunk.um00.component("Mx")
```

`Dataset.from_conduit(node)` is the standard constructor. The helper does
not import Ascent globals implicitly. `dataset.domain(domain_id)` finds a
local domain by its `state/domain_id`. Shared values are available as
`dataset.boundary_margin` and `dataset.config`. Dataset construction verifies
that both values have one and the same local owner.  

Centered field accessors are `chunk.E`, `chunk.B`, and
`chunk.centered_field(name)`. Published moments are grouped dynamically under
`chunk.umNN` and `chunk.moment_field(name)`. Raw accessors are `chunk.uf`, `chunk.uj`,
`chunk.raw_field(name)`, `chunk.uf_owned`, and `chunk.uj_owned`; a raw field also
provides `.owned` and `.interior()`. Neighbor arrays are
`chunk.neighbor_domain_ids` and `chunk.neighbor_ranks`. Particle accessors are
`chunk.particleNN` and `chunk.particles.particleNN`. Requesting an unpublished
particle species returns an empty `(0, 7)` `float64` array.  

Python always restores three spatial axes in `(z, y, x)` order:  

```text
E.array, B.array:                 (nz, ny, nx, 3)
umNN.array, mass current only:    (nz, ny, nx, 4)
umNN.array, both moment groups:   (nz, ny, nx, 14)
uf.array:                         (nz_raw, ny_raw, nx_raw, 6)
uj.array:                         (nz_raw, ny_raw, nx_raw, 4)
particleNN.array:                 (Np_active, 7)
neighbor_domain_ids, ranks:       (27,)
```

Inactive spatial axes have extent one. Canonical component order is independent
of Conduit child insertion order. Moment order is `(M0, Mx, My, Mz)` followed,
when enabled, by `(Ttt, Txx, Tyy, Tzz, Ttx, Tty, Ttz, Txy, Tyz, Tzx)`.
`Field.component(name)` restores a normalized NumPy view of one component and
preserves an external source view where Conduit permits it. Moment components
also support attribute access such as `chunk.um00.Ttx`. Grouped centered
`.array` access stacks canonical components and therefore materializes a NumPy
array. Raw-field and particle `.array` reshape directly published storage
without copying.  

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
additional per-domain copies of shared configuration. It does not expose raw
storage to Blueprint mesh filters, preserve custom raw data through arbitrary
mesh-only pipelines, or provide performance and memory benchmarks as an
automated test gate.  
