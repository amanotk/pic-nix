import json
import pathlib
from collections.abc import Mapping

import numpy as np
from mpi4py import MPI

from picnix.ascent import Dataset


def scalar(node):
    value = node.value() if hasattr(node, "value") else node
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("expected a scalar Conduit value")
        return value.reshape(-1)[0].item()
    return value.item() if isinstance(value, np.generic) else value


def has_path(node, path):
    if hasattr(node, "has_path"):
        return node.has_path(path)
    for name in path.split("/"):
        if not isinstance(node, Mapping) or name not in node:
            return False
        node = node[name]
    return True


comm = MPI.Comm.f2py(ascent_mpi_comm_id())  # noqa: F821
dataset = Dataset.from_conduit(ascent_data())  # noqa: F821
chunk = next(dataset.local_chunks())

magnetic = chunk.B.array
magnetic_magnitude = np.linalg.norm(magnetic, axis=-1)
mass_density = chunk.um00.M0

local_extrema = np.array(
    [
        magnetic_magnitude.min(),
        magnetic_magnitude.max(),
        mass_density.min(),
        mass_density.max(),
    ],
    dtype=np.float64,
)
global_minima = np.empty(2, dtype=np.float64)
global_maxima = np.empty(2, dtype=np.float64)
comm.Allreduce(local_extrema[[0, 2]], global_minima, op=MPI.MIN)
comm.Allreduce(local_extrema[[1, 3]], global_maxima, op=MPI.MAX)

raw_shape = list(chunk.uf.shape) if has_path(chunk.node, "pic/raw/uf") else None
particle_count = (
    int(chunk.particle00.shape[0])
    if has_path(chunk.node, "pic/particles/particle00/xu")
    else 0
)
local_domain = {
    "rank": comm.Get_rank(),
    "domain_id": chunk.domain_id,
    "B_shape": list(magnetic.shape),
    "M0_shape": list(mass_density.shape),
    "uf_shape": raw_shape,
    "particle00_count": particle_count,
}
domains = comm.gather(local_domain, root=0)

cycle = int(scalar(chunk.node["state"]["cycle"]))
print(
    f"rank {comm.Get_rank()}: domain={chunk.domain_id} "
    f"|B|=[{local_extrema[0]:.6g}, {local_extrema[1]:.6g}] "
    f"M0=[{local_extrema[2]:.6g}, {local_extrema[3]:.6g}]"
)

if comm.Get_rank() == 0:
    summary = {
        "cycle": cycle,
        "sampling": "first local chunk on each MPI rank",
        "magnetic_magnitude": {
            "minimum": float(global_minima[0]),
            "maximum": float(global_maxima[0]),
        },
        "species_00_mass_density": {
            "minimum": float(global_minima[1]),
            "maximum": float(global_maxima[1]),
        },
        "domains": domains,
    }
    output = pathlib.Path(f"ascent_inspect_{cycle:06d}.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output}")
