#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import toml


CAUTION = """
     <<< CAUTION >>>
Note that this estimate does not include particle send/recv buffers,
which will be automatically allocated and thus kept minimum.
Also, the memory usage of MPI and other libraries are not included.
Therefore, it is desirable to run the code on a system with the physical
memory more than twice of the estimate given here.
"""


def load_config(filename):
    path = Path(filename)
    if path.suffix == ".toml":
        return toml.load(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    raise ValueError("Unsupported file format. Use .toml or .json")


def estimate(config, **kwargs):
    data = load_config(config)
    parameter = {**data["parameter"], **kwargs}
    nx = parameter["Nx"]
    ny = parameter["Ny"]
    nz = parameter["Nz"]
    cx = parameter["Cx"]
    cy = parameter["Cy"]
    cz = parameter["Cz"]
    ns = parameter["Ns"]
    nb = parameter.get("nb", 2)
    nproc = parameter.get("nproc", 1)
    nppc = parameter.get("nppc", 32)

    mx = nx // cx
    my = ny // cy
    mz = nz // cz
    volume0 = mx * my * mz
    volume1 = (mx + 2 * nb) * (my + 2 * nb) * (mz + 2 * nb)

    # 3 for position, 3 for velocity, 1 for ID, which are all duplicated.
    byte_per_particle = 8 * (3 + 3 + 1) * 2

    # Particle data size, assuming default buffer ratio of 0.2.
    num_particle = volume0 * nppc * 1.2

    # Misc arrays for sorting.
    misc_particle = num_particle * 4 + (volume1 + 1) * 9 * 4

    domain_particle = (num_particle * byte_per_particle + misc_particle) * ns

    # 3 for E, 3 for B, 4 for J, and 14 moments for each species.
    # Implementation-dependent field arrays are deliberately excluded.
    byte_per_em_field = 8 * (6 + 4 + ns * 14)

    halo_field = 2 * byte_per_em_field * (volume1 - volume0)
    domain_field = volume0 * byte_per_em_field

    chunk_total = domain_particle + halo_field + domain_field
    global_total = chunk_total * cx * cy * cz

    mb = 1 / (1024 * 1024)
    gb = 1 / (1024 * 1024 * 1024)

    print("###")
    print(
        "### Estimated Memory Usage "
        f"(Nb = {nb:1d}, Nppc = {nppc:3d}, Nproc = {nproc:6d})"
    )
    print("###")
    print(f"Field             = {domain_field * mb:10.3e} [MB]")
    print(f"Field Halo        = {halo_field * mb:10.3e} [MB]")
    print(f"Particle          = {domain_particle * mb:10.3e} [MB]")
    print(f"Total per Chunk   = {chunk_total * mb:10.3e} [MB]")
    print(f"Total per Process = {global_total / nproc * gb:10.3e} [GB]")
    print(f"Total             = {global_total * gb:10.3e} [GB]")
    print(CAUTION)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Memory Usage Estimator")
    parser.add_argument("filename", help="configuration file to process")
    parser.add_argument("--nproc", type=int, help="number of processes")
    parser.add_argument("--nppc", type=int, help="number of particles per cell")
    parser.add_argument("--nb", type=int, help="number of ghost cells")

    args = vars(parser.parse_args(argv))
    filename = args.pop("filename")
    kwargs = {key: value for key, value in args.items() if value is not None}
    estimate(filename, **kwargs)


if __name__ == "__main__":
    main()
