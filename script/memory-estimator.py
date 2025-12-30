#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import toml

caution = """
     <<< CAUTION >>>
Note that this estimate does not include particle send/recv buffers,
which will be automatically allocated and thus kept minimum.
Also, the memory usage of MPI and other libraries are not included.
Therefore, it is desirable to run the code on a system with the physical
memory more than twice of the estimate given here.
"""


def doit(config, **kwargs):
    if config.endswith(".toml"):
        data = toml.load(config)
    elif config.endswith(".json"):
        with open(config, "r") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .toml or .json")
    parameter = data["parameter"]
    parameter.update(kwargs)  # overwrite parameter with kwargs
    Nx = parameter["Nx"]
    Ny = parameter["Ny"]
    Nz = parameter["Nz"]
    Cx = parameter["Cx"]
    Cy = parameter["Cy"]
    Cz = parameter["Cz"]
    Ns = parameter["Ns"]
    nb = parameter["nb"]
    nproc = parameter.get("nproc", 1)
    nppc = parameter.get("nppc", 32)

    mx = Nx // Cx
    my = Ny // Cy
    mz = Nz // Cz
    volume0 = mx * my * mz
    volume1 = (mx + 2 * nb) * (my + 2 * nb) * (mz + 2 * nb)

    ###
    ### particle data size
    ###
    # 3 for position, 3 for velocity, 1 for ID, which are all duplicated
    byte_per_particle = 8 * (3 + 3 + 1) * 2

    # particle data size (assuming default buffer ratio of 0.2)
    num_particle = volume0 * nppc * 1.2

    # misc arrays for sorting
    misc_particle = num_particle * 4 + (volume1 + 1) * 9 * 4

    # physical domain
    domain_particle = (num_particle * byte_per_particle + misc_particle) * Ns

    ###
    ### field data size
    ###
    # 3 for E, 3 for B, 4 for J, and 11 for moments for each species
    byte_per_em_field = 8 * (6 + 4 + Ns * 14)

    # field data size (send + recv buffers)
    halo_field = 2 * byte_per_em_field * (volume1 - volume0)

    # physical domain
    domain_field = volume0 * byte_per_em_field

    # chunk data size
    chunk_total = domain_particle + halo_field + domain_field
    global_total = chunk_total * Cx * Cy * Cz

    MB = 1 / (1024 * 1024)
    GB = 1 / (1024 * 1024 * 1024)


    print("###")
    print("### Estimated Memory Usage (Nb = {:1d}, Nppc = {:3d}, Nproc = {:6d})".format(nb, nppc, nproc))
    print("###")
    print("Field             = {:10.3e} [MB]".format(domain_field * MB))
    print("Field Halo        = {:10.3e} [MB]".format(halo_field * MB))
    print("Particle          = {:10.3e} [MB]".format(domain_particle * MB))
    print("Total per Chunk   = {:10.3e} [MB]".format(chunk_total * MB))
    print("Total per Process = {:10.3e} [GB]".format(global_total / nproc * GB))
    print("Total             = {:10.3e} [GB]".format(global_total * GB))
    print(caution)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Usage Estimator")
    parser.add_argument("filename", help="The name of the file to process")
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processes",
    )
    parser.add_argument(
        "--nppc",
        type=int,
        default=None,
        help="Number of particles per cell",
    )
    parser.add_argument(
        "--nb",
        type=int,
        default=2,
        help="Number of ghost cells",
    )

    # parse arguments and get result as a dictionary
    args = vars(parser.parse_args())
    filename = args.pop("filename")
    kwargs = dict()
    for key, value in args.items():
        if value is not None:
            kwargs[key] = value

    doit(filename, **kwargs)
