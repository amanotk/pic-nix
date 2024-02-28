#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def doit(config, **kwargs):
    with open(config, "r") as f:
        data = json.load(f)
    parameter = data["parameter"]
    parameter.update(kwargs)  # overwrite parameter with kwargs
    Nx = parameter["Nx"]
    Ny = parameter["Ny"]
    Nz = parameter["Nz"]
    Cx = parameter["Cx"]
    Cy = parameter["Cy"]
    Cz = parameter["Cz"]
    Ns = parameter["Ns"]
    Nb = parameter["nb"]
    Mp = parameter["nppc"]
    Nc = Cx * Cy * Cz

    byte_per_em_field = 8 * (6 + 4 + Ns * 11)
    byte_per_particle = 8 * 7
    particle_duplicate = 2

    mx = Nx // Cx
    my = Ny // Cy
    mz = Nz // Cz
    volume0 = mx * my * mz
    volume1 = (mx + 2) * (my + 2) * (mz + 2)
    volume2 = (mx + 2 * Nb) * (my + 2 * Nb) * (mz + 2 * Nb)
    halo1 = volume1 - volume0
    halo2 = volume2 - volume0

    # particle data size
    num_particle = volume0 * Mp * 2
    misc_particle = num_particle * 4 + (volume2 + 1) * 9 * 4
    domain_particle = (
        num_particle * byte_per_particle * particle_duplicate + misc_particle
    ) * Ns
    halo_particle = halo1 * 4 * Ns * Mp * byte_per_particle

    # field data size
    halo_field = halo2 * 2 * byte_per_em_field
    domain_field = volume2 * byte_per_em_field

    # chunk data size
    chunk_total = halo_particle + domain_particle + halo_field + domain_field
    global_total = chunk_total * Nc

    MB = 1 / (1024 * 1024)
    GB = 1 / (1024 * 1024 * 1024)

    print("### Estimated Memory Usage (Nb = {:1d}, Nppc = {:3d}) ###".format(Nb, Mp))
    print("Particle          = {:10.3e} [MB]".format(domain_particle * MB))
    print("Particle Halo     = {:10.3e} [MB]".format(halo_particle * MB))
    print("Field             = {:10.3e} [MB]".format(domain_field * MB))
    print("Field Halo        = {:10.3e} [MB]".format(halo_field * MB))
    print("Total per Chunk   = {:10.3e} [MB]".format(chunk_total * MB))

    if "nproc" in kwargs:
        nproc = parameter["nproc"]
        print("Total per Process = {:10.3e} [GB]".format(global_total / nproc * GB))
    print("Total             = {:10.3e} [GB]".format(global_total * GB))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Usage Estimater")
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
        default=10,
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
