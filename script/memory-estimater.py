#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import msgpack


def doit(**kwargs):
    # get parameters from kwargs
    Nc = kwargs.get("chunk", 1)
    Nr = kwargs.get("node", 1)
    Ns = kwargs.get("Ns", 2)
    Nb = kwargs.get("Nb", 2)
    cx = kwargs.get("cx", 8)
    cy = kwargs.get("cy", 8)
    cz = kwargs.get("cz", 8)
    nppc = kwargs.get("nppc", 10)
    cfl = kwargs.get("cfl", 0.5)
    mb = 1 / (1024 * 1024)
    gb = 1 / (1024 * 1024 * 1024)
    particle_byte = 8 * 7
    volume0 = cx * cy * cz
    volume1 = (cx + 2) * (cy + 2) * (cz + 2)
    volume2 = (cx + 2 * Nb) * (cy + 2 * Nb) * (cz + 2 * Nb)
    halo1 = volume1 - volume0
    halo2 = volume2 - volume0

    mpi_halo_particle = halo1 * 2 * Ns * nppc * cfl * particle_byte
    physical_particle = volume0 * 4 * Ns * nppc * particle_byte
    mpi_halo_field = halo2 * 2 * (6 + 4 + Ns * 11) * 8
    physical_field = volume2 * (6 + 4 + Ns * 11) * 8
    total = mpi_halo_particle + physical_particle + mpi_halo_field + physical_field

    print("### Chunk Memory Usage [MB] ###")
    print("Physical Particle = {:10.2f}".format(physical_particle * mb))
    print("MPI Halo Particle = {:10.2f}".format(mpi_halo_particle * mb))
    print("Physical Field    = {:10.2f}".format(physical_field * mb))
    print("MPI Halo Field    = {:10.2f}".format(mpi_halo_field * mb))
    print("Chunk Total       = {:10.2f}".format(total * mb))
    print()
    print("### Global Memory Usage [GB] ###")
    print("Node Total        = {:10.2f}".format(total * Nc / Nr * gb))
    print("Total             = {:10.2f}".format(total * Nc * gb))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memory Usage Estimater")
    parser.add_argument("--chunk", type=int, help="Number of chunks")
    parser.add_argument("--node", type=int, help="Number of nodes")
    parser.add_argument("--cx", type=int, default=8, help="Number of cells for a chunk in x")
    parser.add_argument("--cy", type=int, default=8, help="Number of cells for a chunk in y")
    parser.add_argument("--cz", type=int, default=8, help="Number of cells for a chunk in z")
    parser.add_argument("--nppc", type=int, help="Number of particles per cell")
    parser.add_argument(
        "--Ns", nargs=1, type=int, default=2, required=False, help="Number of species"
    )
    parser.add_argument(
        "--Nb", nargs=1, type=int, default=2, required=False, help="Number of boundary cells"
    )
    parser.add_argument(
        "--cfl", nargs=1, type=float, default=0.5, required=False, help="CFL number"
    )

    # parse arguments and get result as a dictionary
    args = vars(parser.parse_args())
    doit(**args)
