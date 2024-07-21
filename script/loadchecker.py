#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt

# global configuration
plt.rcParams.update({"font.size": 12})

import picnix


def doit_job(profile, prefix):
    run = picnix.Run(profile)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.08,
        right=0.82,
        hspace=0.10,
        wspace=0.10,
    )
    status = picnix.plot_loadbalance(run, axs)

    if status == True:
        fig.savefig(prefix + ".png", dpi=120)
    else:
        print("Error: load data was not found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Balance Checker")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="loadbalance",
        help="Prefix used for output image file",
    )
    parser.add_argument("profile", nargs=1, help="run profile")

    args = parser.parse_args()
    profile = args.profile[0]
    doit_job(profile, args.prefix)
