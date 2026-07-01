#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt

from .run import Run
from .utils import plot_loadbalance

plt.rcParams.update({"font.size": 12})


def run(profile, prefix="loadbalance"):
    simulation = Run(profile)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.08,
        right=0.82,
        hspace=0.10,
        wspace=0.10,
    )
    status = plot_loadbalance(simulation, axs)

    if status:
        fig.savefig(f"{prefix}.png", dpi=120)
    else:
        print("Error: load data was not found")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load Balance Checker")
    parser.add_argument(
        "-p",
        "--prefix",
        default="loadbalance",
        help="prefix used for output image file",
    )
    parser.add_argument("profile", help="run profile")

    args = parser.parse_args(argv)
    run(args.profile, args.prefix)


if __name__ == "__main__":
    main()
