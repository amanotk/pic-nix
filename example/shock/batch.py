#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pathlib

import numpy as np
import matplotlib as mpl

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt

# global configuration
plt.rcParams.update({"font.size": 12})

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
import picnix


class Run(picnix.Run):
    def __init__(self, profile, boundary=True):
        super().__init__(profile)
        self.plot_chunk_boundary = boundary

    def summary(self, step, **kwargs):
        data = self.read_field_at(step)
        xc = self.xc
        uf = data["uf"]
        um = data["um"]
        up = self.read_particle_at(step)
        tt = self.get_particle_time_at(step)

        binx = [0] * 3
        biny = [0] * 3
        try:
            binx[0] = kwargs["ele"]["binx"]
            binx[1] = kwargs["ele"]["binx"]
            binx[2] = kwargs["ion"]["binx"]
            biny[0] = kwargs["ele"]["biny"]
            biny[1] = kwargs["ele"]["biny"]
            biny[2] = kwargs["ion"]["biny"]
            xlim = kwargs["xlim"]
            ylim0 = kwargs.get("ylim0", None)
            ylim1 = kwargs.get("ylim1", None)
            roe = kwargs["roe"]
            roi = kwargs["roi"]
            b0 = kwargs["b0"]
        except Exception as e:
            print("Inappropriate keyword arguments")
            raise e

        ## figure and axes
        fig = plt.figure(1, figsize=(6.4, 6.4), dpi=120)
        fig.subplots_adjust(
            top=0.95,
            bottom=0.08,
            left=0.12,
            right=0.85,
            hspace=0.25,
            wspace=0.02,
        )
        gridspec = fig.add_gridspec(
            4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[50, 1]
        )
        axs = [0] * 4
        for i in range(4):
            axs[i] = fig.add_subplot(gridspec[i, 0])

        ## density
        ne = um[..., 0, 0].mean(axis=(0, 1)) / roe
        ni = um[..., 1, 0].mean(axis=(0, 1)) / roi
        plt.sca(axs[0])
        plt.plot(xc, ne, "b-", lw=1.0)
        plt.plot(xc, ni, "r-", lw=1.0)
        axs[0].set_ylabel(r"$N$")
        if ylim0 is not None:
            axs[0].set_ylim(ylim0)

        ## magnetic field
        bx = uf[..., 3].mean(axis=(0, 1)) / b0
        by = uf[..., 4].mean(axis=(0, 1)) / b0
        bz = uf[..., 5].mean(axis=(0, 1)) / b0
        plt.sca(axs[1])
        plt.plot(xc, bx, "r-", lw=1.0)
        plt.plot(xc, by, "g-", lw=1.0)
        plt.plot(xc, bz, "b-", lw=1.0)
        axs[1].set_ylabel(r"$B$")
        if ylim1 is not None:
            axs[1].set_ylim(ylim1)

        ## electron phase space
        fvxe = picnix.Histogram2D(up[0][:, 0], up[0][:, 3], binx[0], biny[0])
        Xe, Ye, Ze = fvxe.pcolormesh_args()
        plt.sca(axs[2])
        plt.pcolormesh(Xe, Ye, Ze, shading="nearest")
        axs[2].set_ylabel(r"$v_x$")
        # colorbar
        fmt = mpl.ticker.FormatStrFormatter("%4.0e")
        cax = fig.add_subplot(gridspec[2, 1])
        plt.colorbar(cax=cax, format=fmt, label=r"$f_e(x, v_x)$")

        ## ion phase space
        fvxi = picnix.Histogram2D(up[1][:, 0], up[1][:, 3], binx[2], biny[2])
        Xi, Yi, Zi = fvxi.pcolormesh_args()
        plt.sca(axs[3])
        plt.pcolormesh(Xi, Yi, Zi, shading="nearest")
        axs[3].set_xlabel(r"x")
        axs[3].set_ylabel(r"$v_x$")
        # colorbar
        fmt = mpl.ticker.FormatStrFormatter("%4.0e")
        cax = fig.add_subplot(gridspec[3, 1])
        plt.colorbar(cax=cax, format=fmt, label=r"$f_i(x, v_x)$")

        ## appearance
        for i in range(4):
            axs[i].set_xlim(xlim)
        fig.align_ylabels(axs)

        if self.plot_chunk_boundary:
            coord = np.array(self.chunkmap["coord"])
            rank = self.get_chunk_rank(step)
            cdelx = self.delh * (self.Nx // self.Cx)
            for i in range(4):
                picnix.plot_chunk_dist1d(axs[i], coord, rank, cdelx, colors="gray")

        fig.suptitle(r"$t = {:6.2f}$".format(tt))

        return fig


def doit_job(profile, prefix, fps, boundary, cleanup):
    run = Run(profile, boundary)

    mime = run.config["parameter"]["mime"]
    sigma = run.config["parameter"]["sigma"]
    u0 = run.config["parameter"]["u0"]

    roe = 1.0
    roi = 1.0 * mime
    b0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)

    # setup plot
    ebinx = (0, run.Nx, run.Nx + 1)
    ibinx = (0, run.Nx, run.Nx + 1)
    ebiny = (-1.0, +1.0, 81)
    ibiny = (-0.2, +0.2, 81)
    kwargs = dict(
        ele=dict(binx=ebinx, biny=ebiny),
        ion=dict(binx=ibinx, biny=ibiny),
        xlim=(0, run.Nx * run.delh),
        ylim0=(0.5, 6.5),
        ylim1=(-1, +10),
        roe=roe,
        roi=roi,
        b0=b0,
    )

    # for all snapshots
    for step in run.step_particle:
        fig = run.summary(step, **kwargs)
        fig.savefig("{:s}-{:08d}.png".format(prefix, step))
        plt.close(fig)

    # convert to mp4
    picnix.convert_to_mp4("{:s}".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="shock",
        help="Prefix used for output image and movie files",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=10,
        help="Frame/sec used for encoding movie file",
    )
    parser.add_argument(
        "-b",
        "--boundary",
        action="store_true",
        default=True,
        help="Show chunk boundary",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        default=False,
        help="Cleanup intermediate image files",
    )
    parser.add_argument("profile", nargs=1, help="run profile")

    args = parser.parse_args()
    profile = args.profile[0]
    doit_job(profile, args.prefix, args.fps, args.boundary, args.cleanup)
