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
import analysis


class Run(analysis.Run):
    def __init__(self, profile):
        super().__init__(profile)

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
            me = kwargs["me"]
            mi = kwargs["mi"]
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
        gridspec = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[50, 1])
        axs = [0] * 4
        for i in range(4):
            axs[i] = fig.add_subplot(gridspec[i, 0])

        ## density
        ne = (um[..., 0, 0] + um[..., 1, 0]).mean(axis=(0, 1)) / me
        ni = um[..., 2, 0].mean(axis=(0, 1)) / mi
        plt.sca(axs[0])
        plt.plot(xc, ne, "b-")
        plt.plot(xc, ni, "r-")
        axs[0].set_ylabel(r"$N$")
        if ylim0 is not None:
            axs[0].set_ylim(ylim0)

        ## electric field
        ex = uf[..., 0].mean(axis=(0, 1))
        plt.sca(axs[1])
        plt.plot(xc, ex, "k-")
        axs[1].set_ylabel(r"$E_x$")
        if ylim1 is not None:
            axs[1].set_ylim(ylim1)

        ## electron phase space
        fvx0 = analysis.Histogram2D(up[0][:, 0], up[0][:, 3], binx[0], biny[0])
        fvx1 = analysis.Histogram2D(up[1][:, 0], up[1][:, 3], binx[1], biny[1])
        x0, y0, z0 = fvx0.pcolormesh_args()
        x1, y1, z1 = fvx1.pcolormesh_args()
        Xe = x0
        Ye = y0
        Ze = z0 + z1
        plt.sca(axs[2])
        plt.pcolormesh(Xe, Ye, Ze, shading="nearest")
        axs[2].set_ylabel(r"$v_x$")
        # colorbar
        fmt = mpl.ticker.FormatStrFormatter("%4.0e")
        cax = fig.add_subplot(gridspec[2, 1])
        plt.colorbar(cax=cax, format=fmt, label=r"$f_e(x, v_x)$")

        ## ion phase space
        fvx2 = analysis.Histogram2D(up[2][:, 0], up[2][:, 3], binx[2], biny[2])
        Xi, Yi, Zi = fvx2.pcolormesh_args()
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

        fig.suptitle(r"$t = {:6.2f}$".format(tt))

        return fig


def doit_job(profile, prefix, fps, cleanup):
    run = Run(profile)

    # setup plot
    ebinx = (0, run.Nx, run.Nx + 1)
    ibinx = (0, run.Nx, run.Nx + 1)
    ebiny = (-30, +30, 61)
    ibiny = (-5, +5, 61)
    kwargs = dict(
        ele=dict(binx=ebinx, biny=ebiny),
        ion=dict(binx=ibinx, biny=ibiny),
        xlim=(0, run.Nx * run.delh),
        ylim0=(0.5, 2.5),
        ylim1=(-10, +10),
        me=1.0e0,
        mi=1.0e2,
    )

    # check field and particle snapshot time
    time1 = run.time_field
    time2 = run.time_particle
    if time1.size != time2.size or np.allclose(time1, time2) == False:
        raise ValueError("snapshots of field and particle do not match")

    # for all snapshots
    for step in run.step_particle:
        fig = run.summary(step, **kwargs)
        fig.savefig("{:s}-{:08d}.png".format(prefix, step))
        plt.close(fig)

    # convert to mp4
    analysis.convert_to_mp4("{:s}".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="twostream",
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
        "-c",
        "--cleanup",
        action="store_true",
        default=False,
        help="Cleanup intermediate image files",
    )
    parser.add_argument("profile", nargs=1, help="run profile")

    args = parser.parse_args()
    profile = args.profile[0]
    doit_job(profile, args.prefix, args.fps, args.cleanup)
