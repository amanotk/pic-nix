#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import sys

import matplotlib as mpl
import numpy as np

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

    def summary(self, step):
        data = self.read_at("field", step)
        qm = [particle["qm"] for particle in self.config["parameter"]["particle"]]
        xc = data["xc"]
        yc = data["yc"]
        uf = data["uf"]
        um = data["um"]
        tt = self.get_time_at("field", step)
        xlim = (0, self.Nx * self.delh)
        ylim = (0, self.Ny * self.delh)

        ## figure and axes
        fig = plt.figure(1, figsize=(9.6, 3.6), dpi=120)
        fig.subplots_adjust(
            top=0.80,
            bottom=0.14,
            left=0.075,
            right=0.95,
            hspace=0.05,
            wspace=0.35,
        )
        gridspec = fig.add_gridspec(2, 3, height_ratios=[2, 50], width_ratios=[1, 1, 1])
        axs = [0] * 3
        cxs = [0] * 3
        for i in range(3):
            axs[i] = fig.add_subplot(gridspec[1, i])
            axs[i].set_aspect("equal")
            cxs[i] = fig.add_subplot(gridspec[0, i])

        # coordinate
        X, Y = np.broadcast_arrays(xc[None, :], yc[:, None])

        ## density
        plt.sca(axs[0])
        ro = um[..., 0].sum(axis=(0, 3)) / um.shape[0] / 2
        plt.pcolormesh(X, Y, ro, shading="nearest")
        # colorbar
        plt.colorbar(cax=cxs[0], orientation="horizontal")
        cxs[0].xaxis.set_ticks_position("top")
        cxs[0].set_title(r"$N$")

        ## magnetic field strength
        plt.sca(axs[1])
        b0 = (uf[..., 3] ** 2 + uf[..., 4] ** 2 + uf[..., 5] ** 2).mean(axis=(0))
        plt.pcolormesh(X, Y, b0, shading="nearest")
        plt.colorbar(cax=cxs[1], orientation="horizontal")
        cxs[1].xaxis.set_ticks_position("top")
        cxs[1].set_title(r"$|B|$")

        ## current density
        Nx = xc.size
        Ny = yc.size
        plt.sca(axs[2])
        jz = np.zeros((Ny, Nx))
        for i in range(len(qm)):
            jz[...] += qm[i] * um[..., i, 3].mean(axis=(0))
        plt.pcolormesh(X, Y, jz, shading="nearest")
        plt.colorbar(cax=cxs[2], orientation="horizontal")
        cxs[2].xaxis.set_ticks_position("top")
        cxs[2].set_title(r"$J_z$")

        ## appearance
        for i in range(3):
            axs[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
            axs[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_xlabel(r"$x / c/\omega_{pe}$")
            axs[i].set_ylabel(r"$y / c/\omega_{pe}$")
            ax_pos = axs[i].get_position()
            cx_pos = cxs[i].get_position()
            cxs[i].set_position([ax_pos.x0, cx_pos.y0, ax_pos.width, cx_pos.height])

        if self.plot_chunk_boundary:
            coord = np.array(self.chunkmap["coord"])
            rank = self.get_chunk_rank(step)
            cdelx = self.delh * self.Nx // self.Cx
            cdely = self.delh * self.Ny // self.Cy
            for i in range(3):
                picnix.plot_chunk_dist2d(axs[i], coord, rank, cdelx, cdely, colors="white")

        fig.suptitle(r"$\omega_{{pe}} t = {:6.2f}$".format(tt), x=0.5, y=0.99)

        return fig


def doit_job(profile, prefix, fps, boundary, cleanup):
    run = Run(profile, boundary)

    # for all snapshots
    for step in run.get_step("field"):
        fig = run.summary(step)
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
        default="weibel",
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
