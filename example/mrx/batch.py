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

    def calc_reconnected_flux(self, step):
        parameter = self.config["parameter"]
        mime = parameter["mime"]
        delh = parameter["delh"] / np.sqrt(mime)
        b0 = np.sqrt(parameter["sigma"])

        data = self.read_at("field", step)
        yc = data["yc"]
        uf = data["uf"]
        um = data["um"]
        iy = yc.size // 2

        return 0.5 * np.sum(np.abs(uf[:, iy, :, 4]) * delh) / (b0 * self.Nz)

    def plot_reconnected_flux(self, time, flux, filename):
        fig = plt.figure(1, figsize=(8, 6), dpi=120)
        plt.plot(time, flux, "k-")
        plt.xlim(0, np.max(time))
        plt.xlabel(r"$\Omega_{ci} t$")
        plt.ylabel(r"$\Psi_{rec}$")
        plt.savefig(filename)
        plt.close(fig)

    def summary2d(self, step):
        ## normalization
        parameter = self.config["parameter"]
        cc = 1.0
        me = 1.0
        mi = me * parameter["mime"]
        wpe = 1.0
        wce = wpe * np.sqrt(parameter["sigma"])
        wpi = wpe / np.sqrt(mi)
        wci = wce / mi
        vae = wce / wpe
        vai = wci / wpi
        b0 = wce / wpe
        ncs = parameter["ncs"]
        nbg = parameter["nbg"]
        T = 1 / wci
        L = cc / wpi

        data = self.read_at("field", step)
        xc = data["xc"] / L
        yc = data["yc"] / L
        uf = data["uf"]
        um = data["um"]
        tt = self.get_time_at("field", step) / T
        xlim = (0, self.Nx * self.delh / L)
        ylim = (0, self.Ny * self.delh / L)

        ## figure and axes
        fig = plt.figure(1, figsize=(8, 6), dpi=120)
        fig.subplots_adjust(
            top=0.90,
            bottom=0.10,
            left=0.10,
            right=0.90,
            hspace=0.10,
            wspace=0.25,
        )
        gridspec = fig.add_gridspec(
            3, 5, height_ratios=[50, 5, 50], width_ratios=[50, 2, 10, 50, 2]
        )
        axs = [0] * 4
        cxs = [0] * 4
        # main axes
        axs[0] = fig.add_subplot(gridspec[0, 0])
        axs[1] = fig.add_subplot(gridspec[0, 3])
        axs[2] = fig.add_subplot(gridspec[2, 0])
        axs[3] = fig.add_subplot(gridspec[2, 3])
        # colorbar axes
        cxs[0] = fig.add_subplot(gridspec[0, 1])
        cxs[1] = fig.add_subplot(gridspec[0, 4])
        cxs[2] = fig.add_subplot(gridspec[2, 1])
        cxs[3] = fig.add_subplot(gridspec[2, 4])

        # coordinate
        X, Y = np.broadcast_arrays(xc[None, :], yc[:, None])

        # plot
        roe = um[..., 0, 0].mean(axis=(0,))
        roi = um[..., 1, 0].mean(axis=(0,))
        vex = um[..., 0, 1].mean(axis=(0,)) / (roe + 1e-32)
        vix = um[..., 1, 1].mean(axis=(0,)) / (roi + 1e-32)
        bz = uf[..., 5].mean(axis=(0,))

        # normalization and smoothiing
        from scipy import signal

        win = np.hanning(3)[:, None] * np.hanning(3)[None, :]
        roe = signal.convolve2d(roe, win, mode="same", boundary="wrap") / me
        roi = signal.convolve2d(roi, win, mode="same", boundary="wrap") / mi
        vex = signal.convolve2d(vex, win, mode="same", boundary="wrap") / vai
        vix = signal.convolve2d(vix, win, mode="same", boundary="wrap") / vai
        bz = signal.convolve2d(bz, win, mode="same", boundary="wrap") / b0

        data = [roe, bz, vex, vix]
        name = [r"$n_e$", r"$B_z/B_0$", r"$V_{e,x}/V_{A,i}$", r"$V_{i,x}/V_{A,i}$"]
        vmin = [-0.9 * np.max(np.abs(v[+10:-10, :])) for v in data]
        vmax = [+0.9 * np.max(np.abs(v[+10:-10, :])) for v in data]
        vmin[0] = 0.0

        for i in range(4):
            plt.sca(axs[i])
            axs[i].set_aspect("equal")
            # data
            plt.pcolormesh(X, Y, data[i], shading="nearest", vmin=vmin[i], vmax=vmax[i])
            # colorbar
            plt.colorbar(cax=cxs[i], orientation="vertical")
            cxs[i].xaxis.set_ticks_position("top")
            cxs[i].set_title(name[i])
            # appearance
            axs[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            axs[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            ax_pos = axs[i].get_position()
            cx_pos = cxs[i].get_position()
            cxs[i].set_position([cx_pos.x0, ax_pos.y0, cx_pos.width, ax_pos.height])
        axs[0].set_ylabel(r"$y / c/\omega_{pi}$")
        axs[2].set_ylabel(r"$y / c/\omega_{pi}$")
        axs[2].set_xlabel(r"$x / c/\omega_{pi}$")
        axs[3].set_xlabel(r"$x / c/\omega_{pi}$")

        # chunk distribution
        if self.plot_chunk_boundary:
            coord = np.array(self.chunkmap["coord"])
            rank = self.get_chunk_rank(step)
            cdelx = self.delh * self.Nx / self.Cx / L
            cdely = self.delh * self.Ny / self.Cy / L
            picnix.plot_chunk_dist2d(axs[0], coord, rank, cdelx, cdely, colors="white", width=0.5)

        # title
        fig.suptitle(r"$\Omega_{{ci}} t = {:6.2f}$".format(tt), x=0.5, y=0.99)

        return fig


def doit_job(profile, prefix, fps, cleanup):
    run = Run(profile)

    mime = run.config["parameter"]["mime"]
    wce = np.sqrt(run.config["parameter"]["sigma"])
    time = run.get_time("field") * wce / mime
    flux = np.zeros((len(time),))

    # for all snapshots
    for i, step in enumerate(run.get_step("field")):
        flux[i] = run.calc_reconnected_flux(step)
        fig = run.summary2d(step)
        fig.savefig("{:s}-{:08d}.png".format(prefix, step))
        plt.close(fig)

    # plot reconnected flux
    run.plot_reconnected_flux(time, flux, "recflux.png")

    # convert to mp4
    picnix.convert_to_mp4("{:s}".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summary Plot Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="mrx",
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
