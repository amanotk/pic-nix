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

# result of linear analysis
LINEAR_THEORY_NPZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "buneman-linear-theory.npz"
)


class Run(picnix.Run):
    def __init__(self, profile):
        super().__init__(profile)

        # read and store all data
        Nstep = len(self.get_step("field"))
        Ns = self.Ns
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        tt = np.zeros((Nstep,), dtype=np.float64)
        uf = np.zeros((Nstep, Nz, Ny, Nx, 6), dtype=np.float64)
        um = np.zeros((Nstep, Nz, Ny, Nx, Ns, 14), dtype=np.float64)

        # read
        for i, step in enumerate(self.get_step("field")):
            data = self.read_at("field", step)
            tt[i] = self.get_time_at("field", step)
            uf[i, ...] = data["uf"]
            um[i, ...] = data["um"]

        # store as dict
        self.data = dict(
            tt=tt,
            uf=uf,
            um=um,
        )

    def linear_theory(self):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        qm = 1.0
        wp = parameter["wp"]
        e0 = cc * wp / qm
        tt = self.data["tt"]
        uf = self.data["uf"]
        Nx = self.Nx
        delh = self.delh

        # take Fourier transform for transverse B-field in complex representation
        ex = uf[..., 0].mean(axis=(1, 2)) / e0
        kk = np.fft.fftfreq(Nx, delh) * 2 * np.pi
        ek = np.fft.fft(ex, axis=-1) / Nx

        # result of linear dispersion analysis
        disp = np.load(LINEAR_THEORY_NPZ)

        tzero = 30.0
        mode_to_plot = [14, 15, 16, 17]

        # plot
        fig, axs = plt.subplots(len(mode_to_plot), figsize=(6, 10), sharex=True)
        fig.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.15,
            right=0.95,
            hspace=0.20,
            wspace=0.10,
        )

        for imode, mode in enumerate(mode_to_plot):
            plt.sca(axs[imode])
            # plot simulation result
            ik = disp["k"].searchsorted(kk[mode])
            gm = disp["wim"][ik]
            plt.plot(tt, np.abs(ek[:, +mode]), "b-", label="simulation")
            # plot theoretical growth curve
            izero = tt.searchsorted(tzero)
            ezero = np.abs(ek[izero, +mode])
            plt.plot(tt, ezero * np.exp(gm * (tt - tzero)), "r--", label="theory")
            # appearance
            plt.title(
                r"mode = {:d} $(k c/\omega_{{pe}}$ = {:5.2f})".format(mode, kk[mode])
            )
            plt.semilogy()
            plt.legend(loc="lower right")
            plt.ylabel(r"$|E_x|$".format(mode))
            plt.xlim(0, tt[-1])
            plt.ylim(1e-4, 1e-1)
            plt.grid()

        plt.sca(axs[-1])
        plt.xlabel(r"$\omega_{pe} t$")

        return fig

    def energy_history(self):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        tt = self.data["tt"]
        uf = self.data["uf"]
        um = self.data["um"]

        # show energy history
        Wf = 0.5 * np.sum(uf**2, axis=(1, 2, 3, 4))
        We = np.sum(um[..., 0, 4] * cc - um[..., 0, 0] * cc**2, axis=(1, 2, 3))
        Wi = np.sum(um[..., 1, 4] * cc - um[..., 1, 0] * cc**2, axis=(1, 2, 3))
        Wr = np.sum(um[..., 2, 4] * cc - um[..., 2, 0] * cc**2, axis=(1, 2, 3))
        Wtot = Wf[0] + We[0] + Wi[0] + Wr[0]

        fig, axs = plt.subplots(2, figsize=(8, 6), sharex=True)
        fig.subplots_adjust(
            top=0.95,
            bottom=0.15,
            left=0.15,
            right=0.75,
            hspace=0.10,
            wspace=0.10,
        )

        plt.sca(axs[0])
        plt.plot(tt, Wf / Wtot, label="field")
        plt.plot(tt, We / Wtot, label="electron")
        plt.plot(tt, Wi / Wtot, label="incoming ion")
        plt.plot(tt, Wr / Wtot, label="reflected ion")
        plt.ylabel("Energy")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.grid()

        plt.sca(axs[1])
        err = 100 * ((Wf + We + Wi + Wr) / Wtot - 1)
        mag = int(np.log10(np.abs(err).max())) - 1
        fmt = "10^{" + "{:+d}".format(mag) + "}"
        err /= 10**mag
        plt.plot(tt, err, "k-")
        plt.xlabel(r"$\omega_{pe} t$")
        plt.ylabel("Energy Conservation Error [%]")
        plt.xlim(tt[0], tt[-1])
        plt.grid()
        # format y axis
        formatter = mpl.ticker.FuncFormatter(
            lambda x, pos: r"{:+2.0f} $\times {:s}$".format(x, fmt)
        )
        axs[1].yaxis.set_major_formatter(formatter)

        for ax in axs:
            ax.set_axisbelow(True)
        fig.align_ylabels(axs)

        return fig

    def summary(self, step, **kwargs):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        wp = parameter["wp"]
        mime = parameter["mime"]
        xc = self.xc
        qm = 1.0
        n0 = wp**2 / qm**2
        e0 = cc * wp / qm

        # find index for the same time
        tt = self.get_time_at("particle", step)
        index_p = np.argmin(np.abs(self.get_time("particle") - tt))
        index_f = np.argmin(np.abs(self.get_time("field") - tt))
        step_p = self.get_step("particle")[index_p]
        step_f = self.get_step("field")[index_f]

        # read field
        data = self.read_at("field", step_f)
        uf = data["uf"]
        um = data["um"]

        # read particle
        particle = self.read_at("particle", step_p)
        up = [particle["up00"], particle["up01"], particle["up02"]]

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
        ne = um[..., 0, 0].mean(axis=(0, 1)) / n0
        ni = (um[..., 1, 0] + um[..., 2, 0]).mean(axis=(0, 1)) / (mime * n0)
        plt.sca(axs[0])
        plt.plot(xc, ne, "b-")
        plt.plot(xc, ni, "r-")
        axs[0].set_ylabel(r"$N$")
        if ylim0 is not None:
            axs[0].set_ylim(ylim0)

        ## electric field
        ex = uf[..., 0].mean(axis=(0, 1)) / e0
        plt.sca(axs[1])
        plt.plot(xc, ex, "k-")
        axs[1].set_ylabel(r"$E_x$")
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
        fvx1 = picnix.Histogram2D(up[1][:, 0], up[1][:, 3], binx[2], biny[2])
        fvx2 = picnix.Histogram2D(up[2][:, 0], up[2][:, 3], binx[2], biny[2])
        x1, y1, z1 = fvx1.pcolormesh_args()
        x2, y2, z2 = fvx2.pcolormesh_args()
        Xi = x1
        Yi = y1
        Zi = z1 + z2
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

    # comparison with linear theory
    fig = run.linear_theory()
    fig.savefig("{:s}-linear.png".format(prefix))
    plt.close(fig)

    # energy history
    fig = run.energy_history()
    fig.savefig("{:s}-energy.png".format(prefix))
    plt.close(fig)

    # for all snapshots
    ebinx = (0, run.delh * run.Nx, run.Nx + 1)
    ibinx = (0, run.delh * run.Nx, run.Nx + 1)
    ebiny = (-1.0, +1.0, 101)
    ibiny = (-0.5, +0.5, 201)
    kwargs = dict(
        ele=dict(binx=ebinx, biny=ebiny),
        ion=dict(binx=ibinx, biny=ibiny),
        xlim=(0, run.Nx * run.delh),
        ylim0=(0.5, 1.5),
        ylim1=(-1.2e-1, +1.2e-1),
    )
    for step in run.get_step("particle"):
        fig = run.summary(step, **kwargs)
        fig.savefig("{:s}-summary-{:08d}.png".format(prefix, step))
        plt.close(fig)
    # convert to mp4
    picnix.convert_to_mp4("{:s}-summary".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="buneman",
        help="Prefix used for output image and movie files",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=2,
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
