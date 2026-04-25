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
LINEAR_THEORY_NPZ = "alfven-linear-theory.npz"


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
        wp = parameter["wp"]
        mime = parameter["mime"]
        sigma = parameter["sigma"]
        b0 = cc * np.sqrt(sigma)
        vai = cc * np.sqrt(sigma / mime)
        wci = wp * np.sqrt(sigma) / mime

        tt = self.data["tt"] * wci
        uf = self.data["uf"]
        Nx = self.Nx
        delh = self.delh

        # take Fourier transform for transverse B-field in complex representation
        bt = (uf[..., 4] - 1j * uf[..., 5]).mean(axis=(1, 2)) / b0
        kk = np.fft.fftfreq(Nx, delh / (vai / wci)) * 2 * np.pi
        bk = np.fft.fft(bt, axis=-1) / Nx

        # result of linear dispersion picnix
        disp = np.load(LINEAR_THEORY_NPZ)

        tzero = 10.0
        mode_to_plot = [2, 3, 4, 5, 6]

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
            plt.plot(tt, np.abs(bk[:, +mode]), "b-", label="simulation")
            # plot theoretical growth curve
            izero = tt.searchsorted(tzero)
            bzero = np.abs(bk[izero, +mode])
            plt.plot(tt, bzero * np.exp(gm * (tt - tzero)), "r--", label="theory")
            # appearance
            plt.title(
                r"mode = {:d} $(k v_{{A}}/\Omega_{{ci}}$ = {:5.2f})".format(
                    mode, kk[mode]
                )
            )
            plt.semilogy()
            plt.legend(loc="lower right")
            plt.ylabel(r"$|B_y - i B_z|/B_0$".format(mode))
            plt.ylim(1e-4, 1e0)
            plt.grid()

        plt.sca(axs[-1])
        plt.xlabel(r"$\Omega_{ci} t$")

        return fig

    def energy_history(self):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        wp = parameter["wp"]
        mime = parameter["mime"]
        sigma = parameter["sigma"]
        wci = wp * np.sqrt(sigma) / mime

        tt = self.data["tt"] * wci
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
        plt.xlabel(r"$\Omega_{ci} t$")
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

    def helicity_decomposition(self):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        wp = parameter["wp"]
        mime = parameter["mime"]
        sigma = parameter["sigma"]
        b0 = cc * np.sqrt(sigma)
        vai = cc * np.sqrt(sigma / mime)
        wci = wp * np.sqrt(sigma) / mime

        Nx = self.Nx
        xc = self.xc / (vai / wci)
        tt = self.data["tt"] * wci
        uf = self.data["uf"]

        # show time evolution with helicity decomposition
        by = uf[..., 4].mean(axis=(1, 2)) / b0
        bz = uf[..., 5].mean(axis=(1, 2)) / b0
        bt = by - 1j * bz
        bk = np.fft.fft(bt, axis=-1)
        bp = np.zeros_like(bk)
        bm = np.zeros_like(bk)

        ip = np.arange(0, Nx // 2 + 1, 1)
        im = np.arange(Nx, Nx // 2 - 1, -1)
        im[0] = 0

        # positive helicity
        bp[...] = 0.0
        bp[..., ip] = bk[..., ip]
        bp = np.fft.ifft(bp, axis=-1)
        bpy = +bp.real
        bpz = -bp.imag

        # positive helicity
        bm[...] = 0.0
        bm[..., im] = bk[..., im]
        bm = np.fft.ifft(bm, axis=-1)
        bmy = +bm.real
        bmz = -bm.imag

        # plot
        fig = plt.figure(1, figsize=(8, 4), dpi=120)
        fig.subplots_adjust(
            top=0.85,
            bottom=0.15,
            left=0.10,
            right=0.95,
            hspace=0.10,
            wspace=0.20,
        )
        gridspec = fig.add_gridspec(
            2,
            3,
            height_ratios=[2, 50],
            width_ratios=[1, 1, 1],
            hspace=0.05,
            wspace=0.25,
        )

        axs = [0] * 3
        cxs = [0] * 3
        for i in range(3):
            axs[i] = fig.add_subplot(gridspec[1, i])
            cxs[i] = fig.add_subplot(gridspec[0, i])

        X, T = np.broadcast_arrays(xc[None, :], tt[:, None])

        plt.sca(axs[0])
        plt.pcolormesh(X, T, bz, shading="nearest", clim=[-0.8, +0.8])
        plt.colorbar(cax=cxs[0], orientation="horizontal")
        cxs[0].xaxis.set_ticks_position("top")
        cxs[0].set_title(r"$B_z$ (raw)")

        plt.sca(axs[1])
        plt.pcolormesh(X, T, bpz, shading="nearest", clim=[-0.8, +0.8])
        plt.colorbar(cax=cxs[1], orientation="horizontal")
        cxs[1].xaxis.set_ticks_position("top")
        cxs[1].set_title(r"$B_z^{+}$")

        plt.sca(axs[2])
        plt.pcolormesh(X, T, bmz, shading="nearest", clim=[-0.1, +0.1])
        plt.colorbar(cax=cxs[2], orientation="horizontal")
        cxs[2].xaxis.set_ticks_position("top")
        cxs[2].set_title(r"$B_z^{-}$")

        for ax in axs:
            ax.set_xlabel(r"$x / c/\omega_{pi}$")
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.0))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1.0))
        axs[0].set_ylabel(r"$\Omega_{ci} t$")

        return fig

    def velocity_distribution(self, step, vmin=None, vmax=None):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        wp = parameter["wp"]
        mime = parameter["mime"]
        sigma = parameter["sigma"]
        vai = cc * np.sqrt(sigma / mime)
        wci = wp * np.sqrt(sigma) / mime

        # read and plot velocity distribution function
        tp = self.get_time_at("particle", step) * wci
        particle = self.read_at("particle", step)
        up = [particle["up00"], particle["up01"], particle["up02"]]

        # calculate f(v)
        vx = []
        vy = []
        for i in range(1, 3):
            vx.append(up[i][:, 3] / vai)
            vy.append(up[i][:, 4] / vai)
        vx = np.concatenate(vx)
        vy = np.concatenate(vy)
        hist2d = picnix.Histogram2D(vx, vy, (-2.0, +4.0, 151), (-3.0, +3.0, 151))
        X, Y, Z = hist2d.pcolormesh_args()
        Z = Z / np.sum(Z)
        vmax = 1.0e-2 if vmax is None else vmax
        vmin = vmax * 1.0e-3 if vmin is None else vmin
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        # plot
        cb_ratio = 0.05
        figx = 4 * (1 + 9 * cb_ratio)
        figy = 4 * (1 + 4 * cb_ratio)
        fig = plt.figure(1, figsize=(figx, figy), dpi=120)
        fig.subplots_adjust(
            top=1 - 2 * cb_ratio,
            bottom=2 * cb_ratio,
            left=3 * cb_ratio,
            right=1 - 4 * cb_ratio,
            hspace=cb_ratio,
            wspace=cb_ratio,
        )
        gridspec = fig.add_gridspec(
            1,
            2,
            height_ratios=[1],
            width_ratios=[1, cb_ratio],
        )

        ax = fig.add_subplot(gridspec[0, 0])
        cx = fig.add_subplot(gridspec[0, 1])

        plt.sca(ax)
        plt.pcolormesh(X, Y, Z, norm=norm)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, +4.0)
        ax.set_ylim(-3.0, +3.0)
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$v_x / v_{A}$")
        ax.set_ylabel(r"$v_y / v_{A}$")
        ax.set_title(r"$\Omega_{{ci}} t =$ {:5.2f}".format(tp))

        # colorbar
        plt.colorbar(cax=cx, orientation="vertical")
        cx.set_ylabel(r"$f(v)$")

        ax_pos = ax.get_position()
        cx_pos = cx.get_position()
        cx.set_position([cx_pos.x1, ax_pos.y0, cx_pos.width, ax_pos.height])

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

    # time evolution with helicity decomposition
    fig = run.helicity_decomposition()
    fig.savefig("{:s}-helicity.png".format(prefix))
    plt.close(fig)

    # for all snapshots
    for step in run.get_step("particle"):
        fig = run.velocity_distribution(step)
        fig.savefig("{:s}-vdf-{:08d}.png".format(prefix, step))
        plt.close(fig)
    # convert to mp4
    picnix.convert_to_mp4("{:s}-vdf".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="alfven",
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
