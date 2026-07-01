#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import numpy as np

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 12})

import picnix

LINEAR_THEORY_NPZ = "heatflux-linear-theory.npz"


class Run(picnix.Run):
    def __init__(self, profile):
        super().__init__(profile)

        Nstep = len(self.get_step("field"))
        Ns = self.Ns
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        tt = np.zeros((Nstep,), dtype=np.float64)
        uf = np.zeros((Nstep, Nz, Ny, Nx, 6), dtype=np.float64)

        for i, step in enumerate(self.get_step("field")):
            data = self.read_at("field", step)
            tt[i] = self.get_time_at("field", step)
            uf[i, ...] = data["uf"]

        self.data = dict(tt=tt, uf=uf)

    def _lookup_gamma(self, kpara, kperp, disp):
        # npz axis convention is swapped relative to simulation:
        #   disp["kpara"]  holds the perpendicular wavenumber axis
        #   disp["kperp"]  holds the parallel wavenumber axis
        # so we match the simulation kperp against disp["kpara"] (axis 0),
        # and the simulation kpara against disp["kperp"] (axis 1).
        ip = np.argmin(np.abs(disp["kpara"] - kperp))
        ir = np.argmin(np.abs(disp["kperp"] - kpara))
        g = disp["w_imag"][ip, ir]
        if not np.isnan(g):
            return g
        for di in range(-3, 4):
            for dj in range(-3, 4):
                ip2, ir2 = ip + di, ir + dj
                if 0 <= ip2 < disp["w_imag"].shape[0] and 0 <= ir2 < disp["w_imag"].shape[1]:
                    g2 = disp["w_imag"][ip2, ir2]
                    if not np.isnan(g2) and g2 > 0:
                        return g2
        return np.nan

    def linear_theory(self):
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        sigma = parameter["sigma"]
        theta = parameter["theta"]
        phi = parameter["phi"]

        b0 = cc * np.sqrt(sigma)
        tr = theta / 180.0 * np.pi
        pr = phi / 180.0 * np.pi
        b0z = b0 * np.sin(tr) * np.sin(pr)

        tt = self.data["tt"]
        uf = self.data["uf"]
        Nx = self.Nx
        Ny = self.Ny
        delh = self.delh

        dbz = (uf[:, 0, :, :, 5] - b0z) / b0
        bk = np.fft.fft2(dbz, axes=(1, 2)) / (Nx * Ny)

        disp = np.load(LINEAR_THEORY_NPZ)

        modes = [(1, 1), (1, 2), (1, 3)]
        tzero = 100.0

        fig, axs = plt.subplots(len(modes), 1, figsize=(6, 8), sharex=True)
        fig.subplots_adjust(
            top=0.95,
            bottom=0.10,
            left=0.15,
            right=0.95,
            hspace=0.30,
        )

        for ax, (mx, my) in zip(axs, modes):
            amp = np.sqrt(np.abs(bk[:, my, mx]) ** 2 + np.abs(bk[:, Ny - my, mx]) ** 2)

            kx = 2 * np.pi * mx / (Nx * delh)
            ky = 2 * np.pi * my / (Ny * delh)
            gm = self._lookup_gamma(kx, ky, disp)

            izero = tt.searchsorted(tzero)
            azero = amp[izero]

            ax.plot(tt, amp, "b-", label="simulation")
            ax.plot(
                tt,
                azero * np.exp(gm * (tt - tzero)),
                "r--",
                label=r"theory ($\gamma/\omega_{{pe}} = {:6.4f}$)".format(gm),
            )
            ax.set_title(
                r"mode = $({:d}, \pm {:d})$  $k_x\,c/\omega_{{pe}} = {:.3f}$,  $k_y\,c/\omega_{{pe}} = {:.3f}$".format(
                    mx, my, kx, ky
                )
            )
            ax.set_ylabel(r"$|\delta B_z(k_x, k_y)| / B_0$")
            ax.set_xlim(0, tt[-1])
            ax.set_ylim(1e-4, 1e-1)
            ax.semilogy()
            ax.legend(loc="lower right")
            ax.grid()

        axs[-1].set_xlabel(r"$\omega_{pe} t$")

        return fig

    def summary(self, step):
        data = self.read_at("field", step)
        xc = data["xc"]
        yc = data["yc"]
        uf = data["uf"]
        tt = self.get_time_at("field", step)
        xlim = (0, self.Nx * self.delh)
        ylim = (0, self.Ny * self.delh)

        cc = self.config["parameter"]["cc"]
        sigma = self.config["parameter"]["sigma"]
        theta = self.config["parameter"]["theta"]
        phi = self.config["parameter"]["phi"]

        b0 = cc * np.sqrt(sigma)
        tr = theta / 180.0 * np.pi
        pr = phi / 180.0 * np.pi

        b0x = b0 * np.cos(tr)
        b0y = b0 * np.sin(tr) * np.cos(pr)
        b0z = b0 * np.sin(tr) * np.sin(pr)

        X, Y = np.broadcast_arrays(xc[None, :], yc[:, None])

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

        dbx = (uf[..., 3] - b0x).mean(axis=(0)) / b0
        dby = (uf[..., 4] - b0y).mean(axis=(0)) / b0
        dbz = (uf[..., 5] - b0z).mean(axis=(0)) / b0

        plt.sca(axs[0])
        plt.pcolormesh(X, Y, dbx, shading="nearest")
        plt.colorbar(cax=cxs[0], orientation="horizontal")
        cxs[0].xaxis.set_ticks_position("top")
        cxs[0].set_title(r"$\delta B_x / B_0$")

        plt.sca(axs[1])
        plt.pcolormesh(X, Y, dby, shading="nearest")
        plt.colorbar(cax=cxs[1], orientation="horizontal")
        cxs[1].xaxis.set_ticks_position("top")
        cxs[1].set_title(r"$\delta B_y / B_0$")

        plt.sca(axs[2])
        plt.pcolormesh(X, Y, dbz, shading="nearest")
        plt.colorbar(cax=cxs[2], orientation="horizontal")
        cxs[2].xaxis.set_ticks_position("top")
        cxs[2].set_title(r"$\delta B_z / B_0$")

        for i in range(3):
            axs[i].xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
            axs[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_xlabel(r"$x / c/\omega_{pe}$")
            axs[i].set_ylabel(r"$y / c/\omega_{pe}$")
            ax_pos = axs[i].get_position()
            cx_pos = cxs[i].get_position()
            cxs[i].set_position([ax_pos.x0, cx_pos.y0, ax_pos.width, cx_pos.height])

        fig.suptitle(r"$\omega_{{pe}} t = {:6.2f}$".format(tt), x=0.5, y=0.99)

        return fig


def doit_job(profile, prefix, fps, cleanup, linear_only):
    run = Run(profile)

    fig = run.linear_theory()
    fig.savefig("{:s}-linear.png".format(prefix))
    plt.close(fig)

    if linear_only:
        return

    for step in run.get_step("field"):
        fig = run.summary(step)
        fig.savefig("{:s}-{:08d}.png".format(prefix, step))
        plt.close(fig)

    picnix.convert_to_mp4("{:s}".format(prefix), fps, cleanup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="heatflux",
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
    parser.add_argument(
        "-l",
        "--linear-only",
        action="store_true",
        default=False,
        help="Only run linear theory comparison, skip per-snapshot images and movie",
    )
    parser.add_argument("profile", nargs=1, help="run profile")

    args = parser.parse_args()
    profile = args.profile[0]
    doit_job(profile, args.prefix, args.fps, args.cleanup, args.linear_only)
