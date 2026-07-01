#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import numpy as np

mpl.use("Agg") if __name__ == "__main__" else None
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"font.size": 12})

import picnix


def field_aligned_basis(theta, phi):
    """Return field-aligned basis vectors e1 (||B), e2, e3 (perp B)."""
    tr = theta / 180.0 * np.pi
    pr = phi / 180.0 * np.pi
    ct, st = np.cos(tr), np.sin(tr)
    cp, sp = np.cos(pr), np.sin(pr)
    e1 = np.array([ct, st * cp, st * sp])
    e2 = np.array([-st, ct * cp, ct * sp])
    e3 = np.array([0.0, -sp, cp])
    return e1, e2, e3


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
        um = np.zeros((Nstep, Nz, Ny, Nx, Ns, 14), dtype=np.float64)

        for i, step in enumerate(self.get_step("field")):
            data = self.read_at("field", step)
            tt[i] = self.get_time_at("field", step)
            uf[i, ...] = data["uf"]
            um[i, ...] = data["um"]

        self.data = dict(tt=tt, uf=uf, um=um)

    def magnetic_energy(self):
        """Plot delta B^2 / B_0^2 vs time (log scale)."""
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        sigma = parameter["sigma"]

        b0_sq = cc * cc * sigma
        ncells = self.Nx * self.Ny * self.Nz

        tt = self.data["tt"]
        uf = self.data["uf"]

        b_sq = np.sum(uf[..., 3] ** 2 + uf[..., 4] ** 2 + uf[..., 5] ** 2, axis=(1, 2, 3))
        db_sq = b_sq / (ncells * b0_sq) - 1.0

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(tt, np.maximum(db_sq, 1e-12), "b-", label=r"$\delta B^2 / B_0^2$")
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$\delta B^2 / B_0^2$")
        ax.set_xlim(tt[0], tt[-1])
        ax.set_ylim(1e-4, 1e-0)
        ax.grid()

        fig.tight_layout()
        return fig

    def anisotropy_history(self):
        """Plot electron T_perp / T_parallel vs time (linear scale)."""
        parameter = self.config["parameter"]
        theta = parameter["theta"]
        phi = parameter["phi"]

        e1, e2, e3 = field_aligned_basis(theta, phi)

        tt = self.data["tt"]
        um = self.data["um"]

        # electron moments (species 0)
        rho = um[..., 0, 0]
        vx = um[..., 0, 1] / rho
        vy = um[..., 0, 2] / rho
        vz = um[..., 0, 3] / rho

        # pressure tensor components (non-relativistic)
        Pxx = um[..., 0, 5] / rho - vx * vx
        Pyy = um[..., 0, 6] / rho - vy * vy
        Pzz = um[..., 0, 7] / rho - vz * vz
        Pxy = um[..., 0, 11] / rho - vx * vy
        Pyz = um[..., 0, 12] / rho - vy * vz
        Pzx = um[..., 0, 13] / rho - vz * vx

        # rotate to field-aligned frame: T_par = e_i^T P e_i
        T_par = (
            e1[0] ** 2 * Pxx
            + e1[1] ** 2 * Pyy
            + e1[2] ** 2 * Pzz
            + 2.0 * e1[0] * e1[1] * Pxy
            + 2.0 * e1[1] * e1[2] * Pyz
            + 2.0 * e1[2] * e1[0] * Pzx
        )
        T_perp1 = (
            e2[0] ** 2 * Pxx
            + e2[1] ** 2 * Pyy
            + e2[2] ** 2 * Pzz
            + 2.0 * e2[0] * e2[1] * Pxy
            + 2.0 * e2[1] * e2[2] * Pyz
            + 2.0 * e2[2] * e2[0] * Pzx
        )
        T_perp2 = (
            e3[0] ** 2 * Pxx
            + e3[1] ** 2 * Pyy
            + e3[2] ** 2 * Pzz
            + 2.0 * e3[0] * e3[1] * Pxy
            + 2.0 * e3[1] * e3[2] * Pyz
            + 2.0 * e3[2] * e3[0] * Pzx
        )
        T_perp = 0.5 * (T_perp1 + T_perp2)

        # spatial average of T_perp / T_par
        ratio = T_perp / T_par
        ratio_mean = np.mean(ratio, axis=(1, 2, 3))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tt, ratio_mean, "b-", label=r"$T_\perp / T_\parallel$ (electrons)")
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$T_\perp / T_\parallel$")
        ax.set_xlim(tt[0], tt[-1])
        ax.set_ylim(1.0, 3.5)
        ax.grid()

        fig.tight_layout()
        return fig

    def helicity_decomposition(self):
        """Plot transverse B field with helicity decomposition (x-t diagram)."""
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        sigma = parameter["sigma"]
        theta = parameter["theta"]
        phi = parameter["phi"]

        b0 = cc * np.sqrt(sigma)
        e1, e2, e3 = field_aligned_basis(theta, phi)

        Nx = self.Nx
        xc = self.xc
        tt = self.data["tt"]
        uf = self.data["uf"]

        # project transverse B onto field-aligned basis (average over y, z)
        bx = uf[..., 3].mean(axis=(1, 2))
        by = uf[..., 4].mean(axis=(1, 2))
        bz = uf[..., 5].mean(axis=(1, 2))
        b_perp1 = (bx * e2[0] + by * e2[1] + bz * e2[2]) / b0
        b_perp2 = (bx * e3[0] + by * e3[1] + bz * e3[2]) / b0

        # complex transverse field and helicity decomposition
        bt = b_perp1 - 1j * b_perp2
        bk = np.fft.fft(bt, axis=-1)

        ip = np.arange(0, Nx // 2 + 1, 1)
        im = np.arange(Nx, Nx // 2 - 1, -1)
        im[0] = 0

        bp = np.zeros_like(bk)
        bp[..., ip] = bk[..., ip]
        bp = np.fft.ifft(bp, axis=-1)
        bp_perp2 = -bp.imag

        bm = np.zeros_like(bk)
        bm[..., im] = bk[..., im]
        bm = np.fft.ifft(bm, axis=-1)
        bm_perp2 = -bm.imag

        # plot
        fig = plt.figure(figsize=(10, 4), dpi=120)
        fig.subplots_adjust(
            top=0.85,
            bottom=0.15,
            left=0.08,
            right=0.95,
            hspace=0.10,
            wspace=0.25,
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

        panels = [
            (b_perp2, r"$B_z$ (raw)", [-0.4, +0.4]),
            (bp_perp2, r"$B_z^{+}$", [-0.4, +0.4]),
            (bm_perp2, r"$B_z^{-}$", [-0.4, +0.4]),
        ]

        for ax, cx, (data, title, clim) in zip(axs, cxs, panels):
            plt.sca(ax)
            plt.pcolormesh(X, T, data, shading="nearest", clim=clim, cmap="viridis")
            plt.colorbar(cax=cx, orientation="horizontal")
            cx.xaxis.set_ticks_position("top")
            cx.set_title(title)

        for ax in axs:
            ax.set_xlabel(r"$x / c/\omega_{pe}$")
        axs[0].set_ylabel(r"$\omega_{pe} t$")

        return fig

    def vdf_difference(self):
        """Plot initial, final, and difference of electron VDF in (v_par, v_perp)."""
        parameter = self.config["parameter"]
        cc = parameter["cc"]
        sigma = parameter["sigma"]
        theta = parameter["theta"]
        phi = parameter["phi"]
        betae_para = parameter["betae_para"]
        betae_perp = parameter["betae_perp"]

        vae = cc * np.sqrt(sigma)
        vte_para = vae * np.sqrt(0.5 * betae_para)
        vte_perp = vae * np.sqrt(0.5 * betae_perp)

        e1, e2, e3 = field_aligned_basis(theta, phi)

        steps = self.get_step("particle")
        step_init = steps[0]
        step_final = steps[-1]

        data_init = self.read_at("particle", step_init)
        data_final = self.read_at("particle", step_final)

        up0_init = data_init["up00"]
        up0_final = data_final["up00"]

        def rotate_velocities(up):
            vx = up[:, 3]
            vy = up[:, 4]
            vz = up[:, 5]
            v_par = vx * e1[0] + vy * e1[1] + vz * e1[2]
            v_perp1 = vx * e2[0] + vy * e2[1] + vz * e2[2]
            v_perp2 = vx * e3[0] + vy * e3[1] + vz * e3[2]
            v_perp = np.sqrt(v_perp1**2 + v_perp2**2)
            return v_par / vae, v_perp / vae

        vpar_init, vperp_init = rotate_velocities(up0_init)
        vpar_final, vperp_final = rotate_velocities(up0_final)

        Nbins = 41
        vpar_range = (
            -4.0 * vte_para / vae,
            +4.0 * vte_para / vae,
            Nbins,
        )
        vperp_range = (
            0.0,
            4.0 * vte_perp / vae,
            Nbins,
        )

        hist_init = picnix.Histogram2D(vpar_init, vperp_init, vpar_range, vperp_range)
        hist_final = picnix.Histogram2D(vpar_final, vperp_final, vpar_range, vperp_range)

        X, Y, Z_init = hist_init.pcolormesh_args()
        _, _, Z_final = hist_final.pcolormesh_args()

        # normalize to probability
        Z_init = Z_init / np.sum(Z_init)
        Z_final = Z_final / np.sum(Z_final)
        Z_diff = Z_final - Z_init

        t_init = self.get_time_at("particle", step_init)
        t_final = self.get_time_at("particle", step_final)

        # 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        for ax in axes:
            ax.set_aspect("equal")

        # initial VDF
        ax = axes[0]
        vmax = max(Z_init.max(), 1e-10)
        norm = mpl.colors.LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        im0 = ax.pcolormesh(X, Y, Z_init, norm=norm, cmap="viridis", shading="nearest")
        cax0 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im0, cax=cax0)
        ax.set_title(
            r"$f_{{\rm initial}}(v_{{\parallel}}, v_{{\perp}}); \, \omega_{{pe}} t = {:.1f}$".format(
                t_init
            )
        )
        ax.set_xlabel(r"$v_\parallel / v_{Ae}$")
        ax.set_ylabel(r"$v_\perp / v_{Ae}$")

        # final VDF
        ax = axes[1]
        vmax = max(Z_final.max(), 1e-10)
        norm = mpl.colors.LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        im1 = ax.pcolormesh(X, Y, Z_final, norm=norm, cmap="viridis", shading="nearest")
        cax1 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1)
        ax.set_title(
            r"$f_{{\rm final}}(v_{{\parallel}}, v_{{\perp}}); \, \omega_{{pe}} t = {:.1f}$".format(
                t_final
            )
        )
        ax.set_xlabel(r"$v_\parallel / v_{Ae}$")

        # difference
        ax = axes[2]
        vlim = max(np.abs(Z_diff).max(), 1e-10)
        im2 = ax.pcolormesh(X, Y, Z_diff, cmap="RdBu_r", vmin=-vlim, vmax=+vlim, shading="nearest")
        cax2 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax2)
        ax.set_title(r"$\Delta f = f_{{\rm final}} - f_{{\rm initial}}$")
        ax.set_xlabel(r"$v_\parallel / v_{Ae}$")

        fig.tight_layout()
        return fig


def doit_job(profile, prefix):
    run = Run(profile)

    fig = run.magnetic_energy()
    fig.savefig("{:s}-db2.png".format(prefix))
    plt.close(fig)

    fig = run.anisotropy_history()
    fig.savefig("{:s}-anisotropy.png".format(prefix))
    plt.close(fig)

    fig = run.helicity_decomposition()
    fig.savefig("{:s}-helicity.png".format(prefix))
    plt.close(fig)

    fig = run.vdf_difference()
    fig.savefig("{:s}-vdf.png".format(prefix))
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="anisotropy",
        help="Prefix used for output image files",
    )
    parser.add_argument("profile", nargs=1, help="run profile")

    args = parser.parse_args()
    profile = args.profile[0]
    doit_job(profile, args.prefix)
