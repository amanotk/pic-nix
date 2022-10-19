#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tqdm
from concurrent import futures

import numpy as np
import matplotlib as mpl

# for batch mode
mpl.use("Agg")
from matplotlib import pyplot as plt

# for analysis package
sys.path.append("../script")
import analysis

# for summary plot
summary_plot_config = dict(
    figure=dict(
        width=6.4,
        height=6.4,
    ),
    subplot=dict(
        top=0.95,
        bottom=0.08,
        left=0.12,
        right=0.85,
        hspace=0.25,
        wspace=0.02,
    ),
)


def doit_parallel(cfgfile, **kwargs):
    run = analysis.Run(cfgfile)

    # default argument
    ebinx = (0, run.Nx, run.Nx + 1)
    ibinx = (0, run.Nx, run.Nx + 1)
    ebiny = (-30, +30, 61)
    ibiny = (-5, +5, 61)
    kw = dict(
        ele=dict(binx=ebinx, biny=ebiny),
        ion=dict(binx=ibinx, biny=ibiny),
        xlim=(0, run.Nx * run.delh),
        ylim0=(0.5, 2.5),
        ylim1=(-10, +10),
        me=1.0e0,
        mi=1.0e2,
    )
    # overwrite
    kw.update(**kwargs)

    if run.time_field.size == run.time_particle.size:
        Nt = len(run.file_field)
        with futures.ProcessPoolExecutor() as executor:
            future_list = []
            for step in range(Nt):
                xc = run.xc
                uf, um = run.read_field_at(step)
                up = run.read_particle_at(step)
                if run.time_field[step] != run.time_particle[step]:
                    raise ValueError("snapshots of field and particle do not match")
                kw["time"] = run.time_field[step]
                kw["output"] = "beam-summary-{:08d}.png".format(step)
                # submit job
                future = executor.submit(summary_plot, xc, uf, um, up, **kw)
                future_list.append(future)
            # show progress
            if tqdm is not None:
                progress_bar = tqdm.tqdm(total=Nt, desc="Generating Plot")
                for future in futures.as_completed(future_list):
                    progress_bar.update(1)


def summary_plot(xc, uf, um, up, **kwargs):
    binx = [0] * 3
    biny = [0] * 3
    try:
        binx[0] = kwargs["ele"]["binx"]
        binx[1] = kwargs["ele"]["binx"]
        binx[2] = kwargs["ion"]["binx"]
        biny[0] = kwargs["ele"]["biny"]
        biny[1] = kwargs["ele"]["biny"]
        biny[2] = kwargs["ion"]["biny"]
        output = kwargs["output"]
        xlim = kwargs["xlim"]
        time = kwargs["time"]
        ylim0 = kwargs.get("ylim0", None)
        ylim1 = kwargs.get("ylim1", None)
        me = kwargs.get("me", 1.0)
        mi = kwargs.get("mi", 1.0)
    except Exception as e:
        print("Inappropriate keyword arguments")
        raise e

    ## figure and axes
    figw = summary_plot_config["figure"]["width"]
    figh = summary_plot_config["figure"]["height"]
    figure = plt.figure(1, figsize=(figw, figh), dpi=120)
    figure.clf()
    gridspec = figure.add_gridspec(
        4,
        2,
        width_ratios=[50, 1],
        height_ratios=[1, 1, 1, 1],
        **summary_plot_config["subplot"]
    )
    axs = list()
    for i in range(4):
        axs.append(figure.add_subplot(gridspec[i, 0]))

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
    cax = figure.add_subplot(gridspec[2, 1])
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
    cax = figure.add_subplot(gridspec[3, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_i(x, v_x)$")

    ## appearance
    for i in range(4):
        axs[i].set_xlim(xlim)
    figure.suptitle(r"$t = {:6.2f}$".format(time))

    # save
    plt.savefig(output)


if __name__ == "__main__":
    doit_parallel(sys.argv[1])
