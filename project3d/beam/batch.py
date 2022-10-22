#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tqdm
from concurrent import futures

import numpy as np
import h5py
import matplotlib as mpl

# for batch mode
mpl.use("Agg")
from matplotlib import pyplot as plt

# for analysis package
sys.path.append("../script")
import analysis

# for two-stream instability
summary_plot_config_2stream = dict(
    figure=dict(width=6.4, height=6.4,),
    subplot=dict(
        top=0.95, bottom=0.08, left=0.12, right=0.85, hspace=0.25, wspace=0.02,
    ),
    prefix="2stream",
)

# for weibel instability
summary_plot_config_weibel = dict(
    figure=dict(width=9.6, height=3.6,),
    subplot=dict(
        top=0.80, bottom=0.14, left=0.075, right=0.95, hspace=0.05, wspace=0.35
    ),
    prefix="weibel",
)

# command line for ffmpeg
ffmpeg_cmdline_format = (
    "ffmpeg -r {fps:d} -i {src:s} -vcodec libx264 -pix_fmt yuv420p -r 60 {dst:s}"
)


def convert_to_mp4(prefix, fps):
    import os
    import glob
    import subprocess

    src = prefix + "-%08d.png"
    dst = prefix + ".mp4"
    cmd = ffmpeg_cmdline_format.format(src=src, dst=dst, fps=fps)

    # run
    status = True
    try:
        if os.path.exists(dst):
            os.remove(dst)
        print("Running command : {:s}".format(cmd))
        subprocess.run(cmd.split(), capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        status = False
        print(e.cmd)
        print(e.returncode)
        print(e.output)
        print(e.stdout)
        print(e.stderr)

    # cleanup if succeeded
    if status:
        # remove files
        files = list()
        files.extend(glob.glob("{:s}-*.png".format(prefix)))
        for f in files:
            os.remove(f)


def doit_parallel_2stream(cfgfile, **kwargs):
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

    time1 = run.time_field
    time2 = run.time_particle
    if time1.size != time2.size or np.allclose(time1, time2) == False:
        raise ValueError("snapshots of field and particle do not match")

    Nt = len(run.file_field)
    prefix = summary_plot_config_2stream["prefix"]
    with futures.ProcessPoolExecutor() as executor:
        future_list = []
        for step in range(Nt):
            xc = run.xc
            uf, uj, um = run.read_field_at(step)
            up = run.read_particle_at(step)
            rank, coord, cdelx, cdely, cdelz = run.read_chunkmap_at(step)
            kw["time"] = run.time_field[step]
            kw["output"] = "{:s}-{:08d}.png".format(prefix, step)
            kw["rank"] = rank
            kw["coord"] = coord
            kw["cdelx"] = cdelx
            # submit job
            future = executor.submit(summary_plot_2stream, xc, uf, um, up, **kw)
            future_list.append(future)
        # show progress
        if tqdm is not None:
            progress_bar = tqdm.tqdm(total=Nt, desc="Generating Plot")
            for future in futures.as_completed(future_list):
                progress_bar.update(1)

    # convert to mp4
    convert_to_mp4(prefix, 10)


def doit_parallel_weibel(cfgfile, **kwargs):
    run = analysis.Run(cfgfile)

    # default argument
    kw = dict(xlim=(0, run.Nx * run.delh), ylim=(0, run.Ny + run.delh),)
    # overwrite
    kw.update(**kwargs)

    Nt = len(run.file_field)
    prefix = summary_plot_config_weibel["prefix"]
    with futures.ProcessPoolExecutor() as executor:
        future_list = []
        for step in range(Nt):
            xc = run.xc
            yc = run.yc
            uf, uj, um = run.read_field_at(step)
            rank, coord, cdelx, cdely, cdelz = run.read_chunkmap_at(step)
            kw["time"] = run.time_field[step]
            kw["output"] = "{:s}-{:08d}.png".format(prefix, step)
            kw["rank"] = rank
            kw["coord"] = coord
            kw["cdelx"] = cdelx
            kw["cdely"] = cdely
            # submit job
            future = executor.submit(summary_plot_weibel, xc, yc, uf, uj, um, **kw)
            future_list.append(future)
        # show progress
        if tqdm is not None:
            progress_bar = tqdm.tqdm(total=Nt, desc="Generating Plot")
            progress_bar.update(0)
            for future in futures.as_completed(future_list):
                progress_bar.update(1)

    # convert to mp4
    convert_to_mp4(prefix, 10)


def summary_plot_2stream(xc, uf, um, up, **kwargs):
    # config
    config = summary_plot_config_2stream

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
        me = kwargs["me"]
        mi = kwargs["mi"]
        plot_chunk = kwargs.get("plot_chunk", True)
        rank = kwargs["rank"]
        coord = kwargs["coord"]
        cdelx = kwargs["cdelx"]
    except Exception as e:
        print("Inappropriate keyword arguments")
        raise e

    ## figure and axes
    figw = config["figure"]["width"]
    figh = config["figure"]["height"]
    figure = plt.figure(1, figsize=(figw, figh), dpi=120)
    figure.clf()
    gridspec = figure.add_gridspec(
        4, 2, width_ratios=[50, 1], height_ratios=[1, 1, 1, 1], **config["subplot"]
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
        if plot_chunk:
            analysis.plot_chunk_dist1d(axs[i], coord, rank, cdelx, colors="gray")

    figure.suptitle(r"$t = {:6.2f}$".format(time))

    ## save
    plt.savefig(output)


def summary_plot_weibel(xc, yc, uf, uj, um, **kwargs):
    # config
    config = summary_plot_config_weibel

    try:
        xlim = kwargs["xlim"]
        ylim = kwargs["ylim"]
        time = kwargs["time"]
        output = kwargs["output"]
        plot_chunk = kwargs.get("plot_chunk", True)
        rank = kwargs["rank"]
        coord = kwargs["coord"]
        cdelx = kwargs["cdelx"]
        cdely = kwargs["cdely"]
    except Exception as e:
        print("Inappropriate keyword arguments")
        raise e

    ## figure and axes
    figw = config["figure"]["width"]
    figh = config["figure"]["height"]
    figure = plt.figure(1, figsize=(figw, figh), dpi=120)
    figure.clf()
    gridspec = figure.add_gridspec(
        2, 3, width_ratios=[1, 1, 1], height_ratios=[2, 50], **config["subplot"]
    )
    axs = list()
    for i in range(3):
        ax = figure.add_subplot(gridspec[1, i])
        ax.set_aspect("equal")
        axs.append(ax)

    # coordinate
    X, Y = np.broadcast_arrays(xc[None, :], yc[:, None])

    ## density
    plt.sca(axs[0])
    ro = um[..., 0].sum(axis=(0, 3)) / um.shape[0] / 2
    plt.pcolormesh(X, Y, ro, shading="nearest")
    # colorbar
    cax = figure.add_subplot(gridspec[0, 0])
    plt.colorbar(cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.set_title(r"$N$")

    ## magnetic field strength
    plt.sca(axs[1])
    b0 = (uf[..., 3] ** 2 + uf[..., 4] ** 2 + uf[..., 5] ** 2).mean(axis=(0))
    plt.pcolormesh(X, Y, b0, shading="nearest")
    cax = figure.add_subplot(gridspec[0, 1])
    plt.colorbar(cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.set_title(r"$|B|$")

    ## current density
    plt.sca(axs[2])
    jz = uj[..., 3].mean(axis=(0))
    plt.pcolormesh(X, Y, jz, shading="nearest")
    cax = figure.add_subplot(gridspec[0, 2])
    plt.colorbar(cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.set_title(r"$J_z$")

    ## appearance
    for ax in axs:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        # plot chunk distribution
        if plot_chunk:
            analysis.plot_chunk_dist2d(ax, coord, rank, cdelx, cdely, colors="w")
    figure.suptitle(r"$t = {:6.2f}$".format(time), x=0.5, y=0.98)

    ## save
    plt.savefig(output)


if __name__ == "__main__":
    import optparse

    problem_choices = (None, "2stream", "weibel")
    parser = optparse.OptionParser()
    parser.add_option(
        "-p",
        "--problem",
        dest="problem",
        type="choice",
        choices=problem_choices,
        default=None,
        help="choice of problem",
    )
    options, args = parser.parse_args()

    if options.problem is None:
        pass
    elif options.problem == '2stream':
        doit_parallel_2stream(args[0])
    elif options.problem == 'weibel':
        doit_parallel_weibel(args[0])

