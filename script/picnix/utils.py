#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import pathlib
import numpy as np
import json
import msgpack
import asyncio
import aiofiles

from picnix import DEFAULT_LOG_PREFIX


def get_json_meta(obj):
    meta = obj.get("meta")

    # endian
    endian = meta.get("endian")
    if endian == 1:  # little endian
        byteorder = "<"
    elif endian == 16777216:  # big endian
        byteorder = ">"
    else:
        byteorder = ""
        print("unrecognized endian flag: {}".format(endian))

    datafile = meta.get("rawfile")
    layout = meta.get("layout", meta.get("order", 0))  # for backward compatibility
    chunk_id_range = meta.get("chunk_id_range", None)

    return byteorder, datafile, layout, chunk_id_range


def process_json_content(content, filename):
    obj = json.loads(content)
    byteorder, datafile, layout, chunk_id_range = get_json_meta(obj)
    meta = {
        "byteorder": byteorder,
        "datafile": datafile,
        "layout": layout,
        "chunk_id_range": chunk_id_range,
        "dirname": str(pathlib.Path(filename).parent),
    }
    dataset = obj["dataset"]

    return dataset, meta


def process_meta(meta):
    byteorder = meta["byteorder"]
    layout = meta["layout"]
    datafile = os.sep.join([meta["dirname"], meta["datafile"]])
    return byteorder, layout, datafile


def read_jsonfile(filename):
    with open(filename, "r") as fp:
        content = fp.read()
        return process_json_content(content, filename)


def get_dataset_info(obj, byteorder):
    offset = obj["offset"]
    datatype = byteorder + obj["datatype"]
    shape = obj["shape"]
    return offset, datatype, shape


def read_single_dataset(fp, offset, datatype, shape):
    fp.seek(offset)
    x = np.fromfile(fp, datatype, np.prod(shape)).reshape(shape)
    if len(shape) == 1 and shape[0] == 1:
        x = x[0]
    return x


def read_datafile(dataset, meta, pattern):
    byteorder, layout, datafile = process_meta(meta)

    with open(datafile, "r") as fp:
        data = {}
        for dsname in dataset:
            if not re.match(pattern, dsname):
                continue
            offset, dtype, shape = get_dataset_info(dataset[dsname], byteorder)
            if layout == 0:
                shape = shape[::-1]
            data[dsname] = read_single_dataset(fp, offset, dtype, shape)

    return data


async def async_read_jsonfile(filename):
    try:
        async with aiofiles.open(filename, mode="r") as fp:
            content = await fp.read()
        return process_json_content(content, filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        raise


async def async_read_single_dataset(fp, offset, datatype, shape):
    await fp.seek(offset)
    buffer = await fp.read(np.prod(shape) * np.dtype(datatype).itemsize)
    x = np.frombuffer(buffer, dtype=datatype).reshape(shape)
    if len(shape) == 1 and shape[0] == 1:
        x = x[0]
    return x


async def async_read_datafile(dataset, meta, pattern):
    byteorder = meta["byteorder"]
    layout = meta["layout"]
    dirname = meta["dirname"]
    datafile = os.path.join(dirname, meta["datafile"])

    async with aiofiles.open(datafile, "rb") as fp:
        data = {}
        for dsname in dataset:
            if not re.match(pattern, dsname):
                continue
            offset, dtype, shape = get_dataset_info(dataset[dsname], byteorder)
            if layout == 0:
                shape = shape[::-1]
            data[dsname] = await async_read_single_dataset(fp, offset, dtype, shape)

    return data


def find_record_from_msgpack(filename, rank=None, step=None, name=None):
    data = []
    with open(filename, "rb") as fp:
        stream = fp.read()
        unpacker = msgpack.Unpacker(None, max_buffer_size=len(stream))
        unpacker.feed(stream)
        for record in unpacker:
            if record is None:
                continue
            flag = True
            flag = flag & (rank is None or rank == record.get("rank", -1))
            flag = flag & (step is None or step == record.get("step", -1))
            if name is None and flag:
                data.append(record)
            elif flag:
                data.append(record.get(name, None))

    return data


def convert_array_format(dataset, chunkmap):
    chunkid = np.array(chunkmap["chunkid"])
    coord = np.array(chunkmap["coord"])
    for key in dataset.keys():
        # determine data shape assuming 3D chunk
        csh = list(dataset[key].shape[1:])
        if len(csh) < 3:  # ignore dataset with dimensions < 3
            continue
        gsh = csh.copy()
        gsh[0] = csh[0] * chunkid.shape[0]
        gsh[1] = csh[1] * chunkid.shape[1]
        gsh[2] = csh[2] * chunkid.shape[2]
        csh = tuple(csh)  # shape of each chunk
        gsh = tuple(gsh)  # shape of global array
        # assign to new array
        data = np.zeros(gsh, dataset[key].dtype)
        for iz in range(chunkid.shape[0]):
            for iy in range(chunkid.shape[1]):
                for ix in range(chunkid.shape[2]):
                    ii = chunkid[iz, iy, ix]
                    cx = coord[ii, 0]
                    cy = coord[ii, 1]
                    cz = coord[ii, 2]
                    xslice = slice(cx * csh[2], (cx + 1) * csh[2])
                    yslice = slice(cy * csh[1], (cy + 1) * csh[1])
                    zslice = slice(cz * csh[0], (cz + 1) * csh[0])
                    data[zslice, yslice, xslice, ...] = dataset[key][ii]
        dataset[key] = data
    return dataset


def convert_to_mp4(prefix, fps, cleanup):
    import os
    import glob
    import subprocess

    ffmpeg_cmdline_format = (
        "ffmpeg -r {fps:d} -i {src:s} -vcodec libx264 -pix_fmt yuv420p -r 60 {dst:s}"
    )

    src = prefix + "-%*.png"
    dst = prefix + ".mp4"
    cmd = ffmpeg_cmdline_format.format(src=src, dst=dst, fps=fps)

    # run
    status = True
    try:
        if os.path.exists(dst):
            os.remove(dst)
        subprocess.run(cmd.split(), capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        status = False
        print(e.cmd)
        print(e.returncode)
        print(e.output)
        print(e.stdout)
        print(e.stderr)

    # cleanup if succeeded
    if status and cleanup:
        # remove files
        files = list()
        files.extend(glob.glob("{:s}-*.png".format(prefix)))
        for f in files:
            os.remove(f)


def plot_chunk_dist1d(ax, coord, rank, delx=1, colors="w"):
    import matplotlib as mpl

    cx = coord[:, 0]
    Nx = np.max(cx) + 1
    yy = np.zeros((Nx,), dtype=np.int32)
    yy[cx] = rank
    ix = np.argwhere(yy[+1:] - yy[:-1] != 0)[:, 0]
    for i in ix:
        ax.axvline(delx * i, color=colors, lw=0.5)


def plot_chunk_dist2d(ax, coord, rank, delx=1, dely=1, colors="w", width=1):
    import matplotlib as mpl

    cx = coord[:, 0]
    cy = coord[:, 1]
    Nx = np.max(cx) + 1
    Ny = np.max(cy) + 1
    ix = np.arange(Nx)
    iy = np.arange(Ny)
    zz = np.zeros((Ny, Nx), dtype=np.int32)
    zz[cy, cx] = rank
    Ix, Iy = np.broadcast_arrays(ix[None, :], iy[:, None])
    diffx = np.where(zz[:, +1:] - zz[:, :-1] == 0, 0, 1) * width
    diffy = np.where(zz[+1:, :] - zz[:-1, :] == 0, 0, 1) * width
    # vertical
    xsegments = np.zeros((Ny, Nx - 1, 2, 2), dtype=np.float64)
    xsegments[:, :, 0, 0] = delx * Ix[:, +1:]
    xsegments[:, :, 0, 1] = dely * Iy[:, +1:]
    xsegments[:, :, 1, 0] = delx * Ix[:, +1:]
    xsegments[:, :, 1, 1] = dely * (Iy[:, +1:] + 1)
    # horizontal
    ysegments = np.zeros((Ny - 1, Nx, 2, 2), dtype=np.float64)
    ysegments[:, :, 0, 0] = delx * Ix[+1:, :]
    ysegments[:, :, 0, 1] = dely * Iy[+1:, :]
    ysegments[:, :, 1, 0] = delx * (Ix[+1:, :] + 1)
    ysegments[:, :, 1, 1] = dely * Iy[+1:, :]
    # line segments
    segments = xsegments.reshape(Ny * (Nx - 1), 2, 2)
    xlines = mpl.collections.LineCollection(
        segments, linewidths=diffx.flat, colors=colors
    )
    ax.add_collection(xlines)
    segments = ysegments.reshape((Ny - 1) * Nx, 2, 2)
    ylines = mpl.collections.LineCollection(
        segments, linewidths=diffy.flat, colors=colors
    )
    ax.add_collection(ylines)


def plot_loadbalance(run, axs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # check load data
    status = False
    for diagnostic in run.config["diagnostic"]:
        if diagnostic["name"] == "load":
            status = True
            break
    if status == False:
        return False

    Nt = run.step_load.size
    Nc = run.Cx * run.Cy * run.Cz
    Nr = run.nprocess
    stepload = np.zeros((Nt,))
    chunkload = np.zeros((Nt, Nc))
    rankload = np.zeros((Nt, Nr))

    ## read time stamp
    log = run.config["application"]["log"]
    dirname = pathlib.Path(run.profile).parent / log.get("path", ".")
    filename = log.get("prefix", DEFAULT_LOG_PREFIX) + ".msgpack"
    data = find_record_from_msgpack(str(dirname / filename), rank=0)
    time = np.array([x["timestamp"]["unixtime"] for x in data])
    step = np.array([x["step"] for x in data])

    ## read load data
    for i, s in enumerate(run.step_load):
        stepload[i] = s
        data = run.read_load_at(s)
        load = data["load"].sum(axis=-1)
        rank = data["rank"]
        # load by chunk
        chunkload[i, :] = load
        # load by rank
        bins = np.arange(Nr + 1)
        hist, _ = np.histogram(rank, weights=load, bins=bins)
        rankload[i, :] = hist

    chunkmean = chunkload.mean(axis=1)
    chunkdelta = (chunkload.max(axis=1) - chunkload.min(axis=1)) / chunkmean * 100
    chunksigma = chunkload.std(axis=1) / chunkmean * 100
    rankmean = rankload.mean(axis=1)
    rankdelta = (rankload.max(axis=1) - rankload.min(axis=1)) / rankmean * 100
    ranksigma = rankload.std(axis=1) / rankmean * 100

    ## plot results
    plt.sca(axs[0])
    plt.plot(step[1:], time[+1:] - time[:-1], ".", ms=1)
    plt.semilogy()
    plt.ylabel("Elapsed time / step [s]")
    plt.grid()

    plt.sca(axs[1])
    plt.plot(stepload[1:], rankdelta[1:], label="max - min")
    plt.plot(stepload[1:], ranksigma[1:], label="sigma")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("Rank load imbalance [%]")
    plt.grid()

    plt.sca(axs[2])
    plt.plot(stepload[1:], chunkdelta[1:], label="max - min")
    plt.plot(stepload[1:], chunksigma[1:], label="sigma")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("Chunk load imbalacne [%]")
    plt.grid()

    plt.xlabel("Time step")
    plt.xlim(0, stepload[-1])
    plt.suptitle(
        "# Rank = {:d}, # Chunk = {:d}, Average Imbalance = {:6.2}%".format(
            Nr, Nc, Nr / Nc * 100
        )
    )

    return True
