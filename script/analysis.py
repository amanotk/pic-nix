#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import numpy as np
import h5py
import json
import msgpack
import glob


def get_json_meta(obj):
    meta = obj.get("meta")

    # endian
    endian = meta.get("endian")
    if endian == 1:  # little endian
        byteorder = "<"
    elif endian == 16777216:  # big endian
        byteorder = ">"
    else:
        print("unrecognized endian flag: {}".format(endian))

    # check raw data file
    datafile = meta.get("rawfile")

    # check array order
    order = meta.get("order", 0)

    return byteorder, datafile, order


def get_dataset_info(obj, byteorder):
    offset = obj["offset"]
    datatype = byteorder + obj["datatype"]
    shape = obj["shape"]
    datasize = np.product(shape) * np.dtype(datatype).itemsize
    return offset, datatype, shape


def get_dataset_data(fp, offset, datatype, shape):
    fp.seek(offset)
    x = np.fromfile(fp, datatype, np.prod(shape)).reshape(shape)
    if len(shape) == 1 and shape[0] == 1:
        x = x[0]
    return x


def find_record_from_msgpack(filename, name, step=None):
    with open(filename, "rb") as fp:
        obj = msgpack.load(fp)

    # find data
    data = np.array([x.get(name, None) for x in obj])
    data = np.compress(data != None, data)

    # choose step
    if step is not None:
        for x in data:
            if x.get("step", -1) == step:
                return x

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


def plot_chunk_dist2d(ax, coord, rank, delx=1, dely=1, colors="w"):
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
    diffx = np.where(zz[:, +1:] - zz[:, :-1] == 0, 0, 1)
    diffy = np.where(zz[+1:, :] - zz[:-1, :] == 0, 0, 1)
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
    xlines = mpl.collections.LineCollection(segments, linewidths=diffx.flat, colors=colors)
    ax.add_collection(xlines)
    segments = ysegments.reshape((Ny - 1) * Nx, 2, 2)
    ylines = mpl.collections.LineCollection(segments, linewidths=diffy.flat, colors=colors)
    ax.add_collection(ylines)


def plot_loadbalance(run, nrank, axs):
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
    Nr = nrank
    stepload = np.zeros((Nt,))
    chunkload = np.zeros((Nt, Nc))
    rankload = np.zeros((Nt, Nr))

    ## read time stamp
    pattern = run.format_filename(**run.config["application"]["log"])
    file = glob.glob(pattern)
    time = [0] * len(file)
    step = [0] * len(file)
    for i, filename in enumerate(file):
        push = find_record_from_msgpack(filename, "push")
        time[i] = np.array([p["start"] for p in push])
        step[i] = np.array([p["step"] for p in push])
    time = np.concatenate(time)
    step = np.concatenate(step)

    ## read load data
    for i, s in enumerate(run.step_load):
        stepload[i] = s
        # load by chunk
        chunkload[i, :] = run.read_load_at(s).sum(axis=-1)
        # load by rank
        rebalance = run.find_rebalance_log_at(s)
        boundary = np.array(rebalance["boundary"])
        assert boundary.size == Nr + 1  # check consistency
        hist, _ = np.histogram(np.arange(Nc), weights=chunkload[i, :], bins=boundary)
        rankload[i, :] = hist

    chunkmean = chunkload.mean(axis=1)
    chunkdelta = (chunkload.max(axis=1) - chunkload.min(axis=1)) / chunkmean * 100
    chunksigma = chunkload.std(axis=1) / chunkmean * 100
    rankmean = rankload.mean(axis=1)
    rankdelta = (rankload.max(axis=1) - rankload.min(axis=1)) / rankmean * 100
    ranksigma = rankload.std(axis=1) / rankmean * 100

    ## plot results
    plt.sca(axs[0])
    plt.plot(step[1:], time[+1:] - time[:-1], "-")
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
        "# Rank = {:d}, # Chunk = {:d}, Average Imbalance = {:6.2}%".format(Nr, Nc, Nr / Nc * 100)
    )

    return True


def get_wk_spectrum(f, delt=1.0, delh=1.0):
    if f.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if f.dtype == np.float32 or f.dtype == np.float64:
        # real
        Nt = f.shape[0]
        Nx = f.shape[1]
        P = np.abs(np.fft.fftshift(np.fft.rfft2(f, norm="ortho"), axes=(0,))) ** 2
        w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, delt))
        k = 2 * np.pi * np.fft.rfftfreq(Nx, delh)
        W, K = np.broadcast_arrays(w[:, None], k[None, :])
        return P, W, K
    else:
        # complex
        Nt = f.shape[0]
        Nx = f.shape[1]
        P = np.abs(np.fft.fftshift(np.fft.fft2(f, norm="ortho"), axes=(0, 1))) ** 2
        w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, delt))
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, delh))
        W, K = np.broadcast_arrays(w[:, None], k[None, :])
        return P, W, K


def plot_wk_spectrum(P, W, K, filename, title, **kwargs):
    import matplotlib
    from matplotlib import pyplot as plt

    figure = plt.figure()
    im = plt.pcolormesh(K, W, np.log10(P), shading="nearest")
    cl = plt.colorbar(im)
    # xlim
    kmax = kwargs.get("kmax", 0.25 * K.max())
    kmin = kwargs.get("kmin", 0.25 * K.min())
    plt.xlim(kmin, kmax)
    plt.xlabel(r"$k$")
    # ylim
    wmax = kwargs.get("wmax", W.max())
    wmin = kwargs.get("wmin", W.min())
    plt.ylim(wmin, wmax)
    plt.ylabel(r"$\omega$")
    # clim
    cmax = kwargs.get("cmin", np.log10(P.max()))
    cmin = kwargs.get("cmin", cmax - 4)
    plt.clim(cmin, cmax)
    # save
    plt.title(title)
    plt.savefig(filename)


class Run(object):
    def __init__(self, profile, use_hdf5=False):
        self.use_hdf5 = use_hdf5
        self.profile = profile
        self.read_profile(profile)
        self.set_coordinate()

    def format_filename(self, step=None, **kwargs):
        basedir = pathlib.Path(self.profile).parent
        path = kwargs.get("path", ".")
        prefix = kwargs.get("prefix", "")
        interval = kwargs.get("interval", 1)
        if step is not None:
            filename = "{:s}_{:08d}.msgpack".format(prefix, (step // interval) * interval)
        else:
            filename = "{:s}_*.msgpack".format(prefix)
        return str(basedir / path / filename)

    def read_profile(self, profile):
        # read profile
        with open(profile, "rb") as fp:
            obj = msgpack.load(fp)
            self.timestamp = obj["timestamp"]
            self.chunkmap = obj["chunkmap"]
            self.config = obj["configuration"]

        # store some parameters
        parameter = self.config["parameter"]
        self.Ns = parameter["Ns"]
        self.Nx = parameter["Nx"]
        self.Ny = parameter["Ny"]
        self.Nz = parameter["Nz"]
        self.Cx = parameter["Cx"]
        self.Cy = parameter["Cy"]
        self.Cz = parameter["Cz"]
        self.delt = parameter["delt"]
        self.delh = parameter["delh"]

        # find data files
        if self.use_hdf5:
            ext = "h5"
        else:
            ext = "json"
        basedir = pathlib.Path(profile).parent
        for diagnostic in self.config["diagnostic"]:
            interval = diagnostic["interval"]
            path = basedir / diagnostic["path"]
            # load
            if diagnostic["name"] == "load":
                data = "{}_*.{}".format(diagnostic["prefix"], ext)
                file = sorted(glob.glob(str(path / data)))
                self.step_load = np.arange(len(file)) * interval
                self.time_load = np.arange(len(file)) * self.delt * interval
                self.file_load = file
            # field
            if diagnostic["name"] == "field":
                data = "{}_*.{}".format(diagnostic["prefix"], ext)
                file = sorted(glob.glob(str(path / data)))
                self.step_field = np.arange(len(file)) * interval
                self.time_field = np.arange(len(file)) * self.delt * interval
                self.file_field = file
            # particle
            if diagnostic["name"] == "particle":
                data = "{}_*.{}".format(diagnostic["prefix"], ext)
                file = sorted(glob.glob(str(path / data)))
                self.step_particle = np.arange(len(file)) * interval
                self.time_particle = np.arange(len(file)) * self.delt * interval
                self.file_particle = file

    def set_coordinate(self):
        self.xc = self.delh * np.arange(self.Nx)
        self.yc = self.delh * np.arange(self.Ny)
        self.zc = self.delh * np.arange(self.Nz)

    def get_chunk_rank(self, step):
        rebalance = self.find_rebalance_log_at(step)
        boundary = np.array(rebalance["boundary"])
        rank = np.zeros((boundary[-1],), dtype=np.int32)
        for r in range(boundary.size - 1):
            rank[boundary[r] : boundary[r + 1]] = r
        return rank

    def find_rebalance_log_at(self, step):
        filename = self.format_filename(step, **self.config["application"]["log"])
        return find_record_from_msgpack(filename, "rebalance", step)

    def find_step_index_load(self, step):
        return np.searchsorted(self.step_load, step)

    def find_step_index_field(self, step):
        return np.searchsorted(self.step_field, step)

    def find_step_index_particle(self, step):
        return np.searchsorted(self.step_particle, step)

    def get_load_time_at(self, step):
        index = self.find_step_index_load(step)
        return self.time_load[index]

    def get_field_time_at(self, step):
        index = self.find_step_index_field(step)
        return self.time_field[index]

    def get_particle_time_at(self, step):
        index = self.find_step_index_particle(step)
        return self.time_particle[index]

    def read_load_at(self, step):
        return self.read_load_at_json(step)

    def read_field_at(self, step):
        if self.use_hdf5:
            return self.read_field_at_hdf5(step)
        else:
            return self.read_field_at_json(step)

    def read_particle_at(self, step):
        if self.use_hdf5:
            return self.read_particle_at_hdf5(step)
        else:
            return self.read_particle_at_json(step)

    def read_load_at_json(self, step):
        index = self.find_step_index_load(step)

        # read json
        jsonfile = self.file_load[index]
        with open(jsonfile, "r") as fp:
            obj = json.load(fp)
            byteorder, datafile, order = get_json_meta(obj)
            dataset = obj["dataset"]

        # read data
        datafile = str(pathlib.Path(jsonfile).parent / datafile)
        with open(datafile, "r") as fp:
            offset, dtype, shape = get_dataset_info(dataset.get("load"), byteorder)
            if order == 0:
                shape == shape[::-1]
            data = get_dataset_data(fp, offset, dtype, shape)

        return data

    def read_field_at_json(self, step):
        index = self.find_step_index_field(step)

        # read json
        jsonfile = self.file_field[index]
        with open(jsonfile, "r") as fp:
            obj = json.load(fp)
            byteorder, datafile, order = get_json_meta(obj)
            dataset = obj["dataset"]

        # read data
        datafile = str(pathlib.Path(jsonfile).parent / datafile)
        with open(datafile, "r") as fp:
            data = {}
            for dsname in ("uf", "uj", "um"):
                ds = dataset.get(dsname)
                offset, dtype, shape = get_dataset_info(ds, byteorder)
                if order == 0:
                    shape == shape[::-1]
                data[dsname] = get_dataset_data(fp, offset, dtype, shape)

        # convert array format
        data = convert_array_format(data, self.chunkmap)

        return data

    def read_field_at_hdf5(self, step):
        index = self.find_step_index_field(step)

        with h5py.File(self.file_field[index], "r") as h5fp:
            uf = h5fp.get("/vds/uf")[()]
            uj = h5fp.get("/vds/uf")[()]
            um = h5fp.get("/vds/um")[()]

        return dict(uf=uf, uj=uj, um=um)

    def read_particle_at_json(self, step):
        index = self.find_step_index_particle(step)

        # read json
        jsonfile = self.file_particle[index]
        with open(jsonfile, "r") as fp:
            obj = json.load(fp)
            byteorder, datafile, order = get_json_meta(obj)
            dataset = obj["dataset"]

        # read data
        datafile = str(pathlib.Path(jsonfile).parent / datafile)
        with open(datafile, "r") as fp:
            Ns = self.Ns
            up = [0] * Ns
            for i in range(Ns):
                dsname = "up{:02d}".format(i)
                ds = dataset.get(dsname)
                offset, dtype, shape = get_dataset_info(ds, byteorder)
                if order == 0:
                    shape == shape[::-1]
                up[i] = get_dataset_data(fp, offset, dtype, shape)

        return up

    def read_particle_at_hdf5(self, step):
        index = self.find_step_index_particle(step)

        Ns = self.Ns
        up = list()
        with h5py.File(self.file_particle[index], "r") as h5fp:
            for i in range(Ns):
                dsname = "/up{:02d}".format(i)
                up.append(h5fp.get(dsname)[()])

        return up


class Histogram2D:
    def __init__(self, x, y, binx, biny, logx=False, logy=False):
        binx = self.handle_bin_arg(binx, logx)
        biny = self.handle_bin_arg(biny, logy)
        weights = np.ones_like(x) / x.size
        result = np.histogram2d(x, y, bins=(binx, biny), weights=weights)
        self.density = result[0]
        self.xedges = result[1]
        self.yedges = result[2]

    def handle_bin_arg(self, bin, logscale=False):
        if isinstance(bin, tuple) and logscale == False:
            return np.linspace(bin[0], bin[1], bin[2])
        if isinstance(bin, tuple) and logscale == True:
            return np.geomspace(bin[0], bin[1], bin[2])
        if isinstance(bin, np.ndarray) and bin.ndim == 1:
            return bin
        raise ValueError("Invalid argument")

    def pcolormesh_args(self):
        x = 0.5 * (self.xedges[+1:] + self.xedges[:-1])
        y = 0.5 * (self.yedges[+1:] + self.yedges[:-1])
        X, Y = np.broadcast_arrays(x[:, None], y[None, :])
        return X, Y, self.density


if __name__ == "__main__":
    pass
