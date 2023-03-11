#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import json
import glob


class Run(object):
    def __init__(self, cfgfile):
        self.dirname = os.path.dirname(cfgfile)
        self.read_config(cfgfile)
        self.read_coord(self.file_field)

    def read_config(self, cfgfile):
        cfg = json.loads(open(cfgfile, "r").read())
        self.cfg = cfg
        self.Ns = cfg["parameter"]["Ns"]
        self.Nx = cfg["parameter"]["Nx"]
        self.Ny = cfg["parameter"]["Ny"]
        self.Nz = cfg["parameter"]["Nz"]
        self.Cx = cfg["parameter"]["Cx"]
        self.Cy = cfg["parameter"]["Cy"]
        self.Cz = cfg["parameter"]["Cz"]
        self.delt = cfg["parameter"]["delt"]
        self.delh = cfg["parameter"]["delh"]
        for diagnostic in cfg["diagnostic"]:
            prefix = diagnostic["prefix"]
            path = os.sep.join([self.dirname, diagnostic["path"]])
            interval = diagnostic["interval"]
            file = sorted(glob.glob(os.sep.join([path, prefix]) + "*.h5"))
            # field
            if diagnostic["name"] == "field":
                self.file_field = file
                self.time_field = np.arange(len(file)) * self.delt * interval
            # particle
            if diagnostic["name"] == "particle":
                self.file_particle = file
                self.time_particle = np.arange(len(file)) * self.delt * interval

    def read_coord(self, files):
        with h5py.File(files[0], "r") as h5fp:
            self.xc = np.unique(h5fp.get("xc")[()])
            self.yc = np.unique(h5fp.get("yc")[()])
            self.zc = np.unique(h5fp.get("zc")[()])

    def read_field_all(self):
        Nt = len(self.file_field)
        Ns = self.Ns
        Nz = self.Nz
        Ny = self.Ny
        Nx = self.Nx
        uf = np.zeros((Nt, Nz, Ny, Nx, 6), dtype=np.float64)
        uj = np.zeros((Nt, Nz, Ny, Nx, 4), dtype=np.float64)
        um = np.zeros((Nt, Nz, Ny, Nx, Ns, 11), dtype=np.float64)
        for i in range(Nt):
            with h5py.File(self.file_field[i], "r") as h5fp:
                uf[i, ...] = h5fp.get("/vds/uf")[()]
                uj[i, ...] = h5fp.get("/vds/uj")[()]
                um[i, ...] = h5fp.get("/vds/um")[()]
        self.uf = uf
        self.uj = uj
        self.um = um

    def read_field_at(self, step):
        Ns = self.Ns
        Nz = self.Nz
        Ny = self.Ny
        Nx = self.Nx
        with h5py.File(self.file_field[step], "r") as h5fp:
            uf = h5fp.get("/vds/uf")[()]
            uj = h5fp.get("/vds/uf")[()]
            um = h5fp.get("/vds/um")[()]
        return uf, uj, um

    def read_particle_at(self, step):
        Ns = self.Ns
        up = list()
        with h5py.File(self.file_particle[step], "r") as h5fp:
            for i in range(Ns):
                dsname = "/up{:02d}".format(i)
                up.append(h5fp.get(dsname)[()])
        return up

    def read_chunkmap_at(self, step):
        with h5py.File(self.file_field[step], "r") as h5fp:
            rank = h5fp.get("/chunkmap/rank")[()]
            coord = h5fp.get("/chunkmap/coord")[()]
        cdelx = self.Nx // self.Cx * self.delh
        cdely = self.Ny // self.Cy * self.delh
        cdelz = self.Nz // self.Cz * self.delh
        return rank, coord, cdelx, cdely, cdelz


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


def plot_chunk_dist1d(ax, coord, rank, delx=1, colors="w"):
    import matplotlib as mpl

    cx = coord[:, 0]
    Nx = np.max(cx) + 1
    yy = np.zeros((Nx,), dtype=np.int32)
    yy[cx] = rank
    ix = np.argwhere(yy[+1:] - yy[:-1] != 0)[:,0]
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


if __name__ == "__main__":
    pass
