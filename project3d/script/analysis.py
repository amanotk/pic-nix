#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import json
import glob


class Run(object):
    def __init__(self, cfgfile):
        self.read_config(cfgfile)
        self.read_coord(self.file_field)

    def read_config(self, cfgfile):
        cfg = json.loads(open(cfgfile, "r").read())
        self.cfg = cfg
        self.Ns = cfg["parameter"]["Ns"]
        self.Nx = cfg["parameter"]["Nx"]
        self.Ny = cfg["parameter"]["Ny"]
        self.Nz = cfg["parameter"]["Nz"]
        self.delt = cfg["parameter"]["delt"]
        self.delh = cfg["parameter"]["delh"]
        for diagnostic in cfg["diagnostic"]:
            prefix = diagnostic["prefix"]
            path = diagnostic["path"]
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
        um = np.zeros((Nt, Nz, Ny, Nx, Ns, 11), dtype=np.float64)
        for i in range(Nt):
            with h5py.File(self.file_field[i], "r") as h5fp:
                uf[i, ...] = h5fp.get("/vds/uf")[()]
                um[i, ...] = h5fp.get("/vds/um")[()]
        self.uf = uf
        self.um = um

    def read_field_at(self, step):
        Ns = self.Ns
        Nz = self.Nz
        Ny = self.Ny
        Nx = self.Nx
        uf = np.zeros((Nz, Ny, Nx, 6), dtype=np.float64)
        um = np.zeros((Nz, Ny, Nx, Ns, 11), dtype=np.float64)
        with h5py.File(self.file_field[step], "r") as h5fp:
            uf = h5fp.get("/vds/uf")[()]
            um = h5fp.get("/vds/um")[()]
        return uf, um

    def read_particle_at(self, step):
        Ns = self.Ns
        up = list()
        with h5py.File(self.file_particle[step], "r") as h5fp:
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
        x = 0.5*(self.xedges[+1:] + self.xedges[:-1])
        y = 0.5*(self.yedges[+1:] + self.yedges[:-1])
        X, Y = np.broadcast_arrays(x[:,None], y[None,:])
        return X, Y, self.density


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
