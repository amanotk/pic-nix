#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
