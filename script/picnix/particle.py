#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
