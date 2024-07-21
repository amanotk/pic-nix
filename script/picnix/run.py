#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pathlib
import h5py
import json
import msgpack
import glob

from picnix import (
    DEFAULT_LOG_PREFIX,
    DEFAULT_LOAD_PREFIX,
    DEFAULT_FIELD_PREFIX,
    DEFAULT_PARTICLE_PREFIX,
)
from .utils import *


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
            filename = "{:s}_{:08d}.msgpack".format(
                prefix, (step // interval) * interval
            )
        else:
            filename = "{:s}_*.msgpack".format(prefix)
        return str(basedir / path / filename)

    def format_log_filename(self):
        basedir = pathlib.Path(self.profile).parent
        path = self.config["application"]["log"].get("path", ".")
        prefix = self.config["application"]["log"].get("prefix", DEFAULT_LOG_PREFIX)
        filename = "{:s}.msgpack".format(prefix)
        return str(basedir / path / filename)

    def read_profile(self, profile):
        # read profile
        with open(profile, "rb") as fp:
            obj = msgpack.load(fp)
            self.timestamp = obj["timestamp"]
            self.nprocess = obj["nprocess"]
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
            path = basedir / diagnostic.get("path", ".")
            # load
            if diagnostic["name"] == "load":
                data = "{}_*.{}".format(
                    diagnostic.get("prefix", DEFAULT_LOAD_PREFIX), ext
                )
                file = sorted(glob.glob(str(path / data)))
                self.step_load = np.arange(len(file)) * interval
                self.time_load = np.arange(len(file)) * self.delt * interval
                self.file_load = file
            # field
            if diagnostic["name"] == "field":
                data = "{}_*.{}".format(
                    diagnostic.get("prefix", DEFAULT_FIELD_PREFIX), ext
                )
                file = sorted(glob.glob(str(path / data)))
                self.step_field = np.arange(len(file)) * interval
                self.time_field = np.arange(len(file)) * self.delt * interval
                self.file_field = file
            # particle
            if diagnostic["name"] == "particle":
                data = "{}_*.{}".format(
                    diagnostic.get("prefix", DEFAULT_PARTICLE_PREFIX), ext
                )
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
        filename = self.format_log_filename()
        data = find_record_from_msgpack(filename, rank=0, step=step, name="rebalance")[
            0
        ]
        return data

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
            data = {}
            for dsname in ("load", "rank"):
                ds = dataset.get(dsname)
                offset, dtype, shape = get_dataset_info(ds, byteorder)
                if order == 0:
                    shape == shape[::-1]
                data[dsname] = get_dataset_data(fp, offset, dtype, shape)

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
