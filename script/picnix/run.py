#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import pathlib
import json
import msgpack

from picnix import (
    DEFAULT_LOG_PREFIX,
    DEFAULT_LOAD_PREFIX,
    DEFAULT_FIELD_PREFIX,
    DEFAULT_PARTICLE_PREFIX,
    DEFAULT_TRACER_PREFIX,
)
from .utils import *
from .diag import DiagHandler


class Run(object):
    def __init__(self, profile):
        self.cache = dict()
        self.profile = profile
        self.read_profile(profile)
        self.set_coordinate()

    def format_log_filename(self):
        basedir = pathlib.Path(self.profile).parent
        path = self.config["application"]["log"].get("path", ".")
        prefix = self.config["application"]["log"].get("prefix", DEFAULT_LOG_PREFIX)
        filename = "{:s}.msgpack".format(prefix)
        return str(basedir / path / filename)

    def clear_cache(self):
        self.cache = dict()

    def get_basedir(self):
        profile = pathlib.Path(self.profile)
        basedir = pathlib.Path(self.config["application"].get("basedir", "."))
        basedir1 = profile.parent.parent / basedir
        basedir2 = profile.parent
        if basedir1.absolute() != basedir2.absolute():
            raise ValueError("Inconsistent basedir in profile and application")
        return str(basedir1)

    def get_iomode(self):
        return self.config["application"].get("iomode", "mpiio")

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

        # diagnostic
        basedir = self.get_basedir()
        iomode = self.get_iomode()
        self.diag_handlers = dict()

        for diagnostic in self.config["diagnostic"]:
            handler = DiagHandler.create_handler(diagnostic, basedir, iomode)
            if handler is not None:
                self.diag_handlers[handler.get_prefix()] = handler

    def set_coordinate(self):
        self.xc = self.delh * (np.arange(self.Nx) + 0.5)
        self.yc = self.delh * (np.arange(self.Ny) + 0.5)
        self.zc = self.delh * (np.arange(self.Nz) + 0.5)

    def get_chunk_rank(self, step):
        rebalance = self.find_rebalance_log_at(step)
        boundary = np.array(rebalance["boundary"])
        rank = np.zeros((boundary[-1],), dtype=np.int32)
        for r in range(boundary.size - 1):
            rank[boundary[r] : boundary[r + 1]] = r
        return rank

    def find_rebalance_log_at(self, step):
        filename = self.format_log_filename()
        data = find_record_from_msgpack(filename, rank=0, step=step, name="rebalance")
        return data[0]

    def get_diag_handler(self, prefix):
        return self.diag_handlers[prefix]

    def get_step(self, prefix):
        return self.diag_handlers[prefix].get_step()

    def get_time(self, prefix):
        return self.diag_handlers[prefix].get_time()

    def get_time_at(self, prefix, step):
        return self.diag_handlers[prefix].get_time_at_step(step)

    def read_at(self, prefix, step, pattern=None):
        # return cache if exists
        if prefix in self.cache:
            cache = self.cache[prefix]
            cache_step = cache.get("step", None)
            cache_pattern = cache.get("pattern", None)
            cache_data = cache.get("data", None)
            if (
                cache_data is not None
                and cache_step == step
                and cache_pattern == pattern
            ):
                return cache_data

        # otherwise, read data
        handler = self.diag_handlers[prefix]

        # read data
        data = dict()
        for jsonfile in handler.find_json_at_step(step):
            chunk_split_data = self.read_at_single(jsonfile, pattern)
            for key, val in chunk_split_data.items():
                if not key in data:
                    data[key] = []
                data[key].append(val)

        # concatenate to global array
        for key, val in data.items():
            data[key] = np.concatenate(val, axis=0)

        # convert array format
        if handler.is_chunked_array_conversion_required():
            data = convert_array_format(data, self.chunkmap)

        # store cache
        self.cache[prefix] = {"step": step, "pattern": pattern, "data": data}

        return data

    def read_at_single(self, jsonfile, pattern=None):
        if pattern is None:
            pattern = ".*"

        with open(jsonfile, "r") as fp:
            obj = json.load(fp)
            byteorder, datafile, order = get_json_meta(obj)
            dataset = obj["dataset"]

        # read data
        datafile = str(pathlib.Path(jsonfile).parent / datafile)
        with open(datafile, "r") as fp:
            data = {}
            for dsname in dataset:
                if not re.match(pattern, dsname):
                    continue
                offset, dtype, shape = get_dataset_info(dataset.get(dsname), byteorder)
                if order == 0:
                    shape == shape[::-1]
                data[dsname] = get_dataset_data(fp, offset, dtype, shape)

        return data
