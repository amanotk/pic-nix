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

        # default pattern (read everything)
        if pattern is None:
            pattern = ".*"

        # otherwise, read data
        handler = self.diag_handlers[prefix]
        data = dict()

        if handler.is_chunked_data_shape_uniform():
            data = self.read_at_uniform(handler, step, pattern)
        else:
            data = self.read_at_general(handler, step, pattern)

        # convert array format
        if handler.is_chunked_data_conversion_required():
            data = convert_array_format(data, self.chunkmap)

        # store cache
        self.cache[prefix] = {"step": step, "pattern": pattern, "data": data}

        return data

    def prepare_read(self, all_jsonfiles, pattern):
        dims = dict()
        dtype = dict()
        names = []

        # preparation
        dataset, meta = read_jsonfile(all_jsonfiles[0])
        for key in dataset.keys():
            if not re.match(pattern, key):
                continue
            names.append(key)
            ndim = dataset[key]["ndim"]
            dtype[key] = dataset[key]["datatype"]
            dims[key] = np.zeros((len(all_jsonfiles), ndim), dtype=np.int32)

        return dims, dtype, names

    def read_json_files(self, all_jsonfiles, dataset_names, dims):
        json_contents = [0] * len(all_jsonfiles)
        for i, jsonfile in enumerate(all_jsonfiles):
            json_contents[i] = read_jsonfile(jsonfile)
            dataset, meta = json_contents[i]
            for key in dataset_names:
                dims[key][i, :] = dataset[key]["shape"]

        return json_contents

    def allocate_data_memory(self, dataset_names, dims, dtype):
        data = dict()
        for key in dataset_names:
            dshape = (np.sum(dims[key][:, 0]), *dims[key][0, 1:])
            data[key] = np.zeros(dshape, dtype=dtype[key])

        return data

    def read_at_uniform(self, handler, step, pattern):
        all_jsonfiles = handler.find_json_at_step(step)
        num_jsonfiles = len(all_jsonfiles)

        dims, dtype, names = self.prepare_read(all_jsonfiles, pattern)
        json_contents = self.read_json_files(all_jsonfiles, names, dims)
        result = self.allocate_data_memory(names, dims, dtype)

        # read data
        for i in range(num_jsonfiles):
            dataset, meta = json_contents[i]
            chunk_data = read_datafile(dataset, meta, pattern)
            for key in names:
                chunk_id_range = meta["chunk_id_range"]
                chunk_slice = slice(chunk_id_range[0], chunk_id_range[1] + 1)
                result[key][chunk_slice, ...] = chunk_data[key]

        return result

    def read_at_general(self, handler, step, pattern):
        all_jsonfiles = handler.find_json_at_step(step)
        num_jsonfiles = len(all_jsonfiles)

        dims, dtype, names = self.prepare_read(all_jsonfiles, pattern)
        json_contents = self.read_json_files(all_jsonfiles, names, dims)
        result = self.allocate_data_memory(names, dims, dtype)

        # read data
        addr = {key: 0 for key in names}
        for i in range(num_jsonfiles):
            dataset, meta = json_contents[i]
            chunk_data = read_datafile(dataset, meta, pattern)
            for key in names:
                count = chunk_data[key].shape[0]
                chunk_slice = slice(addr[key], addr[key] + count)
                result[key][chunk_slice, ...] = chunk_data[key]
                addr[key] += count

        return result
