#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import msgpack
import nest_asyncio
import numpy as np
import toml

from picnix import (
    DEFAULT_LOG_PREFIX,
)

from .diag import DiagHandler
from .utils import *


class Run(object):
    def __init__(self, profile, method=None, config=None):
        self.cache = dict()
        self.profile = profile
        self.method = method
        self.read_profile(profile, config)
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

    def read_profile(self, profile, config=None):
        # read profile
        with open(profile, "rb") as fp:
            obj = msgpack.load(fp)
            self.timestamp = obj["timestamp"]
            self.nprocess = obj["nprocess"]
            self.chunkmap = obj["chunkmap"]
            config_in_profile = obj["configuration"]
        # read given config if specified
        if config is None:
            self.config = config_in_profile
        elif config.endswith(".toml"):
            with open(config, "r") as fileobj:
                self.config = toml.load(fileobj)
        elif config.endswith(".json"):
            with open(config, "r") as fileobj:
                self.config = json.load(fileobj)

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
            handler = DiagHandler.create_handler(diagnostic, basedir, iomode, self.method)
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
            if cache_data is not None and cache_step == step and cache_pattern == pattern:
                return cache_data

        # default pattern (read everything)
        if pattern is None:
            pattern = ".*"

        # otherwise, read data
        handler = self.diag_handlers[prefix]

        if self.method == "async":
            nest_asyncio.apply()  # for jupyter notebook

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            data = loop.run_until_complete(Run.async_do_read_at(handler, step, pattern))
        elif self.method == "thread":
            data = Run.thread_do_read_at(handler, step, pattern)
        else:
            data = Run.do_read_at(handler, step, pattern)

        # convert array format
        if handler.is_chunked_data_conversion_required():
            data = convert_array_format(data, self.chunkmap)

        # append auxiliary data if needed
        data = handler.append_auxiliary_data(data, self.config)

        # store cache
        self.cache[prefix] = {"step": step, "pattern": pattern, "data": data}

        return data

    def remove_file_at(self, prefix, step, are_you_sure=False):
        handler = self.diag_handlers[prefix]
        if not are_you_sure:
            print(
                "*** WARNING ***\n"
                "This will remove files for prefix {:s} at step {:d}\n"
                "If you really want to do this, please set\n"
                "   `are_you_sure=True`   \n"
                "and run again!\n".format(prefix, step)
            )
            return

        # now remove files
        jsonfile_to_be_removed = handler.find_json_at_step(step)
        for jsonfile in jsonfile_to_be_removed:
            _, meta = read_jsonfile(jsonfile)
            datafile = os.sep.join([meta["dirname"], meta["datafile"]])
            try:
                os.remove(jsonfile)
                os.remove(datafile)
                self.diag_handlers[prefix].remove_json_at_step(step)
            except OSError as e:
                print(f"Error removing file {jsonfile} or {datafile}: {e}")

    @staticmethod
    def prepare_read(all_jsonfiles, pattern):
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

    @staticmethod
    def allocate_memory(names, dims, dtype):
        data = dict()
        addr = dict()
        for key in names:
            # storage
            dshape = (np.sum(dims[key][:, 0]), *dims[key][0, 1:])
            data[key] = np.zeros(dshape, dtype=dtype[key])
            # address
            addr[key] = np.zeros((dims[key].shape[0] + 1,), dtype=np.int32)
            addr[key][1:] = np.cumsum(dims[key][:, 0])

        return data, addr

    @staticmethod
    def read_json_files(all_json, names, dims):
        json_contents = [0] * len(all_json)
        for i, jsonfile in enumerate(all_json):
            json_contents[i] = read_jsonfile(jsonfile)
            dataset, meta = json_contents[i]
            for key in names:
                dims[key][i, :] = dataset[key]["shape"]
        return json_contents, dims

    @staticmethod
    def read_data_files(result, address, json_contents, names, pattern):
        for i, (dataset, meta) in enumerate(json_contents):
            chunk_data = read_datafile(dataset, meta, pattern)
            for key in names:
                chunk_slice = slice(address[key][i], address[key][i + 1])
                result[key][chunk_slice, ...] = chunk_data[key]

        return result

    @staticmethod
    def do_read_at(handler, step, pattern):
        all_json = handler.find_json_at_step(step)
        dims, dtype, names = Run.prepare_read(all_json, pattern)

        # read json
        json_contents, dims = Run.read_json_files(all_json, names, dims)

        # allocate memory
        result, address = Run.allocate_memory(names, dims, dtype)

        # read data
        result = Run.read_data_files(result, address, json_contents, names, pattern)

        return result

    @staticmethod
    def thread_read_json_files(all_json, names, dims):
        json_contents = [0] * len(all_json)

        # read json files via thread
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            for i, jsonfile in enumerate(all_json):
                future = executor.submit(read_jsonfile, jsonfile)
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    json_contents[i] = future.result()
                except Exception as exc:
                    print(f"JSON file at index {i} generated an exception: {exc}")

        # store dimensions
        for i, (dataset, meta) in enumerate(json_contents):
            for key in names:
                dims[key][i, :] = dataset[key]["shape"]

        return json_contents, dims

    @staticmethod
    def thread_read_data_files(result, address, json_contents, names, pattern):
        # function for read and store data
        def process_datafile(dataset, meta, i):
            chunk_data = read_datafile(dataset, meta, pattern)
            for key in names:
                chunk_slice = slice(address[key][i], address[key][i + 1])
                result[key][chunk_slice, ...] = chunk_data[key]

        # read and store data via thread
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, (dataset, meta) in enumerate(json_contents):
                futures.append(executor.submit(process_datafile, dataset, meta, i))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred while processing a file: {exc}")

        return result

    @staticmethod
    def thread_do_read_at(handler, step, pattern):
        all_json = handler.find_json_at_step(step)
        dims, dtype, names = Run.prepare_read(all_json, pattern)

        # read json
        json_contents, dims = Run.thread_read_json_files(all_json, names, dims)

        # allocate memory
        result, address = Run.allocate_memory(names, dims, dtype)

        # read data
        result = Run.thread_read_data_files(result, address, json_contents, names, pattern)

        return result

    @staticmethod
    async def async_read_json_files(all_json, names, dims):
        # read json files via asyncio
        tasks = []
        for jsonfile in all_json:
            tasks.append(asyncio.create_task(async_read_jsonfile(jsonfile)))
        json_contents = await asyncio.gather(*tasks)

        # store dimensions
        for i, (dataset, meta) in enumerate(json_contents):
            for key in names:
                dims[key][i, :] = dataset[key]["shape"]

        return json_contents, dims

    @staticmethod
    async def async_read_data_files(result, address, json_contents, names, pattern):
        # async function for read and store data
        async def process_datafile(dataset, meta, i):
            chunk_data = await async_read_datafile(dataset, meta, pattern)
            for key in names:
                chunk_slice = slice(address[key][i], address[key][i + 1])
                result[key][chunk_slice, ...] = chunk_data[key]

        # read and store data via asyncio
        tasks = []
        for i, (dataset, meta) in enumerate(json_contents):
            tasks.append(asyncio.create_task(process_datafile(dataset, meta, i)))

        await asyncio.gather(*tasks)

        return result

    @staticmethod
    async def async_do_read_at(handler, step, pattern):
        all_json = handler.find_json_at_step(step)
        dims, dtype, names = Run.prepare_read(all_json, pattern)

        # read json
        json_contents, dims = await Run.async_read_json_files(all_json, names, dims)

        # allocate memory
        result, address = Run.allocate_memory(names, dims, dtype)

        # read data
        result = await Run.async_read_data_files(result, address, json_contents, names, pattern)

        return result
