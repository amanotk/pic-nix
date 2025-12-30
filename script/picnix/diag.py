#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiofiles
import nest_asyncio
import numpy as np

from picnix import (
    DEFAULT_FIELD_PREFIX,
    DEFAULT_LOAD_PREFIX,
    DEFAULT_PARTICLE_PREFIX,
    DEFAULT_TRACER_PREFIX,
)


class DiagHandler(object):
    def __init__(self, name, prefix, basedir, iomode):
        self.name = name
        self.prefix = prefix
        self.basedir = basedir
        self.iomode = iomode
        self.file_pattern = re.compile(r"\d+\.json$")
        self.node_pattern = re.compile(r"node\d+$")

    def match(self, name):
        return name == self.name

    def get_name(self):
        return self.name

    def get_prefix(self):
        return self.prefix

    def is_chunked_data_conversion_required(self):
        return False

    def is_chunked_data_shape_uniform(self):
        return False

    def append_auxiliary_data(self, data, config_root):
        return data

    def setup(self, config):
        self.config = config
        self.file = self.get_file_array(config.get("prefix", self.prefix))
        self.step = np.arange(self.file.shape[1], dtype=np.int32)
        self.time = np.arange(self.file.shape[1], dtype=np.float64)

        # read time and step
        for i, file in enumerate(self.file[0, :]):
            self.step[i], self.time[i] = DiagHandler.read_time_and_step(file)

    def thread_setup(self, config):
        self.config = config
        self.file = self.get_file_array(config.get("prefix", self.prefix))
        self.step = np.arange(self.file.shape[1], dtype=np.int32)
        self.time = np.arange(self.file.shape[1], dtype=np.float64)

        # read time and step via thread
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            for i, file in enumerate(self.file[0, :]):
                future = executor.submit(DiagHandler.read_time_and_step, file)
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    self.step[i], self.time[i] = future.result()
                except Exception as exc:
                    print(f"File at index {i} generated an exception: {exc}")

    async def async_setup(self, config):
        self.config = config
        self.file = self.get_file_array(config.get("prefix", self.prefix))
        self.step = np.arange(self.file.shape[1], dtype=np.int32)
        self.time = np.arange(self.file.shape[1], dtype=np.float64)

        # read time and step via asyncio
        tasks = []
        for i, file in enumerate(self.file[0, :]):
            tasks.append(asyncio.create_task(DiagHandler.async_read_time_and_step(file)))
        result = await asyncio.gather(*tasks)

        for i in range(len(result)):
            self.step[i], self.time[i] = result[i]

    def get_matching_jsons(self, dirname):
        if not os.path.isdir(dirname):
            return []
        files = []
        for f in os.listdir(dirname):
            if self.file_pattern.match(f):
                files.append(os.path.join(dirname, f))
        return sorted(files)

    def get_matching_nodes(self, dirname):
        if not os.path.isdir(dirname):
            return []
        nodes = []
        for f in os.listdir(dirname):
            if self.node_pattern.match(f):
                nodes.append(os.path.join(dirname, f))
        return sorted(nodes)

    def get_file_array(self, prefix):
        if self.iomode == "mpiio":
            dirname = os.sep.join([self.basedir, prefix])
            file = self.get_matching_jsons(dirname)
            return np.array(file).reshape((1, len(file)))
        elif self.iomode == "posix":
            nodedir = self.get_matching_nodes(self.basedir)
            nodenum = len(nodedir)
            file = [0] * nodenum
            for i in range(nodenum):
                dirname = os.sep.join([nodedir[i], prefix])
                file[i] = self.get_matching_jsons(dirname)
            return np.array(file)

    def find_index_at_step(self, step):
        index = np.searchsorted(self.step, step)
        if step == self.step[index]:
            return index
        else:
            return None

    def get_step(self):
        return self.step

    def get_time(self):
        return self.time

    def get_time_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is not None:
            return self.time[index]
        else:
            return None

    def find_json_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is not None:
            return self.file[:, index]
        else:
            return None

    def remove_json_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is None:
            return

        # remove entries at index
        self.file = np.delete(self.file, index, axis=1)
        self.step = np.delete(self.step, index)
        self.time = np.delete(self.time, index)

    @staticmethod
    def read_time_and_step(filename):
        with open(filename, "r") as fp:
            obj = json.load(fp)
            step = obj["meta"]["step"]
            time = obj["meta"]["time"]
        return step, time

    @staticmethod
    async def async_read_time_and_step(filename):
        async with aiofiles.open(filename, mode="r") as fp:
            content = await fp.read()
            obj = json.loads(content)
            step = obj["meta"]["step"]
            time = obj["meta"]["time"]
        return step, time

    @staticmethod
    def create_handler(config, basedir, iomode, method=None):
        if "name" not in config:
            return None

        # create handler
        if config["name"] == "load":
            prefix = config.get("prefix", DEFAULT_LOAD_PREFIX)
            handler = LoadDiagHandler(prefix, basedir, iomode)
        elif config["name"] == "field":
            prefix = config.get("prefix", DEFAULT_FIELD_PREFIX)
            handler = FieldDiagHandler(prefix, basedir, iomode)
        elif config["name"] == "particle":
            prefix = config.get("prefix", DEFAULT_PARTICLE_PREFIX)
            handler = ParticleDiagHandler(prefix, basedir, iomode)
        elif config["name"] == "tracer":
            prefix = config.get("prefix", DEFAULT_TRACER_PREFIX)
            handler = TracerDiagHandler(prefix, basedir, iomode)
        else:
            return None

        # setup handler
        if method == "async":
            nest_asyncio.apply()  # for jupyter notebook

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(handler.async_setup(config))
        elif method == "thread":
            handler.thread_setup(config)
        else:
            handler.setup(config)

        return handler


class LoadDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("load", prefix, basedir, iomode)

    def is_chunked_data_shape_uniform(self):
        return True


class FieldDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("field", prefix, basedir, iomode)

    def is_chunked_data_conversion_required(self):
        return True

    def is_chunked_data_shape_uniform(self):
        return True

    def append_auxiliary_data(self, data, config_root):
        decimate = self.config.get("decimate", 1)
        parameter = config_root["parameter"]

        nx = self.calc_decimated_grid(parameter["Nx"], parameter["Cx"], decimate)
        ny = self.calc_decimated_grid(parameter["Ny"], parameter["Cy"], decimate)
        nz = self.calc_decimated_grid(parameter["Nz"], parameter["Cz"], decimate)
        dx = (parameter["Nx"] // nx) * parameter["delh"]
        dy = (parameter["Ny"] // ny) * parameter["delh"]
        dz = (parameter["Nz"] // nz) * parameter["delh"]

        data["xc"] = dx * (np.arange(nx) + 0.5)
        data["yc"] = dy * (np.arange(ny) + 0.5)
        data["zc"] = dz * (np.arange(nz) + 0.5)

        return data

    def calc_decimated_grid(self, num_grid, num_chunk, decimate):
        m = num_grid // num_chunk

        if m <= decimate:
            grid_per_chunk = 1
        elif (decimate < 0) or (decimate > 0 and m % decimate == 0):
            grid_per_chunk = m // decimate
        else:
            # something wrong
            raise ValueError("Invalid decimate value: {}".format(decimate))

        return grid_per_chunk * num_chunk


class ParticleDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("particle", prefix, basedir, iomode)


class TracerDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("tracer", prefix, basedir, iomode)
