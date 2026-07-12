#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import h5py
import numpy as np

from .utils import read_datafile, read_jsonfile


class DiagStorage:
    kind = "base"

    def get_step(self):
        return self.step

    def get_time(self):
        return self.time

    def find_index_at_step(self, step):
        index = np.searchsorted(self.step, step)
        if index < self.step.size and step == self.step[index]:
            return index
        return None

    def get_time_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is None:
            return None
        return self.time[index]

    def read_particle_at(self, step, pattern, start=None, stop=None):
        return {}

    def read_particle_id_at(self, step, pattern, start=None, stop=None):
        return {}


class JsonDiagStorage(DiagStorage):
    kind = "json"

    def __init__(self, name, prefix, basedir, iomode):
        self.name = name
        self.prefix = prefix
        self.basedir = Path(basedir)
        self.iomode = iomode
        self.file_pattern = re.compile(r"\d+\.json$")
        self.node_pattern = re.compile(r"node\d+$")

    def setup(self):
        self.file = self.get_file_array()
        if self.file.ndim != 2 or self.file.shape[1] == 0:
            raise FileNotFoundError(
                f"no JSON diagnostic files found for prefix {self.prefix!r} in {self.basedir}"
            )
        self.step = np.arange(self.file.shape[1], dtype=np.int32)
        self.time = np.arange(self.file.shape[1], dtype=np.float64)
        for i, filename in enumerate(self.file[0, :]):
            self.step[i], self.time[i] = self.read_time_and_step(filename)

    def get_matching_jsons(self, dirname):
        if not dirname.is_dir():
            return []
        return sorted(
            str(path)
            for path in dirname.iterdir()
            if self.file_pattern.match(path.name)
        )

    def get_matching_nodes(self):
        if not self.basedir.is_dir():
            return []
        return sorted(
            path
            for path in self.basedir.iterdir()
            if path.is_dir() and self.node_pattern.match(path.name)
        )

    def get_file_array(self):
        if self.iomode == "mpiio":
            files = self.get_matching_jsons(self.basedir / self.prefix)
            return np.array(files).reshape((1, len(files)))
        if self.iomode == "posix":
            nodes = self.get_matching_nodes()
            files = [self.get_matching_jsons(node / self.prefix) for node in nodes]
            return np.array(files)
        raise ValueError(f"unsupported iomode: {self.iomode}")

    def find_json_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is None:
            return None
        return self.file[:, index]

    def remove_json_at_step(self, step):
        index = self.find_index_at_step(step)
        if index is None:
            return
        self.file = np.delete(self.file, index, axis=1)
        self.step = np.delete(self.step, index)
        self.time = np.delete(self.time, index)

    @staticmethod
    def read_time_and_step(filename):
        with open(filename, "r") as fp:
            obj = json.load(fp)
        return obj["meta"]["step"], obj["meta"]["time"]

    @staticmethod
    def prepare_read(all_jsonfiles, pattern):
        dims = {}
        dtype = {}
        names = []
        dataset, _ = read_jsonfile(all_jsonfiles[0])
        for key in dataset:
            if not re.match(pattern, key):
                continue
            names.append(key)
            ndim = dataset[key]["ndim"]
            dtype[key] = dataset[key]["datatype"]
            dims[key] = np.zeros((len(all_jsonfiles), ndim), dtype=np.int32)
        return dims, dtype, names

    @staticmethod
    def read_json_files(all_json, names, dims):
        json_contents = [None] * len(all_json)
        for i, jsonfile in enumerate(all_json):
            json_contents[i] = read_jsonfile(jsonfile)
            dataset, _ = json_contents[i]
            for key in names:
                dims[key][i, :] = dataset[key]["shape"]
        return json_contents, dims

    @staticmethod
    def allocate_memory(names, dims, dtype):
        data = {}
        address = {}
        for key in names:
            dshape = (np.sum(dims[key][:, 0]), *dims[key][0, 1:])
            data[key] = np.zeros(dshape, dtype=dtype[key])
            address[key] = np.zeros((dims[key].shape[0] + 1,), dtype=np.int32)
            address[key][1:] = np.cumsum(dims[key][:, 0])
        return data, address

    @staticmethod
    def read_data_files(result, address, json_contents, names, pattern):
        for i, (dataset, meta) in enumerate(json_contents):
            chunk_data = read_datafile(dataset, meta, pattern)
            for key in names:
                chunk_slice = slice(address[key][i], address[key][i + 1])
                result[key][chunk_slice, ...] = chunk_data[key]
        return result

    @staticmethod
    def normalize_range(start, stop, size):
        return slice(start, stop).indices(size)[:2]

    @staticmethod
    def read_particle_rows(dataset, meta, name, start, stop):
        byteorder = meta["byteorder"]
        layout = meta["layout"]
        datafile = Path(meta["dirname"]) / meta["datafile"]
        offset = dataset[name]["offset"]
        dtype = np.dtype(byteorder + dataset[name]["datatype"])
        shape = tuple(dataset[name]["shape"])
        if layout == 0:
            shape = shape[::-1]
        if len(shape) != 2:
            raise ValueError(f"particle dataset {name!r} must be 2-D, got {shape}")
        row_shape = shape[1:]
        row_count = stop - start
        row_size = int(np.prod(row_shape))
        with open(datafile, "rb") as fp:
            fp.seek(offset + start * row_size * dtype.itemsize)
            values = np.fromfile(fp, dtype, row_count * row_size)
        return values.reshape((row_count, *row_shape))

    @staticmethod
    def particle_dataset_shape(dataset, meta, name):
        shape = tuple(dataset[name]["shape"])
        if meta["layout"] == 0:
            shape = shape[::-1]
        return shape

    def read_particle_range_at(self, step, pattern, start, stop, include_id):
        if self.name != "particle":
            return {}
        all_json = self.find_json_at_step(step)
        if all_json is None:
            return {}
        dims, _, names = self.prepare_read(all_json, pattern)
        json_contents, dims = self.read_json_files(all_json, names, dims)
        result = {}
        for name in names:
            shapes = [
                self.particle_dataset_shape(dataset, meta, name)
                for dataset, meta in json_contents
            ]
            total = int(np.sum([shape[0] for shape in shapes]))
            range_start, range_stop = self.normalize_range(start, stop, total)
            if range_start >= range_stop:
                dtype = np.dtype(
                    json_contents[0][1]["byteorder"]
                    + json_contents[0][0][name]["datatype"]
                )
                result[name] = (
                    np.empty((0,), dtype=np.uint64)
                    if include_id
                    else np.empty((0, shapes[0][1] - 1), dtype=dtype)
                )
                continue
            chunks = []
            address = np.zeros((len(shapes) + 1,), dtype=np.int64)
            address[1:] = np.cumsum([shape[0] for shape in shapes])
            for i, (dataset, meta) in enumerate(json_contents):
                local_start = max(range_start, address[i]) - address[i]
                local_stop = min(range_stop, address[i + 1]) - address[i]
                if local_start >= local_stop:
                    continue
                rows = self.read_particle_rows(
                    dataset, meta, name, int(local_start), int(local_stop)
                )
                if include_id:
                    chunks.append(reinterpret_particle_id(rows[:, -1]))
                else:
                    chunks.append(rows[:, :-1])
            result[name] = np.concatenate(chunks, axis=0)
        return result

    def read_raw_at(self, step, pattern):
        all_json = self.find_json_at_step(step)
        if all_json is None:
            return {}
        dims, dtype, names = self.prepare_read(all_json, pattern)
        json_contents, dims = self.read_json_files(all_json, names, dims)
        result, address = self.allocate_memory(names, dims, dtype)
        return self.read_data_files(result, address, json_contents, names, pattern)

    def read_at(self, step, pattern):
        data = self.read_raw_at(step, pattern)
        if self.name != "particle":
            return data
        return {name: values[:, :-1] for name, values in data.items()}

    def read_particle_at(self, step, pattern, start=None, stop=None):
        return self.read_particle_range_at(step, pattern, start, stop, include_id=False)

    def read_particle_id_at(self, step, pattern, start=None, stop=None):
        return self.read_particle_range_at(step, pattern, start, stop, include_id=True)


class Hdf5VdsDiagStorage(DiagStorage):
    kind = "hdf5-vds"

    def __init__(self, name, prefix, vds_path):
        self.name = name
        self.prefix = prefix
        self.vds_path = Path(vds_path)

    def setup(self):
        with h5py.File(self.vds_path, "r") as h5fp:
            layout = h5fp.attrs.get("picnix_hdf5_layout")
            if layout != "prefix-vds-v1":
                raise ValueError(f"invalid PIC-NIX HDF5 VDS layout in {self.vds_path}")
            prefix = h5fp.attrs.get("prefix")
            if prefix != self.prefix:
                raise ValueError(
                    f"HDF5 VDS prefix mismatch in {self.vds_path}: {prefix!r} != {self.prefix!r}"
                )
            if self.prefix not in h5fp:
                raise ValueError(
                    f"missing prefix group {self.prefix!r} in {self.vds_path}"
                )
            group = h5fp[self.prefix]
            if "step" not in group or "time" not in group:
                raise ValueError(f"missing step/time datasets in {self.vds_path}")
            self.step = group["step"][...]
            self.time = group["time"][...]

    def step_group_name(self, step):
        index = self.find_index_at_step(step)
        if index is None:
            return None
        return f"{int(self.step[index]):08d}"

    def read_at(self, step, pattern):
        group_name = self.step_group_name(step)
        if group_name is None:
            return {}
        data = {}
        with h5py.File(self.vds_path, "r") as h5fp:
            group = h5fp[self.prefix][group_name]
            for name, dataset in group.items():
                if name.endswith("_id") or not re.match(pattern, name):
                    continue
                data[name] = dataset[...]
        return data

    @staticmethod
    def normalize_range(start, stop, size):
        return slice(start, stop).indices(size)[:2]

    def read_particle_at(self, step, pattern, start=None, stop=None):
        if self.name != "particle":
            return {}
        group_name = self.step_group_name(step)
        if group_name is None:
            return {}
        data = {}
        with h5py.File(self.vds_path, "r") as h5fp:
            group = h5fp[self.prefix][group_name]
            for name, dataset in group.items():
                if name.endswith("_id") or not re.match(pattern, name):
                    continue
                range_start, range_stop = self.normalize_range(
                    start, stop, dataset.shape[0]
                )
                data[name] = dataset[range_start:range_stop, ...]
        return data

    def read_particle_id_at(self, step, pattern, start=None, stop=None):
        if self.name != "particle":
            return {}
        group_name = self.step_group_name(step)
        if group_name is None:
            return {}
        data = {}
        with h5py.File(self.vds_path, "r") as h5fp:
            group = h5fp[self.prefix][group_name]
            for name, dataset in group.items():
                if not name.endswith("_id"):
                    continue
                base = name[: -len("_id")]
                if re.match(pattern, base):
                    range_start, range_stop = self.normalize_range(
                        start, stop, dataset.shape[0]
                    )
                    data[base] = dataset[range_start:range_stop]
        return data


def reinterpret_particle_id(raw_id):
    return np.ascontiguousarray(raw_id).view(np.uint64)


def create_diag_storage(name, prefix, basedir, iomode):
    vds_path = Path(basedir) / "hdf5" / f"{prefix}.vds.h5"
    if vds_path.exists():
        storage = Hdf5VdsDiagStorage(name, prefix, vds_path)
    else:
        storage = JsonDiagStorage(name, prefix, basedir, iomode)
    storage.setup()
    return storage
