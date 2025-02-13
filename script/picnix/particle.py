#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import tqdm


def sort_and_split_particle_id(particle):
    if particle.ndim != 2 or particle.shape[1] != 7:
        raise ValueError("Invalid particle array")

    pshape = particle.shape
    pid = np.frombuffer(particle.tobytes(), dtype=np.int64).reshape(pshape)[:, 6]
    index = np.argsort(-pid)

    return particle[index, 0:6], pid[index]


def is_valid_tracer_hdf5(hdffile, name):
    with h5py.File(hdffile, "r") as fp:
        has_group = name in fp
        has_time = has_group and "time" in fp[name]
        has_step = has_group and "step" in fp[name]
        status = has_group and has_time and has_step
    return status


def convert_tracer_to_hdf5(run, species, hdffile):
    group = "up{:02d}".format(species)

    if os.path.exists(hdffile):
        # check if the file is valid
        if is_valid_tracer_hdf5(hdffile, group) == False:
            raise ValueError("Invalid file: {}".format(hdffile))
    else:
        # new file
        with h5py.File(hdffile, "w") as fp:
            root = fp.create_group(group)
            root.create_dataset("step", (0,), maxshape=(None,), dtype=np.int32)
            root.create_dataset("time", (0,), maxshape=(None,), dtype=np.float64)
            root.create_group("xp")
            root.create_group("id")

    tracer_time = run.get_time("tracer")
    tracer_step = run.get_step("tracer")

    for i, step in enumerate(tqdm.tqdm(tracer_step)):
        # read data
        try:
            data = run.read_at("tracer", step)[group]
            data_xp, data_id = sort_and_split_particle_id(data)
        except Exception as e:
            print("Error at step: {:08d}".format(step))
            print(e)
            continue

        with h5py.File(hdffile, "a") as fp:
            root = fp[group]
            ds_name = "{:08d}".format(step)
            ds_step = root["step"]
            ds_time = root["time"]
            group_xp = root["xp"]
            group_id = root["id"]

            if np.any(ds_step[()] == step):
                # ignore
                print("Skipping data at step: {:08d}".format(step))
                continue

            # expand size of step and time
            ds_step.resize((len(ds_step) + 1,))
            ds_time.resize((len(ds_time) + 1,))
            ds_step[-1] = tracer_step[i]
            ds_time[-1] = tracer_time[i]

            # create dataset
            group_xp.create_dataset(ds_name, data=data_xp)
            group_id.create_dataset(ds_name, data=data_id)


def remove_tracer_file_after_confirmation(run, species, hdffile):
    group = "up{:02d}".format(species)

    if not (os.path.exists(hdffile) and is_valid_tracer_hdf5(hdffile, group)):
        raise ValueError("Invalid file: {}".format(hdffile))

    status = True
    tracer_time = run.get_time("tracer")
    tracer_step = run.get_step("tracer")

    with h5py.File(hdffile, "r") as fp:
        root = fp[group]
        ds_step = root["step"][()]
        ds_time = root["time"][()]
        group_xp = root["xp"]
        group_id = root["id"]

        for i, step in enumerate(tqdm.tqdm(tracer_step)):
            # read data
            try:
                data = run.read_at("tracer", step)[group]
                data_xp, data_id = sort_and_split_particle_id(data)
            except Exception as e:
                print("Error at step: {}".format(step))
                print(e)
                continue

            # check consistency of data
            name = "{:08d}".format(step)
            index = np.searchsorted(ds_step, tracer_step[i])
            is_step_valid = ds_step[index] == tracer_step[i]
            is_time_valid = ds_time[index] == tracer_time[i]
            is_xp_valid = np.all(group_xp[name] == data_xp)
            is_id_valid = np.all(group_id[name] == data_id)

            is_everything_okay = (
                is_step_valid and is_time_valid and is_xp_valid and is_id_valid
            )

            if is_everything_okay == False:
                status = False
                print("Data at step: {:08d} is invalid".format(step))

    if status:
        print("")
        print("")
        print("")
        print("The original data is consistent with HDF5 file {}".format(hdffile))
        print("")
        print("")
        print("")

        CONFIRMATION_ENV = "PICNIX_REMOVE_ORIGINAL_TRACER_FILES"

        if os.environ.get(CONFIRMATION_ENV) != "YES":
            print("If you want to remove original files, set environment variable")
            print("{} to 'YES' and run again!".format(CONFIRMATION_ENV))
            print("")
            return
        else:
            print("Removing original files...")
            handler = run.get_diag_handler("tracer")

            for step in tracer_step:
                json_files = handler.find_json_at_step(step)
                data_files = [f.replace(".json", ".data") for f in json_files]
                for fn_json, fn_data in zip(json_files, data_files):
                    os.remove(fn_json)
                    os.remove(fn_data)
            print("Done!")


class Tracer:
    def __init__(self, hdffile, species=0):
        self.hdffile = hdffile
        self.species = species
        self.group = "up{:02d}".format(species)
        self.time, self.step = self.read_time_and_step(species)

    def read_time_and_step(self, species):
        with h5py.File(self.hdffile, "r") as fp:
            root = fp[self.group]
            time = root["time"][()]
            step = root["step"][()]
        return time, step

    def get_time(self):
        return self.time

    def get_step(self):
        return self.step

    def get_id_at(self, step):
        name = "{:08d}".format(step)

        with h5py.File(self.hdffile, "r") as fp:
            root = fp[self.group]
            data = root["id"][name][()]
        return data

    def get_xp_at(self, step):
        name = "{:08d}".format(step)

        with h5py.File(self.hdffile, "r") as fp:
            root = fp[self.group]
            data = root["xp"][name][()]
        return data

    def get_xp_all(self, id):
        N = self.step.size
        xp = np.zeros((N, id.size, 6), dtype=np.float64)

        with h5py.File(self.hdffile, "r") as fp:
            root = fp[self.group]

            for i in range(N):
                step = self.step[i]
                name = "{:08d}".format(step)
                data_id = root["id"][name][()]
                data_xp = root["xp"][name][()]
                # assume IDs are all negative and sorted
                index = np.searchsorted(-data_id, -id)
                data_id = np.take(data_id, index, mode="clip")
                data_xp = np.take(data_xp, index, mode="clip", axis=0)
                # select data and fill with NaN if not found
                xp[i, :, :] = np.where(
                    data_id[:, np.newaxis] == id[:, np.newaxis],
                    data_xp,
                    np.nan,
                )
        return xp


class Histogram2D:
    def __init__(self, x, y, binx, biny, logx=False, logy=False):
        binx = self.handle_bin_arg(binx, logx)
        biny = self.handle_bin_arg(biny, logy)
        result = np.histogram2d(x, y, bins=(binx, biny))
        # count for each bin
        self.count = result[0]
        self.xedges = result[1]
        self.yedges = result[2]
        # density (count per area)
        deltax = np.diff(self.xedges)
        deltay = np.diff(self.yedges)
        self.area = deltax[:, np.newaxis] * deltay[np.newaxis, :]
        self.density = self.count / self.area

    def handle_bin_arg(self, bin, logscale=False):
        if isinstance(bin, tuple) and logscale == False:
            return np.linspace(bin[0], bin[1], bin[2])
        if isinstance(bin, tuple) and logscale == True:
            return np.geomspace(bin[0], bin[1], bin[2])
        if isinstance(bin, np.ndarray) and bin.ndim == 1:
            return bin
        raise ValueError("Invalid argument")

    def pcolormesh_args(self, density=True):
        x = 0.5 * (self.xedges[+1:] + self.xedges[:-1])
        y = 0.5 * (self.yedges[+1:] + self.yedges[:-1])
        Z = self.density if density == True else self.count
        X, Y = np.broadcast_arrays(x[:, None], y[None, :])
        return X, Y, Z
