#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import glob

from picnix import (
    DEFAULT_LOG_PREFIX,
    DEFAULT_LOAD_PREFIX,
    DEFAULT_FIELD_PREFIX,
    DEFAULT_PARTICLE_PREFIX,
    DEFAULT_TRACER_PREFIX,
)


class DiagHandler(object):
    def __init__(self, name, prefix, basedir, iomode):
        self.name = name
        self.prefix = prefix
        self.basedir = basedir
        self.iomode = iomode

    def match(self, name):
        return name == self.name

    def get_name(self):
        return self.name

    def get_prefix(self):
        return self.prefix

    def is_chunked_array_conversion_required(self):
        return False

    def setup(self, config):
        self.config = config
        prefix = config.get("prefix", self.prefix)
        dirname = self.format_dirname(prefix)
        pattern = self.pattern_filename("", ".json")
        self.file = sorted(glob.glob(dirname + pattern))
        self.step = np.arange(len(self.file), dtype=np.int32)
        self.time = np.arange(len(self.file), dtype=np.float64)
        for i, file in enumerate(self.file):
            with open(file, "r") as fp:
                obj = json.load(fp)
                self.step[i] = obj["meta"]["step"]
                self.time[i] = obj["meta"]["time"]

    def format_dirname(self, prefix):
        dirname = os.sep.join([self.basedir, prefix]) + os.sep
        if self.iomode == "mpiio":
            return dirname
        elif self.iomode == "posix":
            return dirname

    def format_filename(self, prefix, ext, step):
        return prefix + "{:08d}".format(step) + ext

    def pattern_filename(self, prefix, ext):
        return prefix + "*" + ext

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
            return self.file[index]
        else:
            return None

    @staticmethod
    def create_handler(config, basedir, iomode):
        if "name" not in config:
            return None
        # create handler
        if config["name"] == "load":
            prefix = config.get("prefix", DEFAULT_LOAD_PREFIX)
            handler = LoadDiagHandler(prefix, basedir, iomode)
            handler.setup(config)
            return handler
        elif config["name"] == "field":
            prefix = config.get("prefix", DEFAULT_FIELD_PREFIX)
            handler = FieldDiagHandler(prefix, basedir, iomode)
            handler.setup(config)
            return handler
        elif config["name"] == "particle":
            prefix = config.get("prefix", DEFAULT_PARTICLE_PREFIX)
            handler = ParticleDiagHandler(prefix, basedir, iomode)
            handler.setup(config)
            return handler
        elif config["name"] == "tracer":
            prefix = config.get("prefix", DEFAULT_TRACER_PREFIX)
            handler = TracerDiagHandler(prefix, basedir, iomode)
            handler.setup(config)
            return handler
        else:
            return None


class LoadDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("load", prefix, basedir, iomode)


class FieldDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("field", prefix, basedir, iomode)

    def is_chunked_array_conversion_required(self):
        return True


class ParticleDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("particle", prefix, basedir, iomode)


class TracerDiagHandler(DiagHandler):
    def __init__(self, prefix, basedir, iomode):
        super().__init__("tracer", prefix, basedir, iomode)
