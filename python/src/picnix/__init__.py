#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python Data Analysis Tool for PIC-NIX."""

from importlib import import_module

DEFAULT_LOG_PREFIX = "log"
DEFAULT_LOAD_PREFIX = "load"
DEFAULT_FIELD_PREFIX = "field"
DEFAULT_PARTICLE_PREFIX = "particle"
DEFAULT_TRACER_PREFIX = "tracer"

_LAZY_MODULES = {
    "utils": ".utils",
    "field": ".field",
    "ohm": ".ohm",
    "particle": ".particle",
    "run": ".run",
}

__all__ = [
    "DEFAULT_LOG_PREFIX",
    "DEFAULT_LOAD_PREFIX",
    "DEFAULT_FIELD_PREFIX",
    "DEFAULT_PARTICLE_PREFIX",
    "DEFAULT_TRACER_PREFIX",
    "Run",
    "Tracer",
    "Histogram2D",
    "get_wk_spectrum",
    "plot_wk_spectrum",
    "sort_and_split_particle_id",
    "is_valid_tracer_hdf5",
    "convert_tracer_to_hdf5",
    "remove_tracer_file_after_confirmation",
    "solve_ohm_1d",
    "solve_ohm_2d",
    "calc_e_ohm_1d",
    "calc_e_ohm_2d",
    "transform_moments",
]


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module

    for module_name, module_path in _LAZY_MODULES.items():
        module = import_module(module_path, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
