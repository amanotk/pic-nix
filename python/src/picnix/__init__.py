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
