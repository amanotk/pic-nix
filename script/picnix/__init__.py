#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Python Data Analysis Tool for PIC-NIX
"""

DEFAULT_LOG_PREFIX = "log"
DEFAULT_LOAD_PREFIX = "load"
DEFAULT_FIELD_PREFIX = "field"
DEFAULT_PARTICLE_PREFIX = "particle"
DEFAULT_TRACER_PREFIX = "tracer"

from .utils import *
from .field import *
from .particle import *
from .run import *
