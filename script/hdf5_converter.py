#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compatibility wrapper for the packaged PIC-NIX HDF5 converter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from picnix.hdf5_converter import main


if __name__ == "__main__":
    main()
