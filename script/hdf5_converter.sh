#!/bin/sh

set -eu

exec "${PYTHON:-python3}" -m picnix.hdf5_converter "$@"
