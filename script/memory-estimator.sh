#!/bin/sh

set -eu

exec "${PYTHON:-python3}" -m picnix.memory_estimator "$@"
