#!/usr/bin/env bash
set -euo pipefail

NPROC="${1:-16}"
TMAX="${2:-60}"
CONFIG="${3:-config-run.toml}"

echo "=== Running (np=${NPROC}, tmax=${TMAX}, config=${CONFIG}) ==="
rm -rf data plots
mpiexec -n "${NPROC}" ./main.out -c "${CONFIG}" -t "${TMAX}"

echo "=== Plotting ==="
python3 "$(dirname "$0")/quicklook.py" --basedir data --outdir plots

echo "=== Done: plots/ ==="
