#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert PIC-NIX diagnostic output to per-step HDF5 plus VDS indexes.

The converter treats the input directory as read-only.  It writes one HDF5
data file per prefix per step by default and a lightweight VDS index per
prefix.  Serial HDF5 is used safely by assigning each step file to a single
MPI rank; each rank writes its assigned files in bounded first-dimension
slabs so a full step does not need to fit in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):  # pragma: no cover - depends on MPI runtime
    MPI = None


STEP_WIDTH = 8
NODE_RE = re.compile(r"node\d+$")
JSON_RE = re.compile(r"\d+\.json$")
SKIP_PREFIXES = {"hdf5"}
COMMANDS = {"convert", "verify", "remove-original"}
REMOVE_CONFIRMATION = "remove original diagnostics"
NODE_METADATA_FILES = ("history.txt",)


class SerialComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def barrier(self):
        return None

    def bcast(self, value, root=0):
        return value

    def allgather(self, value):
        return [value]


def get_comm():
    if MPI is None:
        return SerialComm()
    return MPI.COMM_WORLD


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    command = "convert"
    if argv and argv[0] in COMMANDS:
        command = argv[0]
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        prog="picnix-hdf5-convert",
        usage="picnix-hdf5-convert [convert|verify|remove-original] --input-dir INPUT_DIR [options]",
        description="Convert PIC-NIX posix/mpiio diagnostics to HDF5 + VDS",
        epilog=(
            "Commands: convert is the default and runs verification unless --no-verify is set; "
            "verify checks existing HDF5 output and stamps manifest.json; "
            "remove-original interactively deletes verified original .json/.data files."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <input-dir>/hdf5)",
    )
    parser.add_argument(
        "--prefix",
        nargs="+",
        help="diagnostic prefix(es) to convert (default: discover all prefixes)",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="selected step numbers; also accepts comma lists and start:stop[:stride]",
    )
    parser.add_argument(
        "--step-limit", type=int, help="limit number of discovered steps"
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=3,
        help="number of steps per prefix for sample value verification",
    )
    parser.add_argument(
        "--verify-level",
        choices=("fast", "full"),
        default="fast",
        help="verification level: fast checks all HDF5 and sampled originals; full scans all original metadata",
    )
    parser.add_argument("--compression", default="none")
    parser.add_argument("--compression-opts", type=int, default=1)
    parser.add_argument(
        "--field-dtype",
        choices=("source", "float32", "float64"),
        default="float32",
        help="converted dtype for field-like datasets (default: float32)",
    )
    parser.add_argument(
        "--particle-dtype",
        choices=("source", "float32", "float64"),
        default="float32",
        help="converted dtype for particle value columns (default: float32)",
    )
    parser.add_argument(
        "--particle-id-dtype",
        choices=("uint64", "int64"),
        default="uint64",
        help="converted dtype for particle IDs split from the last column",
    )
    parser.add_argument(
        "--target-chunk-mib",
        type=float,
        default=4.0,
        help="target uncompressed HDF5 chunk size in MiB",
    )
    parser.add_argument(
        "--max-buffer-mib",
        type=float,
        default=1024.0,
        help="approximate maximum data buffer per dataset batch in MiB",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing valid step files and regenerate the VDS index",
    )
    parser.add_argument("--no-vds", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="do not run verification after conversion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what remove-original would delete without deleting files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip interactive confirmation for remove-original",
    )
    args = parser.parse_args(argv)
    args.command = command
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.command != "convert" and (args.overwrite or args.resume or args.no_verify):
        parser.error(
            "--overwrite, --resume, and --no-verify are only valid for convert"
        )
    if args.command != "remove-original" and (args.dry_run or args.yes):
        parser.error("--dry-run and --yes are only valid for remove-original")
    if args.command == "convert" and args.no_vds and not args.no_verify:
        parser.error(
            "default verification requires VDS output; use --no-verify with --no-vds"
        )
    return args


def step_name(step):
    return f"{int(step):0{STEP_WIDTH}d}"


def parse_step_tokens(tokens):
    if not tokens:
        return None

    steps = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                values = part.split(":")
                if len(values) not in (2, 3):
                    raise ValueError(f"invalid step range: {part}")
                start = int(values[0])
                stop = int(values[1])
                stride = int(values[2]) if len(values) == 3 else 1
                if stride == 0:
                    raise ValueError(f"invalid zero stride in step range: {part}")
                end = stop + (1 if stride > 0 else -1)
                steps.extend(range(start, end, stride))
            else:
                steps.append(int(part))
    return [step_name(step) for step in steps]


def detect_iomode(input_dir, prefix):
    posix_dir = input_dir / "node000000" / prefix
    mpiio_dir = input_dir / prefix
    if posix_dir.is_dir():
        return "posix"
    if mpiio_dir.is_dir():
        return "mpiio"
    raise FileNotFoundError(
        f"could not detect iomode for prefix {prefix!r} under {input_dir}"
    )


def discover_nodes(input_dir):
    nodes = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and NODE_RE.match(path.name)
    )
    if not nodes:
        raise FileNotFoundError(f"no node directories found under {input_dir}")
    return nodes


def discover_steps(input_dir, prefix, iomode, nodes, selected, limit):
    if iomode == "posix":
        dirname = nodes[0] / prefix
    else:
        dirname = input_dir / prefix

    if not dirname.is_dir():
        raise FileNotFoundError(f"diagnostic directory not found: {dirname}")

    discovered = sorted(
        path.stem for path in dirname.iterdir() if JSON_RE.match(path.name)
    )
    if selected is not None:
        available = set(discovered)
        missing = [stem for stem in selected if stem not in available]
        if missing:
            raise FileNotFoundError(
                f"selected steps not found for prefix {prefix}: {', '.join(missing)}"
            )
        discovered = selected
    if limit is not None:
        discovered = discovered[:limit]
    if not discovered:
        raise FileNotFoundError(f"no steps found for prefix {prefix}")
    return discovered


def looks_like_diagnostic_dir(path):
    if not path.is_dir() or path.name in SKIP_PREFIXES:
        return False
    return any(JSON_RE.match(child.name) for child in path.iterdir() if child.is_file())


def discover_prefixes(input_dir):
    posix_probe = input_dir / "node000000"
    if posix_probe.is_dir():
        prefixes = sorted(
            path.name
            for path in posix_probe.iterdir()
            if looks_like_diagnostic_dir(path)
        )
        if prefixes:
            return prefixes

    prefixes = sorted(
        path.name for path in input_dir.iterdir() if looks_like_diagnostic_dir(path)
    )
    if prefixes:
        return prefixes
    raise FileNotFoundError(f"no diagnostic prefixes found under {input_dir}")


def load_json(path):
    with path.open("r") as fp:
        return json.load(fp)


def endian_prefix(endian):
    if endian == 1:
        return "<"
    if endian == 16777216:
        return ">"
    return ""


def normalized_shape(info, layout):
    shape = tuple(info["shape"])
    if layout == 0:
        return shape[::-1]
    return shape


def dtype_from_info(info, byteorder):
    return np.dtype(byteorder + info["datatype"])


def dtype_choice(name, source_dtype=None):
    if name == "source":
        if source_dtype is None:
            raise ValueError("source dtype requested but no source dtype was provided")
        return np.dtype(source_dtype)
    return np.dtype(name)


def compression_kwargs(args, shape, dtype, target_mib=None):
    compression = None if args.compression.lower() == "none" else args.compression
    kwargs = {
        "compression": compression,
        "shuffle": compression is not None,
    }
    if compression is not None and args.compression_opts is not None:
        kwargs["compression_opts"] = args.compression_opts

    if compression is not None:
        kwargs["chunks"] = compute_hdf5_chunks(
            shape, dtype, args.target_chunk_mib if target_mib is None else target_mib
        )
    return kwargs


def compute_hdf5_chunks(shape, dtype, target_mib):
    if not shape:
        return None
    if len(shape) == 1:
        row_bytes = np.dtype(dtype).itemsize
        trailing = ()
    else:
        row_bytes = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        trailing = tuple(shape[1:])
    target_bytes = max(1, int(target_mib * 1024 * 1024))
    chunk0 = max(1, min(int(shape[0]), max(1, target_bytes // max(1, row_bytes))))
    return (chunk0, *trailing)


def read_dataset_slab(data_path, info, byteorder, layout, start=0, count=None):
    dtype = dtype_from_info(info, byteorder)
    shape = normalized_shape(info, layout)
    if len(shape) == 0:
        total = 1
        slab_shape = shape
        file_offset = int(info["offset"])
    else:
        total = shape[0]
        if count is None:
            count = total - start
        row_count = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        slab_shape = (count, *shape[1:])
        file_offset = int(info["offset"]) + start * row_count * dtype.itemsize

    with data_path.open("rb") as fp:
        fp.seek(file_offset)
        payload = fp.read(int(np.prod(slab_shape)) * dtype.itemsize)
    array = np.frombuffer(payload, dtype=dtype).reshape(slab_shape).copy()
    if len(shape) == 1 and shape[0] == 1:
        return array[0]
    return array


def selected_dataset_names(dataset):
    return sorted(dataset)


def detect_kind(prefix, dataset_names):
    if "particle" in prefix or all(name.startswith("up") for name in dataset_names):
        return "particle"
    return "field"


def posix_json_path(node_dir, prefix, stem):
    return node_dir / prefix / f"{stem}.json"


def mpiio_json_path(input_dir, prefix, stem):
    return input_dir / prefix / f"{stem}.json"


def collect_posix_metadata(nodes, prefix, stem):
    entries = []
    dataset_names = None
    for node in nodes:
        json_path = posix_json_path(node, prefix, stem)
        root = load_json(json_path)
        meta = root["meta"]
        byteorder = endian_prefix(meta.get("endian"))
        layout = meta.get("layout", meta.get("order", 0))
        data_path = json_path.parent / meta["rawfile"]
        names = selected_dataset_names(root["dataset"])
        if dataset_names is None:
            dataset_names = names
        elif names != dataset_names:
            raise ValueError(f"dataset names differ in {json_path}")
        entries.append(
            {
                "node": node.name,
                "json_path": json_path,
                "data_path": data_path,
                "meta": meta,
                "dataset": root["dataset"],
                "byteorder": byteorder,
                "layout": layout,
            }
        )
    return entries, dataset_names or []


def collect_mpiio_metadata(input_dir, prefix, stem):
    json_path = mpiio_json_path(input_dir, prefix, stem)
    root = load_json(json_path)
    meta = root["meta"]
    names = selected_dataset_names(root["dataset"])
    return {
        "json_path": json_path,
        "data_path": json_path.parent / meta["rawfile"],
        "meta": meta,
        "dataset": root["dataset"],
        "byteorder": endian_prefix(meta.get("endian")),
        "layout": meta.get("layout", meta.get("order", 0)),
    }, names


def dataset_shape_dtype_from_posix(entries, name):
    dtype = None
    trailing = None
    rows = 0
    for entry in entries:
        info = entry["dataset"][name]
        current_dtype = dtype_from_info(info, entry["byteorder"])
        shape = normalized_shape(info, entry["layout"])
        if dtype is None:
            dtype = current_dtype
            trailing = shape[1:]
        elif current_dtype != dtype or shape[1:] != trailing:
            raise ValueError(f"inconsistent shape or dtype for dataset {name}")
        rows += shape[0]
    return (rows, *trailing), dtype


def dataset_shape_dtype_from_mpiio(entry, name):
    info = entry["dataset"][name]
    return normalized_shape(info, entry["layout"]), dtype_from_info(
        info, entry["byteorder"]
    )


def particle_value_shape(raw_shape):
    if len(raw_shape) != 2 or raw_shape[1] < 2:
        raise ValueError(
            f"particle datasets must be shaped like (N, columns), got {raw_shape}"
        )
    return (raw_shape[0], raw_shape[1] - 1)


def convert_particle_id(raw_id, dtype):
    # The raw JSON labels particle records as f8, but the last 8-byte slot is a
    # particle ID bit pattern.  Preserve it by reinterpreting bytes, not by
    # numerically converting the nonsensical float64 value.
    return np.ascontiguousarray(raw_id).view(np.uint64).astype(dtype, copy=False)


def add_timing(timing, key, elapsed):
    timing[key] = timing.get(key, 0.0) + elapsed


def iter_posix_batches(entries, name, max_buffer_bytes, timing):
    arrays = []
    buffered = 0
    for entry in entries:
        t0 = time.perf_counter()
        array = read_dataset_slab(
            entry["data_path"],
            entry["dataset"][name],
            entry["byteorder"],
            entry["layout"],
        )
        add_timing(timing, "read", time.perf_counter() - t0)
        if arrays and buffered + array.nbytes > max_buffer_bytes:
            t0 = time.perf_counter()
            batch = np.concatenate(arrays, axis=0)
            add_timing(timing, "assemble", time.perf_counter() - t0)
            yield batch
            arrays = []
            buffered = 0
        arrays.append(array)
        buffered += array.nbytes
    if arrays:
        t0 = time.perf_counter()
        batch = np.concatenate(arrays, axis=0)
        add_timing(timing, "assemble", time.perf_counter() - t0)
        yield batch


def iter_mpiio_slabs(entry, name, max_buffer_bytes, timing):
    info = entry["dataset"][name]
    shape = normalized_shape(info, entry["layout"])
    dtype = dtype_from_info(info, entry["byteorder"])
    row_bytes = (
        int(np.prod(shape[1:])) * dtype.itemsize if len(shape) > 1 else dtype.itemsize
    )
    rows_per_slab = max(1, max_buffer_bytes // max(1, row_bytes))
    for start in range(0, shape[0], rows_per_slab):
        count = min(shape[0] - start, rows_per_slab)
        t0 = time.perf_counter()
        array = read_dataset_slab(
            entry["data_path"], info, entry["byteorder"], entry["layout"], start, count
        )
        add_timing(timing, "read", time.perf_counter() - t0)
        yield array


def create_direct_dataset(group, name, shape, dtype, args):
    return group.create_dataset(
        name, shape=shape, dtype=dtype, **compression_kwargs(args, shape, dtype)
    )


def write_direct_from_batches(group, name, shape, dtype, args, batches, timing):
    t0 = time.perf_counter()
    dataset = create_direct_dataset(group, name, shape, dtype, args)
    add_timing(timing, "create_dataset", time.perf_counter() - t0)
    offset = 0
    for batch in batches:
        count = batch.shape[0]
        t0 = time.perf_counter()
        dataset[offset : offset + count, ...] = batch
        add_timing(timing, "write", time.perf_counter() - t0)
        offset += count
    if offset != shape[0]:
        raise ValueError(f"wrote {offset} rows for {name}, expected {shape[0]}")


def write_particle_blocks(
    group, name, value_shape, value_dtype, id_dtype, args, batches, timing
):
    blocks_group = group.require_group("particles").require_group(name)
    value_group = blocks_group.require_group("value")
    id_group = blocks_group.require_group("id")
    value_paths = []
    id_paths = []
    offset = 0
    for block_index, raw_batch in enumerate(batches):
        block_name = f"block{block_index:06d}"
        t0 = time.perf_counter()
        value_batch = raw_batch[:, :-1].astype(value_dtype, copy=False)
        id_batch = convert_particle_id(raw_batch[:, -1], id_dtype)
        add_timing(timing, "assemble", time.perf_counter() - t0)

        t0 = time.perf_counter()
        value_block = value_group.create_dataset(
            block_name,
            data=value_batch,
            **compression_kwargs(
                args, value_batch.shape, value_dtype, args.target_chunk_mib
            ),
        )
        id_block = id_group.create_dataset(
            block_name,
            data=id_batch,
            **compression_kwargs(args, id_batch.shape, id_dtype, args.target_chunk_mib),
        )
        add_timing(timing, "write", time.perf_counter() - t0)
        value_block.attrs["offset0"] = offset
        id_block.attrs["offset0"] = offset
        value_paths.append(value_block.name)
        id_paths.append(id_block.name)
        offset += value_batch.shape[0]
    if offset != value_shape[0]:
        raise ValueError(f"wrote {offset} rows for {name}, expected {value_shape[0]}")

    t0 = time.perf_counter()
    value_layout = h5py.VirtualLayout(shape=value_shape, dtype=value_dtype)
    id_layout = h5py.VirtualLayout(shape=(value_shape[0],), dtype=id_dtype)
    cursor = 0
    for block_path in value_paths:
        block = group.file[block_path]
        count = block.shape[0]
        source = h5py.VirtualSource(block)
        value_layout[cursor : cursor + count, ...] = source
        cursor += count
    cursor = 0
    for block_path in id_paths:
        block = group.file[block_path]
        count = block.shape[0]
        source = h5py.VirtualSource(block)
        id_layout[cursor : cursor + count] = source
        cursor += count
    group.create_virtual_dataset(name, value_layout)
    group.create_virtual_dataset(f"{name}_id", id_layout)
    add_timing(timing, "internal_vds", time.perf_counter() - t0)


def copy_step_attrs(group, stem, prefix, iomode, kind, meta, source_count):
    group.attrs["step"] = int(meta.get("step", int(stem)))
    group.attrs["time"] = float(meta.get("time", int(stem)))
    group.attrs["layout"] = int(meta.get("layout", meta.get("order", 0)))
    group.attrs["prefix"] = prefix
    group.attrs["iomode"] = iomode
    group.attrs["kind"] = kind
    group.attrs["source_count"] = int(source_count)
    if "endian" in meta:
        group.attrs["endian"] = int(meta["endian"])
    if "chunk_id_range" in meta and meta["chunk_id_range"] is not None:
        group.attrs["first_chunk_id_range"] = np.array(
            meta["chunk_id_range"], dtype=np.int64
        )


def read_existing_step_info(path, stem):
    with h5py.File(path, "r") as h5fp:
        if h5fp.attrs.get("picnix_hdf5_layout") != "step-v1":
            raise OSError(f"not a converter step file: {path}")
        group = h5fp[stem]
        datasets = {}
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"shape": tuple(item.shape), "dtype": str(item.dtype)}
        return {
            "stem": stem,
            "step": int(group.attrs["step"]),
            "time": float(group.attrs["time"]),
            "path": str(path),
            "size": path.stat().st_size,
            "elapsed": 0.0,
            "timing": {
                "metadata": 0.0,
                "hdf5_open": 0.0,
                "create_dataset": 0.0,
                "read": 0.0,
                "assemble": 0.0,
                "write": 0.0,
                "internal_vds": 0.0,
            },
            "kind": group.attrs.get("kind", "unknown"),
            "datasets": datasets,
            "resumed": True,
        }


def write_step_file(args, stem, iomode, nodes):
    max_buffer_bytes = max(1, int(args.max_buffer_mib * 1024 * 1024))
    output_path = args.output_dir / args.prefix / f"{stem}.h5"
    tmp_path = args.output_dir / args.prefix / f"{stem}.h5.tmp"
    if args.resume and output_path.exists():
        try:
            return read_existing_step_info(output_path, stem)
        except OSError:
            output_path.unlink()
    if output_path.exists():
        output_path.unlink()
    if tmp_path.exists():
        tmp_path.unlink()

    timing = {
        "metadata": 0.0,
        "hdf5_open": 0.0,
        "create_dataset": 0.0,
        "read": 0.0,
        "assemble": 0.0,
        "write": 0.0,
        "internal_vds": 0.0,
    }
    t0 = time.perf_counter()
    if iomode == "posix":
        entries, names = collect_posix_metadata(nodes, args.prefix, stem)
        first_meta = entries[0]["meta"]
        source_count = len(entries)
        shape_dtype = {
            name: dataset_shape_dtype_from_posix(entries, name) for name in names
        }
    else:
        entry, names = collect_mpiio_metadata(args.input_dir, args.prefix, stem)
        first_meta = entry["meta"]
        source_count = 1
        shape_dtype = {
            name: dataset_shape_dtype_from_mpiio(entry, name) for name in names
        }
    add_timing(timing, "metadata", time.perf_counter() - t0)

    kind = detect_kind(args.prefix, names)
    if kind == "particle":
        id_dtype = dtype_choice(args.particle_id_dtype)
        output_datasets = {}
        for name, (raw_shape, _) in shape_dtype.items():
            value_shape = particle_value_shape(raw_shape)
            value_dtype = dtype_choice(args.particle_dtype, shape_dtype[name][1])
            output_datasets[name] = {
                "shape": tuple(value_shape),
                "dtype": str(value_dtype),
            }
            output_datasets[f"{name}_id"] = {
                "shape": (value_shape[0],),
                "dtype": str(id_dtype),
            }
    else:
        id_dtype = None
        output_datasets = {
            name: {
                "shape": tuple(shape),
                "dtype": str(dtype_choice(args.field_dtype, source_dtype)),
            }
            for name, (shape, source_dtype) in shape_dtype.items()
        }

    t0 = time.perf_counter()
    h5_open_t0 = time.perf_counter()
    with h5py.File(tmp_path, "w", libver="latest") as h5fp:
        h5fp.attrs["picnix_hdf5_layout"] = "step-v1"
        h5fp.attrs["converter"] = "picnix.hdf5_converter"
        h5fp.attrs["field_dtype"] = args.field_dtype
        h5fp.attrs["particle_dtype"] = args.particle_dtype
        h5fp.attrs["particle_id_dtype"] = args.particle_id_dtype
        group = h5fp.create_group(stem)
        copy_step_attrs(
            group, stem, args.prefix, iomode, kind, first_meta, source_count
        )
        add_timing(timing, "hdf5_open", time.perf_counter() - h5_open_t0)
        for name in names:
            shape, raw_dtype = shape_dtype[name]
            if iomode == "posix":
                batches = iter_posix_batches(entries, name, max_buffer_bytes, timing)
            else:
                batches = iter_mpiio_slabs(entry, name, max_buffer_bytes, timing)

            if kind == "particle":
                value_dtype = dtype_choice(args.particle_dtype, raw_dtype)
                write_particle_blocks(
                    group,
                    name,
                    particle_value_shape(shape),
                    value_dtype,
                    id_dtype,
                    args,
                    batches,
                    timing,
                )
            else:
                value_dtype = dtype_choice(args.field_dtype, raw_dtype)
                write_direct_from_batches(
                    group, name, shape, value_dtype, args, batches, timing
                )

    elapsed = time.perf_counter() - t0
    tmp_path.replace(output_path)
    return {
        "stem": stem,
        "step": int(first_meta.get("step", int(stem))),
        "time": float(first_meta.get("time", int(stem))),
        "path": str(output_path),
        "size": output_path.stat().st_size,
        "elapsed": elapsed,
        "timing": timing,
        "kind": kind,
        "datasets": output_datasets,
    }


def create_prefix_vds(args, step_infos):
    vds_path = args.output_dir / f"{args.prefix}.vds.h5"
    if vds_path.exists():
        vds_path.unlink()

    step_infos = sorted(step_infos, key=lambda item: item["step"])
    with h5py.File(vds_path, "w", libver="latest") as h5fp:
        h5fp.attrs["picnix_hdf5_layout"] = "prefix-vds-v1"
        h5fp.attrs["prefix"] = args.prefix
        root = h5fp.create_group(args.prefix)
        root.create_dataset(
            "step", data=np.array([item["step"] for item in step_infos], dtype=np.int64)
        )
        root.create_dataset(
            "time",
            data=np.array([item["time"] for item in step_infos], dtype=np.float64),
        )
        for info in step_infos:
            step_group = root.create_group(info["stem"])
            source_file = f"{args.prefix}/{info['stem']}.h5"
            for name, dsinfo in sorted(info["datasets"].items()):
                shape = tuple(dsinfo["shape"])
                dtype = np.dtype(dsinfo["dtype"])
                layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
                source = h5py.VirtualSource(
                    source_file, f"/{info['stem']}/{name}", shape=shape
                )
                layout[...] = source
                step_group.create_virtual_dataset(name, layout)
    return vds_path


def write_manifest(args, prefix_results):
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "layout": "picnix-hdf5-v1",
        "source": os_path_rel(args.input_dir, args.output_dir),
        "defaults": {
            "field_dtype": args.field_dtype,
            "particle_dtype": args.particle_dtype,
            "particle_id_dtype": args.particle_id_dtype,
            "compression": args.compression,
            "compression_opts": args.compression_opts,
        },
        "prefixes": prefix_results,
    }
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as fp:
        json.dump(manifest, fp, indent=2)
    tmp_path.replace(manifest_path)
    return manifest_path


def load_manifest(output_dir):
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r") as fp:
        return json.load(fp)


def update_manifest(output_dir, manifest):
    manifest_path = output_dir / "manifest.json"
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as fp:
        json.dump(manifest, fp, indent=2)
    tmp_path.replace(manifest_path)
    return manifest_path


def apply_manifest_defaults(args, manifest):
    defaults = manifest.get("defaults", {})
    args.field_dtype = defaults.get("field_dtype", args.field_dtype)
    args.particle_dtype = defaults.get("particle_dtype", args.particle_dtype)
    args.particle_id_dtype = defaults.get("particle_id_dtype", args.particle_id_dtype)
    args.compression = defaults.get("compression", args.compression)
    args.compression_opts = defaults.get("compression_opts", args.compression_opts)


def select_prefixes(args, manifest=None):
    if args.prefix is not None:
        return args.prefix
    if manifest is not None and manifest.get("prefixes"):
        return [item["prefix"] for item in manifest["prefixes"]]
    return discover_prefixes(args.input_dir)


def raw_files_for_prefix(input_dir, prefix, iomode, nodes, steps):
    files = []
    for stem in steps:
        json_paths = (
            [node / prefix / f"{stem}.json" for node in nodes]
            if iomode == "posix"
            else [input_dir / prefix / f"{stem}.json"]
        )
        for json_path in json_paths:
            root = load_json(json_path)
            files.append(json_path)
            files.append(json_path.parent / root["meta"]["rawfile"])
    return files


def fingerprint_originals(args, prefixes):
    selected_steps = parse_step_tokens(args.steps)
    prefix_info = []
    total_files = 0
    total_bytes = 0
    for prefix in prefixes:
        iomode = detect_iomode(args.input_dir, prefix)
        nodes = discover_nodes(args.input_dir) if iomode == "posix" else []
        steps = discover_steps(
            args.input_dir, prefix, iomode, nodes, selected_steps, args.step_limit
        )
        files = raw_files_for_prefix(args.input_dir, prefix, iomode, nodes, steps)
        byte_count = sum(path.stat().st_size for path in files)
        total_files += len(files)
        total_bytes += byte_count
        prefix_info.append(
            {
                "prefix": prefix,
                "iomode": iomode,
                "steps": len(steps),
                "first_step": steps[0],
                "last_step": steps[-1],
                "nodes": len(nodes),
                "files": len(files),
                "bytes": byte_count,
            }
        )
    return {"prefixes": prefix_info, "files": total_files, "bytes": total_bytes}


def fingerprint_fast(args, prefixes, manifest=None):
    selected_steps = parse_step_tokens(args.steps)
    manifest_prefixes = {
        item["prefix"]: item for item in (manifest or {}).get("prefixes", [])
    }
    prefix_info = []
    total_hdf5_files = 0
    total_hdf5_bytes = 0
    for prefix in prefixes:
        iomode = detect_iomode(args.input_dir, prefix)
        nodes = discover_nodes(args.input_dir) if iomode == "posix" else []
        steps = discover_steps(
            args.input_dir, prefix, iomode, nodes, selected_steps, args.step_limit
        )
        manifest_item = manifest_prefixes.get(prefix, {})
        total_hdf5_files += int(manifest_item.get("files", 0))
        total_hdf5_bytes += int(manifest_item.get("size", 0))
        prefix_info.append(
            {
                "prefix": prefix,
                "iomode": iomode,
                "steps": len(steps),
                "first_step": steps[0],
                "last_step": steps[-1],
                "nodes": len(nodes),
                "hdf5_files": int(manifest_item.get("files", 0)),
                "hdf5_bytes": int(manifest_item.get("size", 0)),
            }
        )
    return {
        "level": "fast",
        "prefixes": prefix_info,
        "hdf5_files": total_hdf5_files,
        "hdf5_bytes": total_hdf5_bytes,
    }


def expected_output_datasets(args, prefix, stem, iomode, nodes):
    if iomode == "posix":
        entries, names = collect_posix_metadata(nodes, prefix, stem)
        shape_dtype = {
            name: dataset_shape_dtype_from_posix(entries, name) for name in names
        }
    else:
        entry, names = collect_mpiio_metadata(args.input_dir, prefix, stem)
        shape_dtype = {
            name: dataset_shape_dtype_from_mpiio(entry, name) for name in names
        }

    kind = detect_kind(prefix, names)
    if kind == "particle":
        id_dtype = dtype_choice(args.particle_id_dtype)
        datasets = {}
        for name, (raw_shape, source_dtype) in shape_dtype.items():
            value_shape = particle_value_shape(raw_shape)
            value_dtype = dtype_choice(args.particle_dtype, source_dtype)
            datasets[name] = {"shape": tuple(value_shape), "dtype": str(value_dtype)}
            datasets[f"{name}_id"] = {
                "shape": (value_shape[0],),
                "dtype": str(id_dtype),
            }
        return kind, datasets, shape_dtype

    datasets = {
        name: {
            "shape": tuple(shape),
            "dtype": str(dtype_choice(args.field_dtype, source_dtype)),
        }
        for name, (shape, source_dtype) in shape_dtype.items()
    }
    return kind, datasets, shape_dtype


def sample_indices(size, count):
    if size <= 0 or count <= 0:
        return []
    candidates = [0, size // 2, size - 1]
    if count > 3:
        candidates.extend(np.linspace(0, size - 1, count, dtype=np.int64).tolist())
    result = []
    for index in candidates:
        index = int(index)
        if index not in result:
            result.append(index)
        if len(result) >= count:
            break
    return result


def sample_stems(steps, count):
    if count <= 0:
        return []
    indices = sample_indices(len(steps), count)
    return [steps[index] for index in indices]


def step_file_datasets(step_path, stem):
    with h5py.File(step_path, "r") as h5fp:
        if h5fp.attrs.get("picnix_hdf5_layout") != "step-v1":
            raise ValueError(f"invalid step layout: {step_path}")
        group = h5fp[stem]
        return {
            name: {"shape": tuple(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in group.items()
            if isinstance(dataset, h5py.Dataset)
        }


def read_posix_row(entries, name, row):
    cursor = 0
    for entry in entries:
        info = entry["dataset"][name]
        shape = normalized_shape(info, entry["layout"])
        next_cursor = cursor + shape[0]
        if cursor <= row < next_cursor:
            return read_dataset_slab(
                entry["data_path"],
                info,
                entry["byteorder"],
                entry["layout"],
                row - cursor,
                1,
            )
        cursor = next_cursor
    raise IndexError(f"row {row} out of range for {name}")


def read_source_row(args, prefix, stem, iomode, nodes, name, row):
    if iomode == "posix":
        entries, _ = collect_posix_metadata(nodes, prefix, stem)
        return read_posix_row(entries, name, row)
    entry, _ = collect_mpiio_metadata(args.input_dir, prefix, stem)
    info = entry["dataset"][name]
    return read_dataset_slab(
        entry["data_path"], info, entry["byteorder"], entry["layout"], row, 1
    )


def verify_sample_values(args, prefix, stem, iomode, nodes, kind, shape_dtype):
    step_path = args.output_dir / prefix / f"{stem}.h5"
    with h5py.File(step_path, "r") as h5fp:
        group = h5fp[stem]
        for name, (raw_shape, source_dtype) in shape_dtype.items():
            for row in sample_indices(raw_shape[0], 3):
                source = read_source_row(args, prefix, stem, iomode, nodes, name, row)
                if kind == "particle":
                    value_dtype = dtype_choice(args.particle_dtype, source_dtype)
                    expected_value = source[:, :-1].astype(value_dtype, copy=False)
                    actual_value = group[name][row : row + 1, ...]
                    np.testing.assert_allclose(actual_value, expected_value)
                    expected_id = convert_particle_id(
                        source[:, -1], dtype_choice(args.particle_id_dtype)
                    )
                    actual_id = group[f"{name}_id"][row : row + 1]
                    np.testing.assert_array_equal(actual_id, expected_id)
                else:
                    value_dtype = dtype_choice(args.field_dtype, source_dtype)
                    expected = source.astype(value_dtype, copy=False)
                    actual = group[name][row : row + 1, ...]
                    np.testing.assert_allclose(actual, expected)


def verify_prefix(args, prefix):
    selected_steps = parse_step_tokens(args.steps)
    iomode = detect_iomode(args.input_dir, prefix)
    nodes = discover_nodes(args.input_dir) if iomode == "posix" else []
    steps = discover_steps(
        args.input_dir, prefix, iomode, nodes, selected_steps, args.step_limit
    )
    vds_path = args.output_dir / f"{prefix}.vds.h5"
    if not vds_path.exists():
        raise FileNotFoundError(f"VDS file not found: {vds_path}")

    print(f"verifying prefix {prefix}: {len(steps)} steps", flush=True)
    sample_set = set(sample_stems(steps, args.verify_samples))
    with h5py.File(vds_path, "r") as vds:
        if vds.attrs.get("picnix_hdf5_layout") != "prefix-vds-v1":
            raise ValueError(f"invalid VDS layout: {vds_path}")
        if vds.attrs.get("prefix") != prefix:
            raise ValueError(f"VDS prefix mismatch: {vds_path}")
        root = vds[prefix]
        if root["step"][...].tolist() != sorted(int(stem) for stem in steps):
            raise ValueError(f"step list mismatch for {prefix}")

        for stem in steps:
            step_path = args.output_dir / prefix / f"{stem}.h5"
            if not step_path.exists():
                raise FileNotFoundError(f"step file not found: {step_path}")
            if args.verify_level == "full" or stem in sample_set:
                kind, expected, shape_dtype = expected_output_datasets(
                    args, prefix, stem, iomode, nodes
                )
            else:
                kind = None
                shape_dtype = None
                expected = step_file_datasets(step_path, stem)
            with h5py.File(step_path, "r") as h5fp:
                if h5fp.attrs.get("picnix_hdf5_layout") != "step-v1":
                    raise ValueError(f"invalid step layout: {step_path}")
                group = h5fp[stem]
                if group.attrs.get("prefix") != prefix:
                    raise ValueError(f"step prefix mismatch: {step_path}")
                for name, info in expected.items():
                    dataset = group[name]
                    if tuple(dataset.shape) != tuple(info["shape"]):
                        raise ValueError(f"shape mismatch for {step_path}:{name}")
                    if str(dataset.dtype) != info["dtype"]:
                        raise ValueError(f"dtype mismatch for {step_path}:{name}")
                    if dataset.shape and dataset.shape[0] > 0:
                        dataset[0:1, ...]

                    vds_dataset = root[stem][name]
                    if tuple(vds_dataset.shape) != tuple(info["shape"]):
                        raise ValueError(
                            f"VDS shape mismatch for {prefix}/{stem}/{name}"
                        )
                    if vds_dataset.shape and vds_dataset.shape[0] > 0:
                        vds_dataset[0:1, ...]

            if stem in sample_set:
                verify_sample_values(
                    args, prefix, stem, iomode, nodes, kind, shape_dtype
                )

    if args.verify_level == "full":
        files = raw_files_for_prefix(args.input_dir, prefix, iomode, nodes, steps)
        file_count = len(files)
        byte_count = sum(path.stat().st_size for path in files)
    else:
        file_count = None
        byte_count = None
    return {
        "prefix": prefix,
        "iomode": iomode,
        "steps": len(steps),
        "nodes": len(nodes),
        "files": file_count,
        "bytes": byte_count,
        "sampled_steps": sorted(sample_set),
    }


def verify_output(args):
    manifest = load_manifest(args.output_dir)
    apply_manifest_defaults(args, manifest)
    prefixes = select_prefixes(args, manifest)
    print("\nPIC-NIX HDF5 verification")
    print("=========================")
    print(f"input-dir:        {args.input_dir}")
    print(f"output-dir:       {args.output_dir}")
    print(f"prefixes:         {', '.join(prefixes)}")
    prefix_results = [verify_prefix(args, prefix) for prefix in prefixes]
    if args.verify_level == "full":
        fingerprint = fingerprint_originals(args, prefixes)
    else:
        fingerprint = fingerprint_fast(args, prefixes, manifest)
    manifest["verification"] = {
        "status": "passed",
        "level": args.verify_level,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verify_samples": args.verify_samples,
        "raw_fingerprint": fingerprint,
        "prefixes": prefix_results,
    }
    update_manifest(args.output_dir, manifest)
    print("verification:     passed")
    print(f"level:            {args.verify_level}")
    if args.verify_level == "full":
        print(f"original files:   {fingerprint['files']}")
        print(f"original size:    {fingerprint['bytes'] / 1024.0 / 1024.0:.3f} MiB")
    print("ready:            original diagnostics can be removed with remove-original")
    return manifest["verification"]


def os_path_rel(path, start):
    return os.path.relpath(path, start=start)


def prepare_output(args, comm):
    rank = comm.Get_rank()
    if rank == 0:
        prefix_dir = args.output_dir / args.prefix
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if prefix_dir.exists():
            if args.overwrite:
                shutil.rmtree(prefix_dir)
                if args.no_vds:
                    vds_path = args.output_dir / f"{args.prefix}.vds.h5"
                    if vds_path.exists():
                        vds_path.unlink()
            elif not args.resume:
                raise FileExistsError(
                    f"{prefix_dir} exists; pass --overwrite, --resume, or choose another output"
                )
        prefix_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()


def print_summary(args, prefix, iomode, nodes, steps, step_infos, vds_path, elapsed):
    total_size = sum(item["size"] for item in step_infos)
    timing_keys = (
        "metadata",
        "hdf5_open",
        "create_dataset",
        "read",
        "assemble",
        "write",
        "internal_vds",
    )
    total_timing = {
        key: sum(item.get("timing", {}).get(key, 0.0) for item in step_infos)
        for key in timing_keys
    }
    print("\nPIC-NIX HDF5 VDS conversion")
    print("===========================")
    print(f"input-dir:        {args.input_dir}")
    print(f"output-dir:       {args.output_dir}")
    print(f"prefix:           {prefix}")
    print(f"iomode:           {iomode}")
    print(f"steps:            {len(steps)}")
    if iomode == "posix":
        print(f"nodes:            {len(nodes)}")
    print(f"data h5 files:    {len(step_infos)}")
    if any(item.get("resumed", False) for item in step_infos):
        print(
            f"resumed files:     {sum(1 for item in step_infos if item.get('resumed', False))}"
        )
    print(f"data h5 size:     {total_size / 1024.0 / 1024.0:.3f} MiB")
    if vds_path is not None:
        print(f"vds file:         {vds_path}")
        print(f"vds size:         {vds_path.stat().st_size} bytes")
    print(f"elapsed:          {elapsed:.3f} s")
    print("")
    print("aggregate step timing")
    for key in timing_keys:
        print(f"  {key:14s}{total_timing[key]:9.3f} s")
    print("")
    print("per-step timing")
    for item in sorted(step_infos, key=lambda x: x["step"]):
        timing = item.get("timing", {})
        print(
            f"  step {item['stem']}: {item['size'] / 1024.0 / 1024.0:.3f} MiB, "
            f"{item['elapsed']:.3f} s, kind={item['kind']}"
            f"{', resumed' if item.get('resumed', False) else ''}"
        )
        print(
            "    "
            f"metadata={timing.get('metadata', 0.0):.3f}s "
            f"read={timing.get('read', 0.0):.3f}s "
            f"assemble={timing.get('assemble', 0.0):.3f}s "
            f"write={timing.get('write', 0.0):.3f}s"
        )


def convert_prefix(args, prefix, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = time.perf_counter()

    args.prefix = prefix
    selected_steps = parse_step_tokens(args.steps)
    iomode = detect_iomode(args.input_dir, prefix)
    nodes = discover_nodes(args.input_dir) if iomode == "posix" else []
    steps = discover_steps(
        args.input_dir, prefix, iomode, nodes, selected_steps, args.step_limit
    )

    prepare_output(args, comm)
    local_steps = steps[rank::size]
    local_infos = [write_step_file(args, stem, iomode, nodes) for stem in local_steps]
    gathered = comm.allgather(local_infos)
    step_infos = [item for sublist in gathered for item in sublist]
    comm.barrier()

    vds_path = None
    if rank == 0 and not args.no_vds:
        vds_path = create_prefix_vds(args, step_infos)
    comm.barrier()

    elapsed = time.perf_counter() - t0
    if rank == 0:
        print_summary(args, prefix, iomode, nodes, steps, step_infos, vds_path, elapsed)

    return {
        "prefix": prefix,
        "iomode": iomode,
        "steps": len(steps),
        "files": len(step_infos),
        "size": sum(item["size"] for item in step_infos),
        "vds": None if vds_path is None else str(vds_path.relative_to(args.output_dir)),
    }


def matching_verification(args, manifest, prefixes):
    verification = manifest.get("verification")
    if not verification or verification.get("status") != "passed":
        raise ValueError("verified HDF5 output is required before remove-original")
    level = verification.get("level", "full")
    current = (
        fingerprint_originals(args, prefixes)
        if level == "full"
        else fingerprint_fast(args, prefixes, manifest)
    )
    expected = verification.get("raw_fingerprint")
    if current != expected:
        raise ValueError("current original files do not match the verified fingerprint")
    return verification, current


def remove_empty_dirs(paths):
    removed = []
    kept = []
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        try:
            path.rmdir()
            removed.append(path)
        except OSError:
            kept.append(path)
    return removed, kept


def relocate_node_metadata(input_dir, node_dirs):
    relocated = []
    kept = []
    if input_dir / "node000000" not in node_dirs:
        return relocated, kept

    for filename in NODE_METADATA_FILES:
        source = input_dir / "node000000" / filename
        destination = input_dir / filename
        if not source.is_file():
            continue

        if destination.exists():
            if (
                destination.is_file()
                and destination.read_bytes() == source.read_bytes()
            ):
                source.unlink()
                relocated.append((source, destination))
            else:
                kept.append((source, destination))
            continue

        shutil.copy2(source, destination)
        source.unlink()
        relocated.append((source, destination))

    return relocated, kept


def estimate_removal_from_verification(verification):
    prefix_items = verification.get("raw_fingerprint", {}).get("prefixes", [])
    file_count = 0
    node_count = 0
    for item in prefix_items:
        sources = item.get("nodes", 0) if item.get("iomode") == "posix" else 1
        file_count += int(item.get("steps", 0)) * int(sources) * 2
        node_count = max(node_count, int(item.get("nodes", 0)))
    return file_count, node_count


def delete_file(path):
    if not path.exists():
        return 0, 0
    size = path.stat().st_size
    path.unlink()
    return 1, size


def delete_step_files(json_path):
    removed = 0
    bytes_freed = 0
    if not json_path.exists():
        return removed, bytes_freed
    root = load_json(json_path)
    raw_path = json_path.parent / root["meta"]["rawfile"]
    count, size = delete_file(raw_path)
    removed += count
    bytes_freed += size
    count, size = delete_file(json_path)
    removed += count
    bytes_freed += size
    return removed, bytes_freed


def delete_verified_originals(args, prefixes):
    selected_steps = parse_step_tokens(args.steps)
    total_removed = 0
    total_bytes = 0
    prefix_dirs = set()
    node_dirs = set()

    for prefix in prefixes:
        prefix_removed = 0
        iomode = detect_iomode(args.input_dir, prefix)
        nodes = discover_nodes(args.input_dir) if iomode == "posix" else []
        steps = discover_steps(
            args.input_dir, prefix, iomode, nodes, selected_steps, args.step_limit
        )
        for stem in steps:
            if iomode == "posix":
                json_paths = [node / prefix / f"{stem}.json" for node in nodes]
            else:
                json_paths = [args.input_dir / prefix / f"{stem}.json"]
            for json_path in json_paths:
                removed, bytes_freed = delete_step_files(json_path)
                total_removed += removed
                prefix_removed += removed
                total_bytes += bytes_freed

        if iomode == "posix":
            for node in nodes:
                prefix_dirs.add(node / prefix)
                node_dirs.add(node)
        else:
            prefix_dirs.add(args.input_dir / prefix)
        print(f"  {prefix}: removed {prefix_removed} files", flush=True)

    return total_removed, total_bytes, prefix_dirs, node_dirs


def remove_original(args):
    manifest = load_manifest(args.output_dir)
    apply_manifest_defaults(args, manifest)
    prefixes = select_prefixes(args, manifest)
    verification, _ = matching_verification(args, manifest, prefixes)
    estimated_files, estimated_nodes = estimate_removal_from_verification(verification)

    print("\nPIC-NIX remove-original")
    print("=======================")
    print(f"input-dir:        {args.input_dir}")
    print(f"output-dir:       {args.output_dir}")
    print(f"prefixes:         {', '.join(prefixes)}")
    print(f"verified level:   {verification.get('level', 'unknown')}")
    print(f"files to remove:  {estimated_files} estimated")
    print(f"node dirs:        {estimated_nodes} candidates")
    print("kept:             hdf5/, profile/log/config/unmatched files")

    if args.dry_run:
        print("dry-run:          no files removed")
        return

    if not args.yes:
        print("")
        print("This will remove original PIC-NIX diagnostic .json/.data files")
        print("covered by the verified HDF5 output. This cannot be undone.")
        response = input(f'Type "{REMOVE_CONFIRMATION}" to continue: ')
        if response != REMOVE_CONFIRMATION:
            raise SystemExit("remove-original cancelled")

    total_removed, total_bytes, prefix_dirs, node_dirs = delete_verified_originals(
        args, prefixes
    )

    relocated_metadata, kept_metadata = relocate_node_metadata(
        args.input_dir, node_dirs
    )
    removed_prefix_dirs, kept_prefix_dirs = remove_empty_dirs(prefix_dirs)
    removed_node_dirs, kept_node_dirs = remove_empty_dirs(node_dirs)

    print("")
    print(f"files removed:    {total_removed}")
    print(f"bytes freed:      {total_bytes / 1024.0 / 1024.0:.3f} MiB")
    if relocated_metadata:
        print(f"metadata moved:   {len(relocated_metadata)} files")
    print(f"dirs removed:     {len(removed_prefix_dirs) + len(removed_node_dirs)}")
    if kept_metadata:
        print(f"metadata kept:    {len(kept_metadata)} files with existing destination")
    if kept_prefix_dirs or kept_node_dirs:
        print(
            "kept non-empty:   "
            f"{len(kept_prefix_dirs)} prefix dirs, {len(kept_node_dirs)} node dirs"
        )


def main():
    args = parse_args()
    comm = get_comm()
    rank = comm.Get_rank()

    args.input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "hdf5"
    args.output_dir = args.output_dir.resolve()

    if args.command == "verify":
        if rank == 0:
            verify_output(args)
        return

    if args.command == "remove-original":
        if comm.Get_size() != 1:
            raise RuntimeError(
                "remove-original must be run in a normal single-process shell"
            )
        remove_original(args)
        return

    requested_prefixes = args.prefix
    prefixes = (
        requested_prefixes
        if requested_prefixes is not None
        else discover_prefixes(args.input_dir)
    )
    prefix_results = []
    for prefix in prefixes:
        result = convert_prefix(args, prefix, comm)
        if rank == 0:
            prefix_results.append(result)
    args.prefix = requested_prefixes

    if rank == 0:
        manifest_path = write_manifest(args, prefix_results)
        print(f"\nmanifest:         {manifest_path}")
        if not args.no_verify:
            verify_output(args)


if __name__ == "__main__":
    main()
