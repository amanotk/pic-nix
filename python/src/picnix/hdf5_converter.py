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


def parse_args():
    parser = argparse.ArgumentParser(
        prog="picnix-hdf5-convert",
        description="Convert PIC-NIX posix/mpiio diagnostics to HDF5 + VDS",
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
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
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


def main():
    args = parse_args()
    comm = get_comm()
    rank = comm.Get_rank()

    args.input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "hdf5"
    args.output_dir = args.output_dir.resolve()

    prefixes = (
        args.prefix if args.prefix is not None else discover_prefixes(args.input_dir)
    )
    prefix_results = []
    for prefix in prefixes:
        result = convert_prefix(args, prefix, comm)
        if rank == 0:
            prefix_results.append(result)

    if rank == 0:
        manifest_path = write_manifest(args, prefix_results)
        print(f"\nmanifest:         {manifest_path}")


if __name__ == "__main__":
    main()
