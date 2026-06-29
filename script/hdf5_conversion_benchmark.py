#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Synthetic benchmark for PIC-NIX output consolidation.

This benchmark intentionally does not depend on real simulation output.  It
creates a small synthetic output layout.  In posix mode this is

    <workdir>/posix/node000000/<prefix>/00000000.{json,data}

and in mpiio mode this is

    <workdir>/mpiio/<prefix>/00000000.{json,data}

then measures two consolidation stages:

1. original files -> block HDF5 files, each holding one or more steps
2. block HDF5 files -> one final HDF5 file per prefix

Run with mpi4py, for example:

    mpirun -np 4 python script/hdf5_conversion_benchmark.py --overwrite

The purpose is to measure whether one-file-per-step is a useful intermediate
for posix mode and whether the final merge behaves like a fast copy or an
expensive recompression on the local h5py/HDF5 build.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI


DEFAULT_PREFIX = "field"
STEP_WIDTH = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic PIC-NIX output -> HDF5 consolidation"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("tmp/hdf5-conversion-benchmark"),
        help="benchmark workspace (default: tmp/hdf5-conversion-benchmark)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove an existing benchmark workspace before running",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--iomode",
        choices=("posix", "mpiio"),
        default="posix",
        help="synthetic original layout to generate",
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--chunks-per-node", type=int, default=8)
    parser.add_argument(
        "--step-block-size",
        type=int,
        default=1,
        help="number of steps per intermediate HDF5 file (1 = per-step HDF5)",
    )
    parser.add_argument(
        "--node-mb",
        type=float,
        default=4.0,
        help="approximate raw data MiB per node per step before metadata",
    )
    parser.add_argument(
        "--entropy",
        choices=("smooth", "random", "mixed"),
        default="mixed",
        help="synthetic data entropy: smooth compresses well, random does not",
    )
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="disable the HDF5 shuffle filter (enabled by default)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="skip final per-prefix merge and only benchmark block HDF5",
    )
    return parser.parse_args()


def step_name(step):
    return f"{step:0{STEP_WIDTH}d}"


def step_blocks(args):
    block_size = max(1, args.step_block_size)
    return [
        list(range(start, min(args.steps, start + block_size)))
        for start in range(0, args.steps, block_size)
    ]


def block_name(block):
    if len(block) == 1:
        return f"{step_name(block[0])}.h5"
    return f"{step_name(block[0])}-{step_name(block[-1])}.h5"


def bytes_to_mib(nbytes):
    return nbytes / (1024.0 * 1024.0)


def directory_size(path):
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def file_size(path):
    if not path.exists():
        return 0
    return path.stat().st_size


def prepare_workspace(args, rank):
    if rank == 0:
        if args.workdir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{args.workdir} exists; pass --overwrite or choose --workdir"
                )
            shutil.rmtree(args.workdir)
        (args.workdir / args.iomode).mkdir(parents=True)
        (args.workdir / "step-h5" / args.prefix).mkdir(parents=True)
        (args.workdir / "final").mkdir(parents=True)


def make_dataset(rng, shape, entropy, step, node):
    if entropy == "random":
        return rng.standard_normal(shape, dtype=np.float64)

    chunks, values = shape
    x = np.linspace(0.0, 2.0 * np.pi, values, endpoint=False, dtype=np.float64)
    base = np.sin(x + 0.1 * step) + np.cos(0.5 * x + node)
    data = np.broadcast_to(base, shape).copy()
    data += 1.0e-6 * (node + 1) * np.arange(chunks, dtype=np.float64)[:, None]
    return data


def build_synthetic_datasets(args, step, node):
    raw_bytes = int(args.node_mb * 1024 * 1024)
    values_total = max(args.chunks_per_node, raw_bytes // np.dtype(np.float64).itemsize)
    values_per_chunk = max(1, values_total // args.chunks_per_node)
    shape = (args.chunks_per_node, values_per_chunk)
    rng = np.random.default_rng(seed=step * 1000003 + node)

    if args.entropy == "mixed":
        smooth = make_dataset(rng, shape, "smooth", step, node)
        random = make_dataset(rng, shape, "random", step, node)
        return {"smooth": smooth, "random": random}

    return {"data": make_dataset(rng, shape, args.entropy, step, node)}


def write_posix_node_file(args, step, node):
    dirname = args.workdir / "posix" / f"node{node:06d}" / args.prefix
    dirname.mkdir(parents=True, exist_ok=True)
    stem = step_name(step)
    data_path = dirname / f"{stem}.data"
    json_path = dirname / f"{stem}.json"
    datasets = build_synthetic_datasets(args, step, node)

    offset = 0
    metadata = {}
    with open(data_path, "wb") as data_fp:
        for name, array in datasets.items():
            contiguous = np.ascontiguousarray(array)
            payload = contiguous.tobytes(order="C")
            data_fp.write(payload)
            metadata[name] = {
                "datatype": "f8",
                "description": f"synthetic {name}",
                "offset": offset,
                "size": len(payload),
                "ndim": contiguous.ndim,
                "shape": list(contiguous.shape),
            }
            offset += len(payload)

    root = {
        "meta": {
            "endian": 1,
            "rawfile": data_path.name,
            "layout": 1,
            "time": float(step),
            "step": int(step),
            "chunk_id_range": [
                node * args.chunks_per_node,
                (node + 1) * args.chunks_per_node - 1,
            ],
        },
        "dataset": metadata,
    }
    with open(json_path, "w") as json_fp:
        json.dump(root, json_fp, indent=2)


def write_data_and_json(data_path, json_path, datasets, meta):
    offset = 0
    metadata = {}
    with open(data_path, "wb") as data_fp:
        for name, array in datasets.items():
            contiguous = np.ascontiguousarray(array)
            payload = contiguous.tobytes(order="C")
            data_fp.write(payload)
            metadata[name] = {
                "datatype": "f8",
                "description": f"synthetic {name}",
                "offset": offset,
                "size": len(payload),
                "ndim": contiguous.ndim,
                "shape": list(contiguous.shape),
            }
            offset += len(payload)

    with open(json_path, "w") as json_fp:
        json.dump({"meta": meta, "dataset": metadata}, json_fp, indent=2)


def write_mpiio_step_file(args, step):
    dirname = args.workdir / "mpiio" / args.prefix
    dirname.mkdir(parents=True, exist_ok=True)
    stem = step_name(step)
    data_path = dirname / f"{stem}.data"
    json_path = dirname / f"{stem}.json"

    node_datasets = [
        build_synthetic_datasets(args, step, node) for node in range(args.nodes)
    ]
    names = sorted(node_datasets[0].keys())
    datasets = {
        name: np.concatenate([node_data[name] for node_data in node_datasets], axis=0)
        for name in names
    }
    meta = {
        "endian": 1,
        "rawfile": data_path.name,
        "layout": 1,
        "time": float(step),
        "step": int(step),
        "chunk_id_range": [0, args.nodes * args.chunks_per_node - 1],
    }
    write_data_and_json(data_path, json_path, datasets, meta)


def read_posix_node_file(json_path):
    with open(json_path, "r") as json_fp:
        root = json.load(json_fp)
    dirname = json_path.parent
    data_path = dirname / root["meta"]["rawfile"]
    datasets = {}
    with open(data_path, "rb") as data_fp:
        for name, info in root["dataset"].items():
            data_fp.seek(info["offset"])
            payload = data_fp.read(info["size"])
            array = np.frombuffer(payload, dtype=np.float64).reshape(info["shape"])
            datasets[name] = array.copy()
    return root["meta"], datasets


def read_single_original_file(json_path):
    return read_posix_node_file(json_path)


def read_original_step(args, step):
    stem = step_name(step)
    if args.iomode == "posix":
        inputs = [
            args.workdir / "posix" / f"node{node:06d}" / args.prefix / f"{stem}.json"
            for node in range(args.nodes)
        ]
        original = [read_posix_node_file(path) for path in inputs]
        names = sorted(original[0][1].keys())
        datasets = {
            name: np.concatenate([node_data[name] for _, node_data in original], axis=0)
            for name in names
        }
    else:
        _, datasets = read_single_original_file(
            args.workdir / "mpiio" / args.prefix / f"{stem}.json"
        )
    return datasets


def write_step_group(h5fp, args, step, datasets):
    stem = step_name(step)
    group = h5fp.create_group(stem)
    group.attrs["step"] = int(step)
    group.attrs["time"] = float(step)
    group.attrs["layout"] = 1
    group.attrs["endian"] = 1
    group.attrs["chunk_id_range"] = np.array(
        [0, args.nodes * args.chunks_per_node - 1], dtype=np.int32
    )
    for name, array in sorted(datasets.items()):
        group.create_dataset(
            name,
            data=array,
            compression=args.compression,
            compression_opts=args.compression_opts,
            shuffle=not args.no_shuffle,
        )


def consolidate_block(args, block):
    out_path = args.workdir / "step-h5" / args.prefix / block_name(block)

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as h5fp:
        for step in block:
            write_step_group(h5fp, args, step, read_original_step(args, step))


def merge_step_files(args):
    final_path = args.workdir / "final" / f"{args.prefix}.h5"
    if final_path.exists():
        final_path.unlink()

    steps = []
    times = []
    with h5py.File(final_path, "w") as dest:
        for step in range(args.steps):
            steps.append(step)
            times.append(float(step))
        for block in step_blocks(args):
            block_path = args.workdir / "step-h5" / args.prefix / block_name(block)
            with h5py.File(block_path, "r") as src:
                for stem in sorted(src.keys()):
                    src.copy(src[stem], dest, name=stem)
        dest.create_dataset("step", data=np.array(steps, dtype=np.int32))
        dest.create_dataset("time", data=np.array(times, dtype=np.float64))
    return final_path


def parallel_for(comm, items, func):
    rank = comm.Get_rank()
    size = comm.Get_size()
    local = items[rank::size]
    t0 = time.perf_counter()
    for item in local:
        func(item)
    comm.barrier()
    elapsed = time.perf_counter() - t0
    return max(comm.allgather(elapsed))


def print_summary(args, generate_time, step_time, merge_time):
    original_size = directory_size(args.workdir / args.iomode)
    step_h5_size = directory_size(args.workdir / "step-h5")
    final_size = file_size(args.workdir / "final" / f"{args.prefix}.h5")

    original_files = args.steps * 2
    if args.iomode == "posix":
        original_files *= args.nodes
    block_files = len(step_blocks(args))
    final_files = 1 if not args.no_merge else 0

    print("\nHDF5 consolidation benchmark")
    print("============================")
    print(f"workdir:              {args.workdir}")
    print(f"iomode:               {args.iomode}")
    print(f"mpi ranks:            {MPI.COMM_WORLD.Get_size()}")
    print(f"nodes:                {args.nodes}")
    print(f"steps:                {args.steps}")
    print(f"step block size:      {max(1, args.step_block_size)}")
    print(f"node MiB/step:        {args.node_mb:.3g}")
    print(f"entropy:              {args.entropy}")
    print(f"compression:          {args.compression}-{args.compression_opts}")
    print(f"shuffle:              {not args.no_shuffle}")
    print("")
    print("timing")
    print(f"  generate original:  {generate_time:9.3f} s")
    print(f"  original -> block h5:{step_time:8.3f} s")
    if not args.no_merge:
        print(f"  block h5 -> final:  {merge_time:9.3f} s")
    print("")
    print("size")
    print(f"  original {args.iomode}:    {bytes_to_mib(original_size):9.3f} MiB")
    print(f"  block h5:           {bytes_to_mib(step_h5_size):9.3f} MiB")
    if step_h5_size > 0:
        print(f"  block/original:     {step_h5_size / original_size:9.3f}")
    if not args.no_merge:
        print(f"  final prefix h5:    {bytes_to_mib(final_size):9.3f} MiB")
        if final_size > 0:
            print(f"  final/original:     {final_size / original_size:9.3f}")
            print(f"  final/block-h5:     {final_size / step_h5_size:9.3f}")
    print("")
    print("file count")
    print(f"  original {args.iomode}:    {original_files}")
    print(f"  block h5:           {block_files}")
    if not args.no_merge:
        print(f"  final prefix h5:    {final_files}")


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    prepare_workspace(args, rank)
    comm.barrier()

    if args.iomode == "posix":
        generate_items = [
            (step, node) for step in range(args.steps) for node in range(args.nodes)
        ]
        generate_time = parallel_for(
            comm,
            generate_items,
            lambda item: write_posix_node_file(args, item[0], item[1]),
        )
    else:
        generate_time = parallel_for(
            comm,
            list(range(args.steps)),
            lambda step: write_mpiio_step_file(args, step),
        )

    step_time = parallel_for(
        comm,
        step_blocks(args),
        lambda block: consolidate_block(args, block),
    )

    merge_time = 0.0
    if not args.no_merge:
        comm.barrier()
        if rank == 0:
            t0 = time.perf_counter()
            merge_step_files(args)
            merge_time = time.perf_counter() - t0
        merge_time = comm.bcast(merge_time, root=0)
        comm.barrier()

    if rank == 0:
        print_summary(args, generate_time, step_time, merge_time)


if __name__ == "__main__":
    main()
