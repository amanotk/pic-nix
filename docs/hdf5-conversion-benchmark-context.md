# HDF5 Conversion Benchmark Context

## Purpose

This branch explores HDF5 conversion strategies for PIC-NIX diagnostic output.  
The immediate production pain point is `posix` I/O mode, where each node writes per-step `.json` and `.data` files.  
For long supercomputer runs this can produce millions of files, causing inode pressure and poor transfer/read performance.  

The long-term goal is to let `picnix.Run` read converted output transparently, so analysis code does not care whether data came from original `.json`/`.data` files or HDF5.  

## Current Branch

Branch name: `feature/hdf5-conversion-benchmark`  

This branch currently contains a synthetic benchmark only.  
It is intentionally not a production converter yet.  

## Benchmark Script

Script: `script/hdf5_conversion_benchmark.py`  

The benchmark generates synthetic PIC-NIX-like diagnostic output in either layout:  

```text
posix:
  <workdir>/posix/node000000/<prefix>/00000000.json
  <workdir>/posix/node000000/<prefix>/00000000.data

mpiio:
  <workdir>/mpiio/<prefix>/00000000.json
  <workdir>/mpiio/<prefix>/00000000.data
```

It then measures consolidation into HDF5:  

```text
original node/mpiio files -> block HDF5 files -> final per-prefix HDF5 file
```

Block HDF5 means each intermediate HDF5 file contains one or more timestep groups.  
`--step-block-size 1` gives one HDF5 file per step.  
Larger values, for example `--step-block-size 64`, reduce intermediate file count further.  

## Example Commands

Small smoke test:  

```sh
mpirun -np 2 python script/hdf5_conversion_benchmark.py \
  --workdir tmp/hdf5-benchmark-smoke \
  --overwrite \
  --iomode posix \
  --nodes 2 \
  --steps 8 \
  --step-block-size 4 \
  --node-mb 0.25 \
  --chunks-per-node 2 \
  --entropy mixed
```

Matched posix/mpiio medium tests used during discussion:  

```sh
mpirun -np 4 python script/hdf5_conversion_benchmark.py \
  --workdir tmp/hdf5-benchmark-posix-medium \
  --overwrite \
  --iomode posix \
  --nodes 8 \
  --steps 16 \
  --node-mb 1 \
  --chunks-per-node 4 \
  --entropy mixed

mpirun -np 4 python script/hdf5_conversion_benchmark.py \
  --workdir tmp/hdf5-benchmark-mpiio-medium \
  --overwrite \
  --iomode mpiio \
  --nodes 8 \
  --steps 16 \
  --node-mb 1 \
  --chunks-per-node 4 \
  --entropy mixed
```

## Observations So Far

Compression performance should be essentially the same for `mpiio` and `posix` if the arrays, chunking, compression filter, and shuffle settings are the same.  
The difference is input topology: `mpiio` has fewer/larger original files, while `posix` has many node-local files.  

Small local results showed similar compression ratios for matched `posix` and `mpiio` synthetic data.  
For a medium mixed case:  

```text
posix:
  original -> block h5: ~1.9 s
  block h5 -> final:   ~0.37 s
  final/original:      ~0.797
  files:               256 -> 16 -> 1

mpiio:
  original -> block h5: ~1.8 s
  block h5 -> final:   ~0.32 s
  final/original:      ~0.797
  files:               32 -> 16 -> 1
```

These local timings are only a smoke test.  
They are affected by small data size, few files, and OS page cache.  
They do not prove production scalability on a supercomputer filesystem.  

## Important Design Lessons

Serial HDF5 does not allow multiple processes to write the same HDF5 file concurrently.  
Any design must guarantee that no two MPI ranks open the same HDF5 file for writing at the same time.  

h5py compression happens during the write call.  
Therefore, a single-writer design would serialize compression and likely perform poorly.  

To parallelize compression without parallel HDF5, each rank should write independent HDF5 files.  
The current benchmark uses block HDF5 files as the safe intermediate.  

The final merge is serial per final file and performs disk I/O.  
It looked fast locally, but we still need real-data testing to know if it is acceptable at scale.  

## Why Block HDF5 Matters

One HDF5 file per step can still be too many files for very long runs.  
Block HDF5 reduces intermediate file count to:  

```text
ceil(Nsteps / step_block_size) * Nprefixes
```

For `posix`, this is a major reduction from:  

```text
2 * Nnodes * Nsteps * Nprefixes
```

Example: with 64 nodes, 10000 steps, 4 prefixes, and block size 100:  

```text
original posix files: 2 * 64 * 10000 * 4 = 5,120,000
block HDF5 files:    100 * 4 = 400
final HDF5 files:    4
```

## Data Size Expectations

Size reduction depends on data entropy.  
Smooth field-like arrays compress better than random particle-like arrays.  
HDF5 `shuffle` is enabled by default in the benchmark because it helps gzip on float64 data.  

Synthetic observations:  

- smooth data: visible reduction with gzip+shuffle,  
- random data: weak reduction,  
- mixed data: intermediate behavior.  

Real PIC output must be benchmarked because actual field/moment/particle entropy will decide the ratio.  

## Proposed Supercomputer Benchmark Plan

Use real data and test one prefix first, likely `field`.  
Inputs needed:  

- path to `profile.msgpack`,  
- scratch/work directory,  
- allowed MPI rank count,  
- wall-clock limit,  
- selected prefix,  
- candidate block sizes such as `1`, `8`, `32`, `128`, or by target block file size.  

Measure:  

- original posix read time,  
- original -> block HDF5 conversion time,  
- block HDF5 file count and size,  
- optional block -> final merge time,  
- final file size,  
- whether merge time remains acceptable at realistic scale.  

If final merge is too expensive, the fallback design is to teach `picnix.Run` to read directly from block HDF5 files.  
That would still solve most of the inode problem and avoid requiring a huge serial final merge.  

## Current Recommendation

Do not implement the full production converter yet.  
First run realistic benchmarks on the supercomputer with real output.  
Focus on `posix -> block HDF5`, then decide whether `block HDF5 -> final per-prefix HDF5` is worth it.  
