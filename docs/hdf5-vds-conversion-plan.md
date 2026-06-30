# HDF5 VDS Conversion Plan  
  
## Goal  
  
Reduce PIC-NIX diagnostic file count while keeping analysis convenient.  
The current direction is to convert original `.json` and `.data` diagnostics into one physical HDF5 file per prefix per step, then create a small HDF5 VDS index file per prefix.  
  
This gives users a single-file entry point without requiring a large serial merge into one physical HDF5 data file.  
It must support both `posix` and `mpiio` input layouts, and `picnix.Run` must remain agnostic from the user perspective.  
  
## Constraints  
  
- Real input data must be treated as read-only.  
- The converter treats original `.json` and `.data` files as read-only.  
- Production conversion defaults to `<input-dir>/hdf5`; temporary benchmark runs should override `--output-dir` and write under `tmp/scratch/...`.  
- Serial HDF5 must not have multiple MPI ranks writing the same file at the same time.  
- Compression must be parallelized by giving different ranks independent output files.  
- Support both `posix` and `mpiio` output.  
- `posix` conversion is primarily for file-count reduction plus modest compression.  
- `mpiio` conversion is mostly a near one-to-one repack into compressed HDF5 plus VDS metadata.  
- `picnix.Run` must hide storage differences from users: original `.json`/`.data`, raw per-step HDF5, and VDS-indexed HDF5 should all be readable through the same high-level API.  
  
## Proposed Layout  
  
Input layout example:  
  
```text  
<run>/data/node000000/field/00000000.json  
<run>/data/node000000/field/00000000.data  
<run>/data/node000001/field/00000000.json  
<run>/data/node000001/field/00000000.data  
```  
  
`mpiio` input layout example:  
  
```text  
<run>/data/field/00000000.json  
<run>/data/field/00000000.data  
<run>/data/field/00032000.json  
<run>/data/field/00032000.data  
```  
  
Converted output layout:  
  
```text  
<run>/data/hdf5/manifest.json  
<run>/data/hdf5/field/00000000.h5  
<run>/data/hdf5/field/00032000.h5  
<run>/data/hdf5/field/00064000.h5  
<run>/data/hdf5/field.vds.h5  
```  
  
Each physical step file contains real datasets for one prefix and one step:  
  
```text  
/00000000/uf  
/00000000/um  
```  
  
The VDS index contains virtual datasets pointing to the physical step files:  
  
```text  
/field/step  
/field/00000000/uf  
/field/00000000/um  
/field/00032000/uf  
/field/00032000/um  
```  
  
Use relative VDS source paths such as `field/00000000.h5` so the converted directory can be moved as a unit.  
  
## Dataset Shape Model  
  
Converted HDF5 datasets should preserve the raw diagnostic array shape before `picnix.Run` applies any analysis-facing conversion.  
The first dimension is the concatenation dimension.  
  
For chunked diagnostics such as `field`, the first dimension is the simulation chunk index within the step.  
For example, sampled real `field` data used:  
  
```text  
uf: (Nchunk_step, 1, 3, 3, 6)  
um: (Nchunk_step, 1, 3, 3, 2, 14)  
```  
  
For `posix`, `Nchunk_step` is produced by concatenating node-local arrays along axis 0 in node order.  
For `mpiio`, the original file should already contain the whole step array, so conversion is nearly one-to-one: read the original dataset and write the same shape to HDF5.  
  
For particle diagnostics, the first dimension is particle records.  
The old raw format stores records as `(Nparticle_step, 7)`, where the first six columns are particle values and the last 8-byte slot is the particle ID.  
Although the JSON metadata labels the full record as `f8`, the ID slot must be preserved by reinterpreting its bytes as `uint64`, not by numerically converting a float value.  
  
The Python reader should continue to own user-facing transformations.  
For example, `FieldDiagHandler.is_chunked_data_conversion_required()` currently causes `Run.read_at()` to convert chunk-major field arrays into global grid arrays.  
The HDF5 reader backend should return the same raw dictionary shape that the current JSON/data reader returns before that conversion.  
  
## Field And Particle Layouts  
  
Field and particle diagnostics should use different internal HDF5 layouts because their access and memory profiles are different.  
  
### Field  
  
Field data is chunk-major and relatively structured.  
The preferred physical step-file layout is one dataset per field quantity:  
  
```text  
<out>/field/00032000.h5  
  /00032000/uf    shape = (Nchunk_step, Nz_chunk, Ny_chunk, Nx_chunk, Nfield)  
  /00032000/um    shape = (Nchunk_step, Nz_chunk, Ny_chunk, Nx_chunk, Ns, Nmoment)  
```  
  
The converter should write these datasets in node batches or first-dimension slabs.  
The final result should still be one physical HDF5 file per prefix per step.  
  
### Particle  
  
Particle data can be much larger and has variable first-dimension length.  
The preferred external contract is still one physical HDF5 file per prefix per step, but the inside of that file may be split into particle blocks.  
  
Example physical layout:  
  
```text  
<out>/particle_full/02080000.h5  
  /02080000/particles/up00/value/block000000  
  /02080000/particles/up00/id/block000000  
  /02080000/particles/up01/value/block000000  
  /02080000/particles/up01/id/block000000  
```  
  
The logical datasets exposed to readers can be VDS datasets inside the same file:  
  
```text  
  /02080000/up00       virtual dataset, shape = (N, 6), dtype = float32 by default  
  /02080000/up00_id    virtual dataset, shape = (N,), dtype = uint64 by default  
  /02080000/up01       virtual dataset, shape = (N, 6), dtype = float32 by default  
  /02080000/up01_id    virtual dataset, shape = (N,), dtype = uint64 by default  
```  
  
This keeps the final data-file count at one HDF5 file per step while allowing memory-bounded writes.  
It also gives `picnix.Run` a simple logical dataset path.  
  
The block size should be chosen from `--max-buffer-mib` and `--target-chunk-mib`.  
For `posix`, particle blocks can naturally follow node batches.  
For `mpiio`, particle blocks can be first-dimension slabs read directly from the original contiguous raw dataset.  
  
This internal block layout does not solve the serial-HDF5 single-writer limit for one huge step, but it does solve memory pressure while preserving the desired one-file-per-step output.  
Parallelism then comes from converting multiple steps and prefixes concurrently.  
  
The planned `picnix` behavior is:  
  
```python  
run.read_at("particle", step)["up00"]         # (N, 6) values  
run.read_particle_id_at("particle", step)["up00"]  # (N,) IDs  
```  
  
No backward compatibility is required for old user-facing `(N, 7)` particle arrays.  
  
## HDF5 Chunking And Compression  
  
HDF5 compression is applied independently to HDF5 chunks.  
Compression ratio therefore depends on the HDF5 chunk shape, data ordering, and data entropy, not only on the logical dataset shape.  
  
A real `field/00032000` sample with 64 nodes showed:  
  
| HDF5 chunking | ratio to raw array bytes | note |  
|---|---:|---|  
| h5py auto chunks | 0.926 | acceptable default, but not optimal |  
| one simulation chunk per HDF5 chunk | 1.015 | bad; metadata and small-chunk overhead dominate |  
| 16 leading chunks per HDF5 chunk | 0.897 | better |  
| 128 leading chunks per HDF5 chunk | 0.876 | good |  
| 1024 leading chunks per HDF5 chunk | 0.870 | slightly better |  
| full dataset as one HDF5 chunk | 0.868 | best ratio, worse random access |  
  
Default chunking should group multiple entries along the first dimension instead of using one simulation chunk per HDF5 chunk.  
Use a target compressed chunk input size, for example 1-4 MiB before compression:  
  
```text  
row_bytes = product(shape[1:]) * dtype.itemsize  
chunk0 = clamp(target_chunk_bytes // row_bytes, 1, shape[0])  
hdf5_chunks = (chunk0, *shape[1:])  
```  
  
This preserves the leading-dimension access model while avoiding very small HDF5 chunks.  
Default conversion should use `float32` output and no compression.  
This is the current best speed/size tradeoff for the tested `run101 field` benchmark.  
Precision-preserving output remains available with `--field-dtype source` and `--particle-dtype source`.  
Compression remains available with `--compression gzip --compression-opts N`, but gzip is not the default because it made HDF5 write/compression the dominant cost in field tests.  
Particle IDs default to `uint64` and are stored separately.  
  
## Memory-Bounded Writing  
  
The converter must not require a full step array to fit in one rank's memory.  
On a Slurm cluster, a one-step-per-rank schedule is useful for avoiding concurrent writes to the same serial HDF5 file, but it can still be memory-bound if each rank materializes the whole step before writing.  
  
Metadata-only estimates from `tmp/ma05-tbn80-run001` show the risk:  
  
| prefix/step | estimated raw data per full step |  
|---|---:|  
| `field/00000000` | 1.030 GiB |  
| `field/00032000` | 1.030 GiB |  
| `particle/00000000` | 2.167 GiB |  
| `particle/01920000` | 2.955 GiB |  
| `particle_full/02080000` | 304.524 GiB |  
  
Therefore the physical step writer should be slab-based:  
  
1. Read only JSON metadata first.  
2. Compute each output dataset's full shape and dtype.  
3. Create the compressed HDF5 dataset with final shape and chosen HDF5 chunk layout.  
4. Stream input data in bounded batches along axis 0.  
5. Write each batch to the corresponding output slice.  
  
For `posix`, batches can be groups of node files.  
For each node batch, read node-local arrays and write the contiguous output slice for that batch.  
  
For `mpiio`, batches can be first-dimension slabs from the original `.data` file.  
Because each dataset is contiguous in the original raw file, a slab starting at row `i0` can be read from:  
  
```text  
file_offset = dataset_offset + i0 * product(shape[1:]) * dtype.itemsize  
```  
  
Required converter controls:  
  
- `--max-buffer-mib`: upper bound for temporary arrays held by one rank.  
- `--node-batch-size`: optional explicit node batch size for `posix`.  
- `--target-chunk-mib`: HDF5 chunk target size, independent from the larger streaming buffer limit.  
  
The memory target should cover input buffers plus HDF5 compression/chunk cache overhead.  
A conservative first implementation should keep only one dataset batch in memory at a time rather than loading all datasets for a step simultaneously.  
  
## Last-Resort Intra-Step Sharding  
  
One physical file per prefix per step still allows only one rank to write that step file with serial HDF5.  
For ordinary `field` and regular `particle`, slab or internal-block writing should avoid memory pressure and may be fast enough.  
For very large outputs such as `particle_full`, one rank streaming and compressing a 300 GiB step may become the wall-clock bottleneck even if memory is bounded.  
  
The preferred first attempt is still a single step file with internal particle blocks.  
Only if that proves too slow should we consider physical intra-step shards for selected large prefixes:  
  
```text  
<out>/particle_full/02080000/shard000000.h5  
<out>/particle_full/02080000/shard000001.h5  
<out>/particle_full/02080000/shard000002.h5  
<out>/particle_full.vds.h5  
```  
  
Each rank writes an independent first-dimension shard.  
The VDS index maps shards back into the logical per-step dataset.  
This violates the preferred one-file-per-step physical layout, so it should remain a last resort.  
It still reduces the original `posix` file count drastically and removes the single-writer bottleneck for huge steps.  
  
## Why Per-Step VDS Groups First  
  
A single rectangular time-stacked VDS dataset is attractive, for example `/field/uf[time, ...]`.  
However, the sampled real data showed node-subset shapes changing between steps because dynamic load balancing changes chunk ownership.  
Particle outputs can also have variable shape across steps.  
  
Therefore the first implementation should expose one virtual dataset per step and dataset name.  
This mirrors the existing diagnostic model and avoids requiring identical shapes across time.  
  
Time-stacked VDS datasets can be added later for prefixes and selections where global shapes are known to be stable.  
The initial reader integration should not require a time-stacked VDS.  
  
## Benchmark Evidence So Far  
  
Real data inspected: `tmp/ma05-tbn80-run001`, read-only.  
The directory is `posix` output with 3072 node directories.  
  
Estimated file-count reduction for one physical HDF5 file per prefix per step:  
  
| prefix | steps | original `.json`/`.data` files | step HDF5 files | reduction |  
|---|---:|---:|---:|---:|  
| `field` | 103 | 632,832 | 103 | 6144x |  
| `field_burst1` | 801 | 4,921,344 | 801 | 6144x |  
| `particle` | 103 | 632,832 | 103 | 6144x |  
| `particle_full` | 4 | 24,576 | 4 | 6144x |  
  
Compression samples using HDF5 `gzip=4` and `shuffle=True` showed modest size reduction: roughly 10-25 percent for sampled real output.  
The main win for `posix` is file-count reduction, not compression ratio.  
For `mpiio`, the file-count gain is smaller, but a 10-20 percent compressed-size reduction may still be worthwhile.  
  
VDS sanity benchmark on real `field` samples:  
  
- Converted 4 steps and 64 nodes into one HDF5 file per step.  
- Created one VDS index file with per-step virtual datasets.  
- VDS index size was 7574 bytes.  
- VDS index creation took 0.146 seconds.  
- Reading a small slice through VDS worked and confirmed `is_virtual == True`.  
- Total sampled original size was 71.765 MiB.  
- Total sampled HDF5 size was 62.796 MiB.  
- Sample ratio was 0.875.  
  
Sampled per-step shapes for `field` using 64 nodes:  
  
| step | `uf` shape | `um` shape | ratio |  
|---|---:|---:|---:|  
| `00000000` | `(9408, 1, 3, 3, 6)` | `(9408, 1, 3, 3, 2, 14)` | 0.774 |  
| `00032000` | `(7802, 1, 3, 3, 6)` | `(7802, 1, 3, 3, 2, 14)` | 0.924 |  
| `00064000` | `(6914, 1, 3, 3, 6)` | `(6914, 1, 3, 3, 2, 14)` | 0.920 |  
| `00096000` | `(6545, 1, 3, 3, 6)` | `(6545, 1, 3, 3, 2, 14)` | 0.914 |  
  
## Implementation Phases  
  
### Phase 1: Standalone Real-Data Converter Prototype  
  
Provide the converter as the packaged executable `picnix-hdf5-convert`, implemented by `picnix.hdf5_converter`.  
Keep `script/hdf5_converter.sh` only as a source-tree convenience wrapper.  
  
Required options:  
  
- `--input-dir`: path to the run `data` directory.  

Optional controls:  

- `--output-dir`: destination directory for converted HDF5 files; default is `<input-dir>/hdf5`.  
- `--prefix`: one or more diagnostic prefixes, for example `field particle`; default is to discover all diagnostic prefixes from the directory layout.  
- `--steps`: optional explicit step list or range for debugging/testing.  
- `--step-limit`: optional limit on discovered steps for debugging/testing.  
- `--compression`: default `none`.  
- `--compression-opts`: default `1` when gzip is requested.  
- `--field-dtype`: `float32`, `float64`, or `source`; default `float32`.  
- `--particle-dtype`: `float32`, `float64`, or `source`; default `float32`.  
- `--particle-id-dtype`: `uint64` or `int64`; default `uint64`.  
- `--target-chunk-mib`: target uncompressed HDF5 chunk size for compressed output; default `4`.  
- `--max-buffer-mib`: target maximum temporary data buffer per rank; default `1024`.  
- `--overwrite`: replace existing converted step files for the selected prefix or prefixes.  
- `--resume`: reuse existing valid step files and regenerate the VDS index.  
- `--no-vds`: only write physical step files.  
  
Behavior:  
  
- Use `--input-dir` as the protocol anchor. Do not require `profile.msgpack`; optional metadata sources can be added later without coupling conversion to profile internals.  
- If `--prefix` is omitted, discover prefixes from `<input-dir>/node000000/*/*.json` for `posix`, or `<input-dir>/*/*.json` for `mpiio`.  
- Detect `posix` or `mpiio` from directory structure.  
- Detect field-like or particle-like layout from prefix/dataset names.  
- Discover all nodes from `node??????` directories for `posix`.  
- Discover steps from `<input-dir>/node000000/<prefix>/*.json` for `posix`, or `<input-dir>/<prefix>/*.json` for `mpiio`.  
- Convert all datasets for each selected or discovered prefix.  
- For `posix`, perform concatenation by streaming node batches into output slices, not by materializing the whole step.  
- For `mpiio`, read first-dimension slabs from the single step `.data` file and preserve its dataset shape.  
- Write one HDF5 file for that prefix and step.  
- For field-like data, prefer direct logical datasets in the step group.  
- For particle-like data, allow internal block groups inside the same step file and expose logical datasets through same-file VDS or reader-side concatenation.  
- Use HDF5 chunks based on the target byte size and preserve the first dimension as the chunk/record dimension.  
- Store useful metadata as HDF5 attributes, including step, time, layout, source prefix, source node range, and conversion settings.  
- Write each step to a visible temporary HDF5 file, for example `00000000.h5.tmp`, and atomically rename it after successful completion.  
- With `--resume`, skip existing valid step files and rebuild the VDS index, so a scheduler kill can be resumed safely.  
- After physical step files are complete, rank 0 writes the VDS index file for each prefix and `manifest.json` for the converted output directory.  
- Normal conversion runs verification by default. Use `--no-verify` only for special cases such as `--no-vds` or staged debugging.  
- Standalone verification is available with `picnix-hdf5-convert verify --input-dir <run>/data`.  
- Original diagnostics can be removed only through `picnix-hdf5-convert remove-original --input-dir <run>/data` after verification passes and stamps `manifest.json`.  
- `remove-original` is intended for a normal interactive shell, not Slurm. It deletes only verified original `.json`/`.data` diagnostics, then removes empty prefix directories and empty `nodeXXXXXX` directories for POSIX output.  
- `remove-original --dry-run` reports what would be deleted; `remove-original --yes` skips the interactive confirmation.  
  
Parallelism:  
  
- Distribute steps across MPI ranks.  
- Each rank writes independent step HDF5 files.  
- Rank 0 creates the VDS index after a barrier.  
  
### Phase 2: Benchmark Matrix  
  
Run the converter on real data with output outside the source directory.  
Start with `field`, then move to `field_burst1`.  
  
Suggested matrix:  
  
| prefix | nodes | steps | purpose |  
|---|---:|---:|---|  
| `field` | 64 | 4 | smoke test and correctness |  
| `field` | 256 | 8 | scaling check |  
| `field` | 1024 | 16 | metadata/read pressure check |  
| `field` | 3072 | 8 | full-node partial-step check |  
| `field` | 3072 | 103 | full regular field conversion |  
| `field_burst1` | 3072 | 32 | burst-output stress sample |  
| `field_burst1` | 3072 | 801 | full burst conversion if earlier runs are acceptable |  
  
Measure:  
  
- Original file count represented.  
- Physical HDF5 file count.  
- VDS file count and size.  
- Conversion wall time.  
- Aggregate read throughput.  
- HDF5 write throughput.  
- Compression ratio.  
- VDS creation time.  
- VDS read sanity time.  
  
### Phase 3: Correctness Checks  
  
For sampled steps and datasets:  
  
- Compare HDF5 dataset shape against the concatenated original arrays.  
- Compare selected values or checksums.  
- Confirm all selected steps appear in the VDS index.  
- Confirm VDS reads return the same values as direct step-file reads.  
- Confirm moving the whole converted output directory preserves VDS readability.  
  
### Phase 4: Analysis API Integration  
  
After the file layout is validated, update Python analysis code to detect and read the HDF5 layouts.  
  
Initial target:  
  
- Allow `picnix.Run` or a lower-level diagnostic reader to accept a converted output directory.  
- Keep the user-facing access pattern close to existing `.json` and `.data` analysis.  
- Avoid forcing users to care whether data came from original `.json`/`.data`, raw per-step HDF5 files, or VDS-indexed HDF5 files.  
  
Implementation direction:  
  
- Keep `Run.read_at(prefix, step, pattern)` as the stable user-facing API.  
- Move storage-specific behavior behind diagnostic reader backends.  
- Existing backend: JSON/data files for `posix` and `mpiio`.  
- New backend: raw per-step HDF5 files.  
- New backend: VDS HDF5 index that resolves datasets to the same raw arrays.  
- `DiagHandler` should discover available storage for each prefix and expose the same `get_step()`, `get_time()`, and `read_at()` semantics.  
- `Run.read_at()` should continue applying existing post-processing such as field chunk-to-global conversion and auxiliary coordinates.  

Concrete reader plan:  

1. Keep `Run(profile, method=None, config=None)` and `Run.read_at(prefix, step, pattern=None)` as the public analysis entry points.  
2. Add a small storage-backend boundary below `DiagHandler`; suggested module name is `picnix.storage` or `picnix.diag_storage`.  
3. Implement `JsonDiagStorage` by moving the current JSON/data discovery and read logic out of `Run`/`DiagHandler` without changing behavior.  
4. Implement `Hdf5VdsDiagStorage` for converted output under `<basedir>/hdf5`.  
5. During `DiagHandler.setup()`, choose storage automatically per prefix: first check `<basedir>/hdf5/<prefix>.vds.h5`; if it is absent, fall back to raw JSON/data according to `iomode`.  
6. If `<basedir>/hdf5/<prefix>.vds.h5` exists but has an invalid layout, mismatched prefix, or missing step/time datasets, raise an error instead of silently falling back to raw data. This avoids accidentally mixing stale converted output with fresh raw output.  
7. Keep `manifest.json` useful for diagnostics and compatibility checks, but do not require it for basic reading; the per-prefix VDS file is the authoritative readable object.  
8. Let the HDF5 backend read `/<prefix>/step`, `/<prefix>/time`, and per-step groups such as `/<prefix>/00000000/<dataset>`.  
9. Preserve the existing regex `pattern` behavior by applying it to dataset names in the selected step group.  
10. Keep field post-processing in `Run.read_at()`: HDF5 returns the same chunk-major raw shape as JSON/data, then `convert_array_format()` and auxiliary coordinates are applied exactly once.  

Storage backend interface:  

```python
class DiagStorage:
    kind: str

    def get_step(self) -> np.ndarray: ...
    def get_time(self) -> np.ndarray: ...
    def get_time_at_step(self, step: int) -> float | None: ...
    def read_at(self, step: int, pattern: str) -> dict[str, np.ndarray]: ...
    def read_particle_id_at(self, step: int, pattern: str) -> dict[str, np.ndarray]: ...
```

`DiagHandler` remains responsible for diagnostic semantics: field data needs chunk-to-global conversion, field appends auxiliary coordinates, and particle data has a separate ID accessor.  
The storage backend is responsible only for locating files and returning raw logical arrays.  

Particle behavior should be normalized across storage formats:  

- `Run.read_at("particle", step)` returns species arrays with shape `(N, 6)`.  
- `Run.read_particle_id_at("particle", step)` returns species ID arrays with shape `(N,)` and integer dtype.  
- For HDF5, values come from logical datasets such as `/particle/00000000/up00`, and IDs come from `/particle/00000000/up00_id`.  
- For raw JSON/data, the particle backend reads `(N, 7)`, returns the first six columns as values, and byte-reinterprets the last raw 8-byte slot as `uint64` IDs.  
- Normal `read_at()` should not include `*_id` datasets by default; IDs should be accessed through `read_particle_id_at()` so users do not need to know which storage format is underneath.  

Suggested implementation stages:  

1. Add `JsonDiagStorage` and update `DiagHandler`/`Run` to delegate reads through it while keeping all current tests passing.  
2. Add `Hdf5VdsDiagStorage` and automatic per-prefix selection from `<basedir>/hdf5/<prefix>.vds.h5`.  
3. Add particle normalization and `Run.read_particle_id_at(prefix, step, pattern=None)` for both raw and HDF5 storage.  
4. Add tests using tiny synthetic raw diagnostics and converted HDF5 generated by `picnix.hdf5_converter`; verify the same user calls work whether `hdf5/` exists or not.  
5. Add real-data smoke checks against `tmp/ma05-tbn80-run101/data` after the full conversion finishes.  
6. Only after the VDS path is stable, consider direct raw step-file HDF5 fallback for `<basedir>/hdf5/<prefix>/<step>.h5` when the VDS file is missing.  

Testing plan:  

- Unit test raw fallback with a tiny POSIX fixture and no `hdf5/` directory.  
- Unit test automatic HDF5 selection when `<basedir>/hdf5/<prefix>.vds.h5` exists.  
- Unit test invalid existing HDF5 VDS raises instead of silently falling back.  
- Unit test `Run.read_at()` returns identical keys and compatible values for raw and HDF5 field diagnostics before and after field post-processing.  
- Unit test particle normalization: raw `(N, 7)` and HDF5 `(N, 6) + *_id` both produce `(N, 6)` values and matching `uint64` IDs.  
- Unit test `pattern` filtering for values and IDs.  
- Keep converter tests separate from reader tests, but reuse the synthetic fixture writer where practical.  
  
## Open Decisions  
  
- Whether the physical step HDF5 files should store step groups named by step, or use a flatter layout inside each step file.  
- Whether to keep one VDS index per prefix or one top-level VDS file for all prefixes.  
- Whether to add optional time-stacked VDS datasets for stable-shape fields.  
- Whether `particle_full` should use compression by default, given its large size and likely high entropy.  
- Whether to support block files later, such as one physical HDF5 file per prefix per N steps, if one file per step is still too many for very long runs.  
  
## Recommended Next Step  
  
Phase 1 prototype exists as `picnix.hdf5_converter` with the `picnix-hdf5-convert` console entry point.  
Real-data smoke tests passed for field, particle, VDS validation, and a 2-rank MPI run.  
Next, run staged real-data benchmarks using `tmp/scratch/...` output directories, starting with `field` at increasing node and step counts.  

For MPI runs in this environment, apply OpenMPI paths from `/home/u10446/shock-analysis/.shockrc`.  
For Slurm runs, prefer node-local `TMPDIR` or PRTE tmpdir MCA options to avoid OpenMPI session files on Lustre.  
