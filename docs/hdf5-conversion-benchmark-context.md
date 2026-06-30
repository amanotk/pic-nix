# HDF5 Conversion Benchmark Context  
  
This document is kept only as historical context.  
The original synthetic benchmark script, `script/hdf5_conversion_benchmark.py`, has been removed.  
  
The current converter is the packaged command:  
  
```sh
picnix-hdf5-convert --input-dir /path/to/run/data
```
  
This location-independent shell wrapper is also available when `picnix` is importable by the selected Python:  
  
```sh
script/hdf5_converter.sh --input-dir /path/to/run/data
```
  
The current design, benchmark results, layout, and reader integration plan are documented in `docs/hdf5-vds-conversion-plan.md`.  
