# Hybrid Reference Integration  

`reference.py` generates canonical validation fixtures from Hybrid3D commit `72abdbc`. It runs each configured beam case twice and refuses to update the fixtures unless every canonical array is exactly reproducible.  

## Requirements  

- A built Hybrid3D `kinetic/beam` executable.  
- `mpiexec`.  
- Python with NumPy and h5py.  

## Generate Fixtures  

```bash
python3 scripts/integration/hybrid/reference.py \
  --executable /path/to/hybrid3d-build/kinetic/beam \
  --output /tmp/hybrid3d-reference-runs \
  --repeats 2
```

The one-step and four-step cases use one MPI rank, an $8 \times 2 \times 2$ grid, and four particles per cell per species. Configuration, canonical NPZ data, and per-array SHA-256 manifests are stored under `reference/`.  

## Canonical Data  

- Fields and fluids retain `[time,z,y,x,component]` order.  
- Moments retain `[time,z,y,x,species,component]` order.  
- Particle IDs are decoded from raw `int64` bits and records are sorted by ID per species and time.  
- Derived arrays include legacy energy partitions, density and transverse-magnetic Fourier modes, and SSOR2 convergence history.  

The fixtures are for validation, not initialization. The production port uses decomposition-independent initialization rather than Hybrid3D's rank-seeded random stream.  

Verify the checked-in fixture contents against their manifests without a Hybrid3D executable:  

```bash
python3 scripts/integration/hybrid/reference.py --verify-only
```
