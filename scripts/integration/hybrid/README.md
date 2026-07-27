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

## Validate The Port  

The CTest suite exercises the Hybrid port against these fixtures and against decomposition/restart/DLB invariants. Configure with tests enabled, then run:  

```bash
ctest --test-dir build --output-on-failure
```

Important Hybrid acceptance checks include:  

- `test_hybrid_shared_state_step0_legacy` and `test_hybrid_shared_state_step1_legacy` compare a one-rank/one-chunk port run initialized from the canonical legacy state at frozen strict tolerances.  
- `test_hybrid_shared_state_ssor_legacy` compares legacy SSOR2 stage offsets, iteration counts, and rounded residual history.  
- `test_hybrid_beam_*decomposition*` compares deterministic beam output across rank counts and chunk topologies.  
- `test_hybrid_beam_restart_np*` compares continuous versus resumed output.  
- `test_hybrid_beam_rebalance_restart_np2_*` proves a forced chunk ownership change, compares fixed versus rebalanced output, and verifies restart after migration.  

The current faithful backend is the legacy SSOR2 Ohm solver. The optional elliptic/PETSc backend is intentionally deferred and must preserve the same discrete Ohm system when implemented.  
