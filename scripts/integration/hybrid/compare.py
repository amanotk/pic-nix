#!/usr/bin/env python3
"""Compare PIC-NIX Hybrid beam output across decompositions and with legacy."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def load_chunk_diagnostics(root: Path) -> list[dict]:
    """Load all diagnostic chunks from a directory tree.

    Returns list of dicts: each has {meta, field, fluid, moment, particles}.
    """
    chunks = []
    for subdir in sorted(root.rglob("rank_*_chunk_*")):
        meta = json.loads((subdir / "meta.json").read_text())
        chunk = {
            "meta": meta,
            "field": read_npy(subdir / "field.npy"),
            "fluid": read_npy(subdir / "fluid.npy"),
            "moment": read_npy(subdir / "moment.npy"),
            "particles": [],
        }
        for s in range(meta["num_species"]):
            pp = subdir / f"particle_{s}.npy"
            chunk["particles"].append(read_npy(pp) if pp.exists() else np.empty((0, 7)))
        chunks.append(chunk)
    chunks.sort(key=lambda c: (c["meta"]["Lbz"], c["meta"]["Lby"], c["meta"]["Lbx"]))
    return chunks


def assemble_4d(chunks, key):
    """Concatenate 4D arrays [z,y,x,c] along z axis (chunks stacked in order)."""
    return np.concatenate([c[key] for c in chunks], axis=0)


def compare_chunk_arrays(ca, cb, key, tol=1e-14):
    if ca.shape != cb.shape:
        return f"{key} shape mismatch: {ca.shape} vs {cb.shape}"
    d = np.max(np.abs(ca - cb))
    if d > tol:
        return f"{key} max diff {d:.2e}"
    return None


def compare_decompositions(dir_a: Path, dir_b: Path, tol=1e-6):
    a = load_chunk_diagnostics(dir_a)
    b = load_chunk_diagnostics(dir_b)

    if len(a) != len(b):
        return [f"chunk count differs: {len(a)} vs {len(b)}"]

    errors = []
    for i, (ca, cb) in enumerate(zip(a, b)):
        label = f"chunk {i} (rank {ca['meta']['rank']})"
        for key in ("field", "fluid", "moment"):
            e = compare_chunk_arrays(ca[key], cb[key], f"{label} {key}", tol)
            if e:
                errors.append(e)

        # Compare particles by species
        for s in range(ca["meta"]["num_species"]):
            pa = ca["particles"][s]
            pb = cb["particles"][s]
            if pa.shape != pb.shape:
                errors.append(f"{label} species {s} particle count: {pa.shape[0]} vs {pb.shape[0]}")
                continue
            if pa.shape[0] == 0:
                continue
            # Sort by ID (column 6) and compare positions (columns 0:6)
            ida = pa[:, 6].view(np.int64)
            idb = pb[:, 6].view(np.int64)
            oa = np.argsort(ida)
            ob = np.argsort(idb)
            if not np.array_equal(ida[oa], idb[ob]):
                only_a = set(ida.tolist()) - set(idb.tolist())
                only_b = set(idb.tolist()) - set(ida.tolist())
                if only_a:
                    errors.append(f"{label} species {s}: {len(only_a)} particles only in A")
                if only_b:
                    errors.append(f"{label} species {s}: {len(only_b)} particles only in B")
                if len(only_a) <= 5:
                    errors.append(f"  only in A: {sorted(only_a)[:5]}")
                if len(only_b) <= 5:
                    errors.append(f"  only in B: {sorted(only_b)[:5]}")
                continue
            d = np.max(np.abs(pa[oa, :6] - pb[ob, :6]))
            if d > tol:
                errors.append(f"{label} species {s} pos/vel max diff {d:.2e}")

    return errors


def compute_diagnostics(root: Path, config_path: Path = None):
    chunks = load_chunk_diagnostics(root)
    field = assemble_4d(chunks, "field")  # [nz, ny, nx, 6]
    fluid = assemble_4d(chunks, "fluid")  # [nz, ny, nx, 10]
    moment = np.concatenate([c["moment"] for c in chunks], axis=0)  # [chunks, nz, ny, nx, Ns, 10]
    # Reshape moment to [nz_global, ny, nx, Ns, 10]
    moment_global = np.concatenate([moment[i] for i in range(moment.shape[0])], axis=0)

    gamma = 5.0 / 3.0
    if config_path and config_path.exists():
        import re
        m = re.search(r"gamma\s*=\s*([0-9.]+)", config_path.read_text())
        if m:
            gamma = float(m.group(1))

    magnetic = np.sum(field[..., 3:6]**2) / (8 * np.pi)
    electron = np.sum(0.5 * fluid[..., 0] * np.sum(fluid[..., 1:4]**2, axis=-1) + fluid[..., 4] / (gamma - 1))
    ion = np.sum(0.5 * fluid[..., 5] * np.sum(fluid[..., 6:9]**2, axis=-1) + fluid[..., 9] / (gamma - 1))

    kinetic = 0.0
    for c in chunks:
        for particles in c["particles"]:
            if particles.shape[0] > 0:
                kinetic += 0.5 * np.sum(particles[:, 3:6]**2)

    density_sum = np.sum(moment_global[..., 0], axis=-1)  # [nz_global, ny, nx]
    density_x = np.mean(density_sum, axis=(0, 1))
    mode_density = np.abs(np.fft.rfft(density_x)) / density_x.shape[0]

    tb = np.mean(field[..., 4] + 1j * field[..., 5], axis=(0, 1))
    mode_tb = np.abs(np.fft.rfft(tb)) / tb.shape[0]

    return {
        "magnetic": float(magnetic),
        "electron": float(electron),
        "ion": float(ion),
        "kinetic": float(kinetic),
        "total": float(magnetic + electron + ion + kinetic),
        "mode_density": [float(v) for v in mode_density],
        "mode_transverse_b": [float(v) for v in mode_tb],
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    dec = sub.add_parser("decomp", help="Verify decomposition independence")
    dec.add_argument("dir_a", type=Path)
    dec.add_argument("dir_b", type=Path)

    diag = sub.add_parser("diag", help="Print energy/mode diagnostics")
    diag.add_argument("data", type=Path)
    diag.add_argument("--config", type=Path)

    args = parser.parse_args()

    if args.cmd == "decomp":
        errors = compare_decompositions(args.dir_a, args.dir_b)
        if errors:
            print("FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            raise SystemExit(1)
        print("OK: decomposition independent")
        return

    if args.cmd == "diag":
        d = compute_diagnostics(args.data, args.config)
        print(json.dumps(d, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
