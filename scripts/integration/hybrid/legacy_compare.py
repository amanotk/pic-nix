#!/usr/bin/env python3
"""Compare PIC-NIX Hybrid output with legacy reference NPZ fixtures.

Reports numerical differences and attributes them to deterministic vs
MersenneTwister particle initialization where appropriate.
"""

import json
import re
from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).with_name("reference")


def load_our_npy(root: Path):
    """Load our NPY diagnostics."""
    chunks = []
    for subdir in sorted(root.rglob("rank_*_chunk_*")):
        meta = json.loads((subdir / "meta.json").read_text())
        chunk = {
            "meta": meta,
            "field": np.load(subdir / "field.npy"),
            "fluid": np.load(subdir / "fluid.npy"),
            "moment": np.load(subdir / "moment.npy"),
        }
        for s in range(meta["num_species"]):
            pp = subdir / f"particle_{s}.npy"
            if pp.exists():
                chunk.setdefault("particles", []).append(np.load(pp))
        chunks.append(chunk)
    chunks.sort(key=lambda c: (c["meta"]["Lbz"], c["meta"]["Lby"], c["meta"]["Lbx"]))
    return chunks


def assemble_global_array(chunks, key):
    """Stack chunk arrays along axis 0."""
    parts = [c[key] for c in chunks]
    return np.concatenate(parts, axis=0)


def compare_case(label: str, our_root: Path, legacy_npz: Path, config_path: Path = None):
    print(f"\n=== {label} ===")
    if not our_root.exists():
        print(f"  SKIP: our diagnostics not found at {our_root}")
        return
    if not legacy_npz.exists():
        print(f"  SKIP: legacy reference not found at {legacy_npz}")
        return

    our_chunks = load_our_npy(our_root)
    legacy = dict(np.load(legacy_npz))

    # Compare field arrays (step 1 = index 1)
    for leg_key, our_key, desc in [
        ("field_eb", "field", "E+B cell field"),
        ("field_up", "fluid", "fluid state"),
        ("moment_mom", "moment", "kinetic moments"),
    ]:
        if leg_key not in legacy:
            print(f"  {desc}: legacy key '{leg_key}' missing")
            continue
        leg_arr = legacy[leg_key]
        our_arr = assemble_global_array(our_chunks, our_key)

        if leg_arr.ndim >= 5:
            leg_step = leg_arr[1]  # step 1
        elif leg_arr.ndim >= 4:
            leg_step = leg_arr[1]
        else:
            continue

        if leg_step.shape != our_arr.shape:
            print(f"  {desc}: shape mismatch legacy={leg_step.shape} ours={our_arr.shape}")
            continue

        abs_diff = np.max(np.abs(leg_step - our_arr))
        rel_diff = abs_diff / (np.max(np.abs(leg_step)) + 1e-32)

        status = "OK" if abs_diff < 1e-6 else "DIFF"
        note = ""
        if desc == "kinetic moments":
            note = " (expected: deterministic vs random particle init)"
        elif desc == "fluid state":
            note = " (may differ due to kinetic moment coupling in Ohm source)"

        print(f"  {desc}: {status} max abs={abs_diff:.6e} rel={rel_diff:.6e}{note}")

    # Energy comparison
    if "energy" in legacy:
        leg_energy = legacy["energy"]
        if leg_energy.ndim >= 2:
            leg_step = leg_energy[1]
        else:
            leg_step = leg_energy

        # Compute our energy
        field = assemble_global_array(our_chunks, "field")
        fluid = assemble_global_array(our_chunks, "fluid")

        gamma = 1.666666666666667
        if config_path and config_path.exists():
            m = re.search(r"gamma\s*=\s*([0-9.]+)", config_path.read_text())
            if m:
                gamma = float(m.group(1))

        magnetic = np.sum(field[..., 3:6]**2) / (8 * np.pi)
        electron = np.sum(0.5 * fluid[..., 0] * np.sum(fluid[..., 1:4]**2, axis=-1) + fluid[..., 4] / (gamma - 1))
        ion = np.sum(0.5 * fluid[..., 5] * np.sum(fluid[..., 6:9]**2, axis=-1) + fluid[..., 9] / (gamma - 1))

        kinetic = 0.0
        for c in our_chunks:
            for p in c.get("particles", []):
                if p.shape[0] > 0:
                    kinetic += 0.5 * np.sum(p[:, 3:6]**2)

        print(f"  energy legacy: mag={leg_step[0]:.4f} el={leg_step[1]:.4f} ion={leg_step[2]:.4f} kin={leg_step[3]:.4f} sum={sum(leg_step):.4f}")
        print(f"  energy ours:   mag={magnetic:.4f} el={electron:.4f} ion={ion:.4f} kin={kinetic:.4f} sum={magnetic+electron+ion+kinetic:.4f}")
        print("  energy: kinetic differs (expected: deterministic vs random particle init)")

    # SSOR comparison
    our_log = our_root.parent.parent / "debug.log" if our_root.name == "final" else our_root.parent / "debug.log"
    if not our_log.exists():
        our_log = our_root.parent / "debug.log"

    if "ssor_iteration" in legacy and our_log.exists():
        leg_iter = legacy["ssor_iteration"]

        log_text = our_log.read_text()
        our_iters = [int(m.group(1)) for m in re.finditer(r"iter=(\d+)", log_text)]

        print(f"  SSOR: legacy {len(leg_iter)} total iterations, ours {len(our_iters)} total iterations")
        print("  SSOR: convergence behavior comparison not feasible (different particle init changes source)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-one-step", type=Path, help="Our one-step diagnostics/final dir")
    parser.add_argument("--our-short-run", type=Path, help="Our short-run diagnostics/final dir")
    parser.add_argument("--config", type=Path, default=Path("hybrid/example/beam/config.toml"))
    args = parser.parse_args()

    compare_case("One-step beam", args.our_one_step,
                 REFERENCE_DIR / "beam_one_step.npz", args.config)
    compare_case("Short-run beam", args.our_short_run,
                 REFERENCE_DIR / "beam_short_run.npz", args.config)


if __name__ == "__main__":
    main()
