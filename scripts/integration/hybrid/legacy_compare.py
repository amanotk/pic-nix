#!/usr/bin/env python3
"""Compare one canonical PIC-NIX Hybrid snapshot with a legacy fixture."""

import argparse
import sys
from pathlib import Path

import numpy as np

from compare import DiagnosticError, compute_diagnostics, load_snapshot


def compare_legacy(
    snapshot_path: Path,
    reference_path: Path,
    config_path: Path,
    *,
    index: int | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> list[str]:
    """Return every mismatch with the selected legacy fixture state."""
    if not reference_path.is_file():
        raise DiagnosticError(f"legacy reference does not exist: {reference_path}")
    snapshot = load_snapshot(snapshot_path)
    try:
        reference = np.load(reference_path, allow_pickle=False)
    except Exception as error:
        raise DiagnosticError(f"cannot load {reference_path}: {error}") from error

    step_index = snapshot["meta"]["step"] if index is None else index
    errors = []
    array_keys = {
        "field": "field_eb",
        "fluid": "field_up",
        "moment": "moment_mom",
    }
    for snapshot_key, reference_key in array_keys.items():
        if reference_key not in reference:
            errors.append(f"legacy reference is missing {reference_key}")
            continue
        if step_index < 0 or step_index >= reference[reference_key].shape[0]:
            errors.append(
                f"legacy reference has no index {step_index} for {reference_key}"
            )
            continue
        actual = snapshot["arrays"][snapshot_key]
        expected = reference[reference_key][step_index]
        if actual.shape != expected.shape:
            errors.append(
                f"{snapshot_key} shape differs: {actual.shape} vs {expected.shape}"
            )
        elif not np.allclose(actual, expected, atol=atol, rtol=rtol):
            errors.append(
                f"{snapshot_key} max absolute difference {np.max(np.abs(actual - expected)):.6e}"
            )

    for species, (actual_ids, actual_state) in enumerate(snapshot["particles"]):
        id_key = f"particle_{species:02d}_id"
        state_key = f"particle_{species:02d}_state"
        if id_key not in reference or state_key not in reference:
            errors.append(
                f"legacy reference is missing particles for species {species}"
            )
            continue
        if step_index < 0 or step_index >= reference[id_key].shape[0]:
            errors.append(f"legacy reference has no particle index {step_index}")
            continue
        expected_ids = reference[id_key][step_index]
        expected_state = reference[state_key][step_index]
        if not np.array_equal(actual_ids, expected_ids):
            errors.append(f"species {species} particle IDs differ")
        elif not np.allclose(actual_state, expected_state, atol=atol, rtol=rtol):
            errors.append(
                f"species {species} particle state max absolute difference "
                f"{np.max(np.abs(actual_state - expected_state)):.6e}"
            )

    diagnostics = compute_diagnostics(snapshot_path, config_path)
    if "energy" not in reference or step_index >= reference["energy"].shape[0]:
        errors.append(f"legacy reference has no energy index {step_index}")
    else:
        actual_energy = np.asarray(
            [
                diagnostics["magnetic"],
                diagnostics["electron"],
                diagnostics["ion"],
                diagnostics["kinetic"],
            ]
        )
        expected_energy = reference["energy"][step_index]
        if not np.allclose(actual_energy, expected_energy, atol=atol, rtol=rtol):
            errors.append(
                f"energy max absolute difference {np.max(np.abs(actual_energy - expected_energy)):.6e}"
            )

    for diagnostic_key in ("mode_density", "mode_transverse_b"):
        if (
            diagnostic_key not in reference
            or step_index >= reference[diagnostic_key].shape[0]
        ):
            errors.append(
                f"legacy reference has no {diagnostic_key} index {step_index}"
            )
            continue
        actual = np.asarray(diagnostics[diagnostic_key])
        expected = np.abs(reference[diagnostic_key][step_index])
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=atol, rtol=rtol
        ):
            errors.append(f"{diagnostic_key} differs")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    args = parser.parse_args()
    try:
        errors = compare_legacy(
            args.snapshot,
            args.reference,
            args.config,
            index=args.index,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (DiagnosticError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("snapshot matches legacy reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
