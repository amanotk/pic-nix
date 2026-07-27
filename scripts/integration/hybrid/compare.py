#!/usr/bin/env python3
"""Validate and compare canonical PIC-NIX Hybrid snapshots."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


class DiagnosticError(RuntimeError):
    """A snapshot is missing, malformed, or internally inconsistent."""


def _load_array(
    path: Path, ndim: int, prefix: tuple[int, ...], *, require_finite: bool = True
) -> np.ndarray:
    if not path.is_file():
        raise DiagnosticError(f"missing array: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:
        raise DiagnosticError(f"cannot load {path}: {error}") from error
    if array.ndim != ndim:
        raise DiagnosticError(f"{path}: expected {ndim} dimensions, got {array.ndim}")
    if array.shape[: len(prefix)] != prefix:
        raise DiagnosticError(
            f"{path}: local shape {array.shape[: len(prefix)]} does not match {prefix}"
        )
    if array.dtype.kind != "f" or array.dtype.itemsize != 8:
        raise DiagnosticError(f"{path}: expected float64 data, got {array.dtype}")
    if require_finite and not np.isfinite(array).all():
        raise DiagnosticError(f"{path}: contains NaN or infinity")
    return array


def _integer_triplet(meta: dict, key: str, source: Path) -> tuple[int, int, int]:
    value = meta.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(item) is not int for item in value)
    ):
        raise DiagnosticError(f"{source}: {key} must be an integer triplet")
    return tuple(value)


def _load_chunk(chunk_dir: Path) -> dict:
    meta_path = chunk_dir / "meta.json"
    if not meta_path.is_file():
        raise DiagnosticError(f"missing metadata: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot load {meta_path}: {error}") from error

    required_scalars = ("rank", "chunk_id", "step", "num_species")
    for key in required_scalars:
        if type(meta.get(key)) is not int:
            raise DiagnosticError(f"{meta_path}: {key} must be an integer")
    for key in ("time", "time_step"):
        if not isinstance(meta.get(key), (int, float)) or not np.isfinite(meta[key]):
            raise DiagnosticError(f"{meta_path}: {key} must be finite")

    offset = _integer_triplet(meta, "offset", meta_path)
    local_dims = _integer_triplet(meta, "local_dims", meta_path)
    global_dims = _integer_triplet(meta, "global_dims", meta_path)
    if any(value < 0 for value in offset) or any(
        value <= 0 for value in local_dims + global_dims
    ):
        raise DiagnosticError(f"{meta_path}: invalid grid dimensions or offset")
    if any(offset[i] + local_dims[i] > global_dims[i] for i in range(3)):
        raise DiagnosticError(f"{meta_path}: local domain is outside the global domain")

    num_species = meta["num_species"]
    masses = meta.get("particle_mass")
    charges = meta.get("particle_charge")
    if not isinstance(masses, list) or len(masses) != num_species:
        raise DiagnosticError(
            f"{meta_path}: particle_mass length does not match num_species"
        )
    if not isinstance(charges, list) or len(charges) != num_species:
        raise DiagnosticError(
            f"{meta_path}: particle_charge length does not match num_species"
        )
    if not np.isfinite(np.asarray(masses + charges, dtype=float)).all():
        raise DiagnosticError(f"{meta_path}: particle metadata must be finite")

    arrays = {
        "field": _load_array(chunk_dir / "field.npy", 4, local_dims),
        "fluid": _load_array(chunk_dir / "fluid.npy", 4, local_dims),
        "moment": _load_array(chunk_dir / "moment.npy", 5, local_dims),
    }
    particles = []
    for species in range(num_species):
        particle = _load_array(
            chunk_dir / f"particle_{species}.npy", 2, (), require_finite=False
        )
        if particle.shape[1:] != (7,):
            raise DiagnosticError(
                f"{chunk_dir}/particle_{species}.npy: expected shape (N, 7), got {particle.shape}"
            )
        if not np.isfinite(particle[:, :6]).all():
            raise DiagnosticError(
                f"{chunk_dir}/particle_{species}.npy: particle state contains NaN or infinity"
            )
        particles.append(particle)

    return {
        "meta": meta,
        "offset": offset,
        "local_dims": local_dims,
        "global_dims": global_dims,
        "arrays": arrays,
        "particles": particles,
    }


def load_snapshot(root: Path) -> dict:
    """Load, validate, and globally assemble one ``step_N`` snapshot."""
    if not root.is_dir():
        raise DiagnosticError(f"snapshot directory does not exist: {root}")
    chunk_dirs = sorted(path for path in root.glob("chunk_*") if path.is_dir())
    if not chunk_dirs:
        raise DiagnosticError(f"no chunk diagnostics found in {root}")
    chunks = [_load_chunk(path) for path in chunk_dirs]

    first = chunks[0]["meta"]
    global_dims = chunks[0]["global_dims"]
    common_keys = (
        "step",
        "time",
        "time_step",
        "num_species",
        "particle_mass",
        "particle_charge",
    )
    chunk_ids = set()
    for chunk in chunks:
        meta = chunk["meta"]
        if chunk["global_dims"] != global_dims:
            raise DiagnosticError(f"{root}: chunks disagree on global dimensions")
        for key in common_keys:
            if meta[key] != first[key]:
                raise DiagnosticError(f"{root}: chunks disagree on {key}")
        if meta["chunk_id"] in chunk_ids:
            raise DiagnosticError(f"{root}: duplicate chunk ID {meta['chunk_id']}")
        chunk_ids.add(meta["chunk_id"])

    assembled = {}
    coverage = np.zeros(global_dims, dtype=np.uint16)
    component_shapes = {}
    for key in ("field", "fluid", "moment"):
        component_shapes[key] = chunks[0]["arrays"][key].shape[3:]
        assembled[key] = np.empty(global_dims + component_shapes[key], dtype=np.float64)

    for chunk in chunks:
        offset = chunk["offset"]
        local = chunk["local_dims"]
        region = tuple(slice(offset[i], offset[i] + local[i]) for i in range(3))
        coverage[region] += 1
        for key, destination in assembled.items():
            source = chunk["arrays"][key]
            if source.shape[3:] != component_shapes[key]:
                raise DiagnosticError(
                    f"{root}: chunks disagree on {key} component shape"
                )
            destination[region] = source

    gaps = int(np.count_nonzero(coverage == 0))
    overlaps = int(np.count_nonzero(coverage > 1))
    if gaps or overlaps:
        raise DiagnosticError(
            f"{root}: invalid global coverage ({gaps} gaps, {overlaps} overlaps)"
        )

    particles = []
    for species in range(first["num_species"]):
        species_particles = np.concatenate(
            [chunk["particles"][species] for chunk in chunks]
        )
        ids = np.ascontiguousarray(species_particles[:, 6], dtype="<f8").view("<i8")
        if np.unique(ids).size != ids.size:
            raise DiagnosticError(
                f"{root}: duplicate particle IDs in species {species}"
            )
        order = np.argsort(ids)
        particles.append((ids[order], species_particles[order, :6]))

    return {
        "meta": first,
        "arrays": assembled,
        "particles": particles,
    }


def compare_snapshots(
    path_a: Path, path_b: Path, *, atol: float = 1.0e-12, rtol: float = 1.0e-12
) -> list[str]:
    """Return all numerical or identity mismatches between two snapshots."""
    snapshot_a = load_snapshot(path_a)
    snapshot_b = load_snapshot(path_b)
    errors = []

    for key in ("step", "time", "time_step", "num_species"):
        if snapshot_a["meta"][key] != snapshot_b["meta"][key]:
            errors.append(
                f"metadata {key} differs: {snapshot_a['meta'][key]} vs {snapshot_b['meta'][key]}"
            )
    for key in ("particle_mass", "particle_charge"):
        if not np.allclose(
            snapshot_a["meta"][key], snapshot_b["meta"][key], atol=atol, rtol=rtol
        ):
            errors.append(f"metadata {key} differs")

    for key in ("field", "fluid", "moment"):
        array_a = snapshot_a["arrays"][key]
        array_b = snapshot_b["arrays"][key]
        if array_a.shape != array_b.shape:
            errors.append(f"{key} shape differs: {array_a.shape} vs {array_b.shape}")
        elif not np.allclose(array_a, array_b, atol=atol, rtol=rtol):
            errors.append(
                f"{key} max absolute difference {np.max(np.abs(array_a - array_b)):.6e}"
            )

    if len(snapshot_a["particles"]) != len(snapshot_b["particles"]):
        errors.append("particle species count differs")
    else:
        for species, ((ids_a, state_a), (ids_b, state_b)) in enumerate(
            zip(snapshot_a["particles"], snapshot_b["particles"])
        ):
            if not np.array_equal(ids_a, ids_b):
                errors.append(f"species {species} particle IDs differ")
            elif not np.allclose(state_a, state_b, atol=atol, rtol=rtol):
                errors.append(
                    f"species {species} particle state max absolute difference "
                    f"{np.max(np.abs(state_a - state_b)):.6e}"
                )
    return errors


def compute_diagnostics(root: Path, config_path: Path | None = None) -> dict:
    """Compute energies and one-dimensional Fourier spectra for a snapshot."""
    snapshot = load_snapshot(root)
    field = snapshot["arrays"]["field"]
    fluid = snapshot["arrays"]["fluid"]
    moment = snapshot["arrays"]["moment"]

    gamma = 5.0 / 3.0
    if config_path is not None:
        if not config_path.is_file():
            raise DiagnosticError(f"configuration does not exist: {config_path}")
        match = re.search(
            r"^\s*gamma\s*=\s*([^#\s]+)", config_path.read_text(), re.MULTILINE
        )
        if match is None:
            raise DiagnosticError(
                f"configuration has no gamma parameter: {config_path}"
            )
        gamma = float(match.group(1))
    if gamma == 1.0:
        raise DiagnosticError("gamma must not equal one")

    magnetic = np.sum(field[..., 3:6] ** 2) / (8.0 * np.pi)
    electron = np.sum(
        0.5 * fluid[..., 0] * np.sum(fluid[..., 1:4] ** 2, axis=-1)
        + fluid[..., 4] / (gamma - 1.0)
    )
    ion = np.sum(
        0.5 * fluid[..., 5] * np.sum(fluid[..., 6:9] ** 2, axis=-1)
        + fluid[..., 9] / (gamma - 1.0)
    )
    kinetic = sum(
        0.5 * snapshot["meta"]["particle_mass"][species] * np.sum(state[:, 3:6] ** 2)
        for species, (_, state) in enumerate(snapshot["particles"])
    )

    density_x = np.mean(np.sum(moment[..., 0], axis=-1), axis=(0, 1))
    mode_density = np.abs(np.fft.rfft(density_x)) / density_x.size
    transverse_b = np.mean(field[..., 4] + 1j * field[..., 5], axis=(0, 1))
    mode_transverse_b = (
        np.abs(np.fft.rfft(transverse_b.real) + 1j * np.fft.rfft(transverse_b.imag))
        / transverse_b.size
    )

    return {
        "magnetic": float(magnetic),
        "electron": float(electron),
        "ion": float(ion),
        "kinetic": float(kinetic),
        "total": float(magnetic + electron + ion + kinetic),
        "mode_density": mode_density.tolist(),
        "mode_transverse_b": mode_transverse_b.tolist(),
    }


def validate_history(root: Path, final_step: int) -> None:
    """Validate that a run retained semantic snapshots and per-step SSOR logs."""
    if final_step < 0:
        raise DiagnosticError("final step must be nonnegative")
    for step in range(final_step + 1):
        snapshot_path = root / "snapshots" / f"step_{step}"
        snapshot = load_snapshot(snapshot_path)
        if snapshot["meta"]["step"] != step:
            raise DiagnosticError(
                f"snapshot step_{step} contains step {snapshot['meta']['step']}"
            )
        diagnostics = compute_diagnostics(snapshot_path)
        energy = np.asarray(
            [
                diagnostics["magnetic"],
                diagnostics["electron"],
                diagnostics["ion"],
                diagnostics["kinetic"],
                diagnostics["total"],
            ]
        )
        if not np.isfinite(energy).all():
            raise DiagnosticError(f"snapshot step_{step} has non-finite energy")
        for key in ("mode_density", "mode_transverse_b"):
            mode = np.asarray(diagnostics[key])
            if mode.size < 2 or not np.isfinite(mode).all():
                raise DiagnosticError(f"snapshot step_{step} has invalid {key}")
        if step == final_step and diagnostics["mode_density"][1] <= 0:
            raise DiagnosticError(f"snapshot step_{step} lost the first density mode")
    for step in range(1, final_step + 1):
        log_path = root / "ssor" / f"step_{step}.log"
        _, _, offsets, converged = _parse_ssor_log(log_path)
        if offsets.size != 4:
            raise DiagnosticError(f"{log_path}: expected three Ohm stages")
        if not all(converged):
            raise DiagnosticError(f"{log_path}: not all Ohm stages converged")


def validate_invariants(root: Path, expected_particles: int | None = None) -> None:
    """Validate accepted-state invariants visible in one diagnostic snapshot."""
    snapshot = load_snapshot(root)
    for key, array in snapshot["arrays"].items():
        if not np.isfinite(array).all():
            raise DiagnosticError(f"{key} contains NaN or infinity")
    for species, (ids, state) in enumerate(snapshot["particles"]):
        if expected_particles is not None and ids.size != expected_particles:
            raise DiagnosticError(
                f"species {species} has {ids.size} particles, expected {expected_particles}"
            )
        if ids.size and not np.all(ids[:-1] <= ids[1:]):
            raise DiagnosticError(
                f"species {species} particle IDs are not canonicalized"
            )
        if not np.isfinite(state).all():
            raise DiagnosticError(
                f"species {species} particle state contains NaN or infinity"
            )


def _parse_ssor_log(
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    if not log_path.is_file():
        raise DiagnosticError(f"missing SSOR log: {log_path}")
    iterations = []
    residuals = []
    offsets = [0]
    converged = []
    stage_pattern = re.compile(
        r"^# Ohm stage (\d+): iterations=(\d+) converged=(true|false)$"
    )
    iter_pattern = re.compile(r"^iter=(\d+), error=([^\s]+)$")
    stage_start = 0
    for line in log_path.read_text().splitlines():
        if match := iter_pattern.match(line):
            iterations.append(int(match.group(1)))
            residuals.append(float(match.group(2)))
            continue
        if match := stage_pattern.match(line):
            expected_stage_iterations = int(match.group(2))
            actual_stage_iterations = len(iterations) - stage_start
            if actual_stage_iterations != expected_stage_iterations:
                raise DiagnosticError(
                    f"{log_path}: Ohm stage {match.group(1)} reports "
                    f"{expected_stage_iterations} iterations but has {actual_stage_iterations}"
                )
            offsets.append(len(iterations))
            converged.append(match.group(3) == "true")
            stage_start = len(iterations)
            continue
        if line.strip():
            raise DiagnosticError(f"{log_path}: unrecognized SSOR log line: {line}")
    return (
        np.asarray(iterations, dtype=np.int32),
        np.asarray(residuals, dtype=np.float64),
        np.asarray(offsets, dtype=np.int32),
        converged,
    )


def compare_ssor_history(log_path: Path, reference_path: Path) -> None:
    """Compare one port SSOR log with the legacy rounded residual history."""
    try:
        reference = np.load(reference_path, allow_pickle=False)
    except Exception as error:
        raise DiagnosticError(f"cannot load {reference_path}: {error}") from error
    required = ("ssor_iteration", "ssor_relative_error", "ssor_offset")
    for key in required:
        if key not in reference:
            raise DiagnosticError(f"legacy reference is missing {key}")
    iterations, residuals, offsets, converged = _parse_ssor_log(log_path)
    if not converged or not all(converged):
        raise DiagnosticError(f"{log_path}: not all Ohm stages converged")
    if not np.array_equal(offsets, reference["ssor_offset"]):
        raise DiagnosticError(f"{log_path}: SSOR offsets differ")
    if not np.array_equal(iterations, reference["ssor_iteration"]):
        raise DiagnosticError(f"{log_path}: SSOR iteration sequence differs")
    actual = np.asarray([f"{value:12.3e}" for value in residuals])
    expected = np.asarray(
        [f"{value:12.3e}" for value in reference["ssor_relative_error"]]
    )
    if not np.array_equal(actual, expected):
        raise DiagnosticError(f"{log_path}: SSOR rounded residual history differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare", aliases=["decomp"], help="compare two globally assembled snapshots"
    )
    compare_parser.add_argument("snapshot_a", type=Path)
    compare_parser.add_argument("snapshot_b", type=Path)
    compare_parser.add_argument("--atol", type=float, default=1.0e-12)
    compare_parser.add_argument("--rtol", type=float, default=1.0e-12)

    diag_parser = subparsers.add_parser(
        "diag", help="print energy and mode diagnostics"
    )
    diag_parser.add_argument("snapshot", type=Path)
    diag_parser.add_argument("--config", type=Path)
    history_parser = subparsers.add_parser(
        "history", help="validate snapshot and SSOR history"
    )
    history_parser.add_argument("diagnostics", type=Path)
    history_parser.add_argument("final_step", type=int)

    invariants_parser = subparsers.add_parser(
        "invariants", help="validate accepted snapshot invariants"
    )
    invariants_parser.add_argument("snapshot", type=Path)
    invariants_parser.add_argument("--expected-particles", type=int)

    ssor_parser = subparsers.add_parser(
        "ssor", help="compare SSOR history with a legacy fixture"
    )
    ssor_parser.add_argument("log", type=Path)
    ssor_parser.add_argument("reference", type=Path)
    args = parser.parse_args()

    try:
        if args.command in ("compare", "decomp"):
            errors = compare_snapshots(
                args.snapshot_a, args.snapshot_b, atol=args.atol, rtol=args.rtol
            )
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print("snapshots match")
            return 0

        if args.command == "history":
            validate_history(args.diagnostics, args.final_step)
            print("diagnostic history is complete")
            return 0

        if args.command == "invariants":
            validate_invariants(args.snapshot, args.expected_particles)
            print("accepted-state invariants hold")
            return 0

        if args.command == "ssor":
            compare_ssor_history(args.log, args.reference)
            print("SSOR history matches legacy reference")
            return 0

        print(json.dumps(compute_diagnostics(args.snapshot, args.config), indent=2))
        return 0
    except (DiagnosticError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
