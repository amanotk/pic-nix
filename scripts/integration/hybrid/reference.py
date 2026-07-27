#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import h5py
import numpy as np


REFERENCE_DIR = Path(__file__).with_name("reference")
CASES = {
    "beam_one_step": REFERENCE_DIR / "beam_one_step.cfg",
    "beam_short_run": REFERENCE_DIR / "beam_short_run.cfg",
}
SSOR_PATTERN = re.compile(r"iter\s*=\s*(\d+), error\s*=\s*([-+\d.eE]+)")
SHARED_MAGIC = b"HYBRIDR6"
SHARED_VERSION = 1


def parse_config(path: Path) -> dict[str, str]:
    config = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def load_hdf5(run_dir: Path) -> dict[str, np.ndarray]:
    arrays = {}
    with h5py.File(run_dir / "field.h5", "r") as field:
        for name in ("time", "delt", "xig", "yig", "zig", "eb", "up"):
            arrays[f"field_{name}"] = np.asarray(field[name])

    with h5py.File(run_dir / "moment.h5", "r") as moment:
        for name in ("time", "delt", "mom"):
            arrays[f"moment_{name}"] = np.asarray(moment[name])

    with h5py.File(run_dir / "particle.h5", "r") as particle:
        arrays["particle_time"] = np.asarray(particle["time"])
        arrays["particle_delt"] = np.asarray(particle["delt"])
        counts = np.asarray(particle["np"])
        arrays["particle_count"] = counts

        for species in range(counts.shape[1]):
            species_counts = counts[:, species]
            if np.any(species_counts != species_counts[0]):
                raise RuntimeError(
                    f"particle count changed for species {species}: {species_counts}"
                )

            count = int(species_counts[0])
            records = np.asarray(particle[f"particle{species:02d}"])[:, :count, :]
            states = np.empty(records[..., :6].shape, dtype=np.float64)
            ids = np.empty(records.shape[:2], dtype=np.int64)

            for step in range(records.shape[0]):
                step_ids = np.ascontiguousarray(records[step, :, 6]).view(np.int64)
                order = np.argsort(step_ids, kind="stable")
                ids[step] = step_ids[order]
                states[step] = records[step, order, :6]

            arrays[f"particle_{species:02d}_id"] = ids
            arrays[f"particle_{species:02d}_state"] = states

    return arrays


def add_diagnostics(
    arrays: dict[str, np.ndarray], config: dict[str, str], debug_path: Path
) -> None:
    field = arrays["field_eb"]
    fluid = arrays["field_up"]
    moment = arrays["moment_mom"]

    magnetic = np.sum(field[..., 3:6] ** 2, axis=(1, 2, 3, 4)) / (8 * np.pi)
    electron = np.sum(
        0.5 * fluid[..., 0] * np.sum(fluid[..., 1:4] ** 2, axis=-1)
        + fluid[..., 4] / (float(config["gamma"]) - 1),
        axis=(1, 2, 3),
    )
    ion = np.sum(
        0.5 * fluid[..., 5] * np.sum(fluid[..., 6:9] ** 2, axis=-1)
        + fluid[..., 9] / (float(config["gamma"]) - 1),
        axis=(1, 2, 3),
    )

    npc = int(config["Npc"])
    beam_density = float(config["nb"])
    particle_mass = ((1 - beam_density) / npc, beam_density / npc)
    kinetic = np.zeros_like(magnetic)
    for species, mass in enumerate(particle_mass):
        velocity = arrays[f"particle_{species:02d}_state"][..., 3:6]
        kinetic += 0.5 * mass * np.sum(velocity**2, axis=(1, 2))

    arrays["energy"] = np.stack((magnetic, electron, ion, kinetic), axis=-1)

    density_x = np.mean(np.sum(moment[..., :, 0], axis=-1), axis=(1, 2))
    arrays["mode_density"] = np.fft.rfft(density_x, axis=-1) / density_x.shape[-1]

    transverse_b = np.mean(field[..., 4] + 1j * field[..., 5], axis=(1, 2))
    arrays["mode_transverse_b"] = (
        np.fft.rfft(transverse_b.real, axis=-1)
        + 1j * np.fft.rfft(transverse_b.imag, axis=-1)
    ) / transverse_b.shape[-1]

    iterations = []
    errors = []
    offsets = [0]
    previous_iteration = None
    for match in SSOR_PATTERN.finditer(debug_path.read_text(encoding="utf-8")):
        iteration = int(match.group(1))
        if previous_iteration is not None and iteration <= previous_iteration:
            offsets.append(len(iterations))
        iterations.append(iteration)
        errors.append(float(match.group(2)))
        previous_iteration = iteration
    offsets.append(len(iterations))

    arrays["ssor_iteration"] = np.asarray(iterations, dtype=np.int32)
    arrays["ssor_relative_error"] = np.asarray(errors, dtype=np.float64)
    arrays["ssor_offset"] = np.asarray(offsets, dtype=np.int32)


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def write_canonical(
    run_dir: Path, config_path: Path, output_path: Path
) -> dict[str, dict[str, object]]:
    config = parse_config(config_path)
    arrays = load_hdf5(run_dir)
    add_diagnostics(arrays, config, run_dir / "debug.log")
    np.savez_compressed(output_path, **arrays)

    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_digest(value),
        }
        for name, value in sorted(arrays.items())
    }


def compare_canonical(first: Path, second: Path) -> None:
    with np.load(first) as lhs, np.load(second) as rhs:
        if lhs.files != rhs.files:
            raise RuntimeError(f"canonical keys differ: {lhs.files} != {rhs.files}")
        for name in lhs.files:
            if not np.array_equal(lhs[name], rhs[name], equal_nan=True):
                difference = np.max(np.abs(lhs[name] - rhs[name]))
                raise RuntimeError(f"{name} is not reproducible; max diff={difference}")


def verify_fixture(case: str) -> None:
    fixture_path = REFERENCE_DIR / f"{case}.npz"
    manifest_path = REFERENCE_DIR / f"{case}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with np.load(fixture_path) as fixture:
        if set(fixture.files) != set(manifest):
            raise RuntimeError(
                f"{case} fixture keys differ from its manifest: {fixture.files}"
            )
        for name, expected in manifest.items():
            value = fixture[name]
            actual = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_digest(value),
            }
            if actual != expected:
                raise RuntimeError(
                    f"{case}:{name} differs from its manifest: {actual} != {expected}"
                )


def export_shared_state(case: str, output_path: Path, index: int = 0) -> None:
    """Export one canonical fixture state for the test-only Hybrid loader."""
    verify_fixture(case)
    fixture_path = REFERENCE_DIR / f"{case}.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        field = np.asarray(fixture["field_eb"][index], dtype="<f8")
        fluid = np.asarray(fixture["field_up"][index], dtype="<f8")
        moment = np.asarray(fixture["moment_mom"][index], dtype="<f8")
        if field.shape[-1] != 6 or fluid.shape[-1] != 10:
            raise RuntimeError("shared fixture has incompatible field or fluid shape")
        if moment.shape[:3] != field.shape[:3] or moment.shape[-1] != 10:
            raise RuntimeError("shared fixture has incompatible moment shape")
        nz, ny, nx = field.shape[:3]
        num_species = moment.shape[3]
        time = float(fixture["field_time"][index])
        time_step = float(fixture["field_delt"][index])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            output.write(
                struct.pack(
                    "<8sII4i2d",
                    SHARED_MAGIC,
                    SHARED_VERSION,
                    index,
                    nz,
                    ny,
                    nx,
                    num_species,
                    time,
                    time_step,
                )
            )
            output.write(field.tobytes(order="C"))
            output.write(fluid.tobytes(order="C"))
            output.write(moment.tobytes(order="C"))
            for species in range(num_species):
                states = np.asarray(
                    fixture[f"particle_{species:02d}_state"][index], dtype="<f8"
                )
                ids = np.asarray(
                    fixture[f"particle_{species:02d}_id"][index], dtype="<i8"
                )
                if states.shape != (ids.size, 6):
                    raise RuntimeError(
                        f"shared fixture species {species} particle shape mismatch"
                    )
                output.write(struct.pack("<q", ids.size))
                for state, particle_id in zip(states, ids):
                    output.write(state.tobytes(order="C"))
                    output.write(struct.pack("<q", int(particle_id)))


def run_case(
    executable: Path, output_root: Path, case: str, repeats: int
) -> dict[str, object]:
    config_path = CASES[case].resolve()
    case_root = output_root / case
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True)

    manifests = []
    canonical_paths = []
    for repeat in range(repeats):
        run_dir = case_root / f"run_{repeat + 1}"
        run_dir.mkdir()
        subprocess.run(
            [
                "mpiexec",
                "-n",
                "1",
                str(executable.resolve()),
                "-c",
                str(config_path),
                "-x",
                "1",
                "-y",
                "1",
                "-z",
                "1",
            ],
            cwd=run_dir,
            check=True,
        )
        canonical_path = run_dir / "canonical.npz"
        manifests.append(write_canonical(run_dir, config_path, canonical_path))
        canonical_paths.append(canonical_path)

    for candidate in canonical_paths[1:]:
        compare_canonical(canonical_paths[0], candidate)

    fixture_path = REFERENCE_DIR / f"{case}.npz"
    shutil.copyfile(canonical_paths[0], fixture_path)
    manifest_path = REFERENCE_DIR / f"{case}.json"
    manifest_path.write_text(
        json.dumps(manifests[0], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_fixture(case)

    return {
        "case": case,
        "config": str(config_path),
        "fixture": str(fixture_path),
        "manifest": str(manifest_path),
        "repeats": repeats,
        "reproducible": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reproducible Hybrid3D beam reference fixtures."
    )
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--export-shared", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cases = args.case or list(CASES)
    if args.verify_only:
        for case in cases:
            verify_fixture(case)
        print(json.dumps({"verified": cases}, indent=2))
        return 0

    if args.export_shared is not None:
        if len(cases) != 1:
            parser.error("--export-shared requires exactly one --case")
        export_shared_state(cases[0], args.export_shared, args.index)
        print(json.dumps({"exported": str(args.export_shared), "index": args.index}))
        return 0

    if args.executable is None or args.output is None:
        parser.error(
            "--executable and --output are required unless --verify-only is used"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = [
        run_case(args.executable, args.output, case, args.repeats) for case in cases
    ]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
