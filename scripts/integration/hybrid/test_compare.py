#!/usr/bin/env python3
"""Tests for strict Hybrid snapshot validation and comparison."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare import (
    DiagnosticError,
    compare_snapshots,
    compute_diagnostics,
    load_snapshot,
)


def _particle_rows(ids: list[int], states: np.ndarray) -> np.ndarray:
    particles = np.zeros((len(ids), 7), dtype="<f8")
    particles[:, :6] = states
    particles[:, 6] = np.asarray(ids, dtype="<i8").view("<f8")
    return particles


def _write_snapshot(
    root: Path,
    chunks: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    *,
    global_dims: tuple[int, int, int] = (1, 1, 4),
    bad: str | None = None,
) -> Path:
    snapshot = root / "step_3"
    field = np.zeros(global_dims + (6,))
    field[..., 3] = np.sqrt(8.0 * np.pi)
    field[0, 0, :, 4] = [1.0, 0.0, -1.0, 0.0]
    field[0, 0, :, 5] = [0.0, 1.0, 0.0, -1.0]
    fluid = np.zeros(global_dims + (10,))
    fluid[..., 0] = 2.0
    fluid[..., 1] = 3.0
    fluid[..., 4] = 4.0
    fluid[..., 5] = 5.0
    fluid[..., 7] = 2.0
    fluid[..., 9] = 2.0
    moment = np.zeros(global_dims + (1, 10))
    moment[0, 0, :, 0, 0] = [1.0, 0.0, -1.0, 0.0]
    states = np.asarray(
        [[0.25, 0.5, 0.5, 1.0, 2.0, 2.0], [2.25, 0.5, 0.5, -1.0, 0.0, 0.0]]
    )

    for chunk_id, (offset, local_dims) in enumerate(chunks):
        chunk_dir = snapshot / f"chunk_{chunk_id}"
        chunk_dir.mkdir(parents=True)
        local_slice = tuple(
            slice(offset[axis], offset[axis] + local_dims[axis]) for axis in range(3)
        )
        np.save(chunk_dir / "field.npy", field[local_slice])
        np.save(chunk_dir / "fluid.npy", fluid[local_slice])
        np.save(chunk_dir / "moment.npy", moment[local_slice])
        owned = np.ones(states.shape[0], dtype=bool)
        for state_axis, grid_axis in enumerate((2, 1, 0)):
            owned &= states[:, state_axis] >= offset[grid_axis]
            owned &= states[:, state_axis] < offset[grid_axis] + local_dims[grid_axis]
        ids = np.arange(states.shape[0], dtype=np.int64)[owned].tolist()
        np.save(chunk_dir / "particle_0.npy", _particle_rows(ids, states[owned]))
        meta = {
            "rank": chunk_id,
            "chunk_id": chunk_id,
            "offset": list(offset),
            "local_dims": list(local_dims),
            "global_dims": list(global_dims),
            "step": 3,
            "time": 0.03,
            "time_step": 0.01,
            "num_species": 1,
            "particle_mass": [2.0],
            "particle_charge": [1.0],
        }
        (chunk_dir / "meta.json").write_text(json.dumps(meta))

    if bad == "overlap":
        meta_path = snapshot / "chunk_1" / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["offset"] = [0, 0, 1]
        meta_path.write_text(json.dumps(meta))
    elif bad == "nan":
        path = snapshot / "chunk_0" / "field.npy"
        array = np.load(path)
        array.flat[0] = np.nan
        np.save(path, array)
    elif bad == "duplicate_id":
        path = snapshot / "chunk_1" / "particle_0.npy"
        array = np.load(path)
        array[0, 6] = np.asarray([0], dtype="<i8").view("<f8")[0]
        np.save(path, array)
    return snapshot


class SnapshotTest(unittest.TestCase):
    def test_reconstructs_different_decompositions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = _write_snapshot(root / "single", [((0, 0, 0), (1, 1, 4))])
            split = _write_snapshot(
                root / "split", [((0, 0, 0), (1, 1, 2)), ((0, 0, 2), (1, 1, 2))]
            )
            self.assertEqual(compare_snapshots(single, split), [])

    def test_reconstructs_cx_cy_cz_decomposition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_dims = (2, 2, 4)
            single = _write_snapshot(
                root / "single", [((0, 0, 0), global_dims)], global_dims=global_dims
            )
            regions = [
                ((z, y, x), (1, 1, 2)) for z in (0, 1) for y in (0, 1) for x in (0, 2)
            ]
            split = _write_snapshot(root / "split", regions, global_dims=global_dims)
            self.assertEqual(compare_snapshots(single, split), [])

    def test_detects_numerical_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_snapshot(root / "first", [((0, 0, 0), (1, 1, 4))])
            second = _write_snapshot(root / "second", [((0, 0, 0), (1, 1, 4))])
            path = second / "chunk_0" / "fluid.npy"
            array = np.load(path)
            array.flat[0] += 1.0
            np.save(path, array)
            self.assertTrue(
                any("fluid" in error for error in compare_snapshots(first, second))
            )
            array.flat[0] -= 1.0
            np.save(path, array)
            path = second / "chunk_0" / "particle_0.npy"
            particles = np.load(path)
            particles[0, 0] += 1.0
            np.save(path, particles)
            self.assertTrue(
                any(
                    "particle state" in error
                    for error in compare_snapshots(first, second)
                )
            )

    def test_rejects_missing_empty_and_malformed_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(DiagnosticError):
                load_snapshot(root / "missing")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(DiagnosticError):
                load_snapshot(empty)
            malformed = _write_snapshot(root / "malformed", [((0, 0, 0), (1, 1, 4))])
            (malformed / "chunk_0" / "meta.json").write_text("not JSON")
            with self.assertRaises(DiagnosticError):
                load_snapshot(malformed)

    def test_rejects_overlap_nan_and_duplicate_particle_id(self):
        for defect in ("overlap", "nan", "duplicate_id"):
            with (
                self.subTest(defect=defect),
                tempfile.TemporaryDirectory() as directory,
            ):
                snapshot = _write_snapshot(
                    Path(directory),
                    [((0, 0, 0), (1, 1, 2)), ((0, 0, 2), (1, 1, 2))],
                    bad=defect,
                )
                with self.assertRaises(DiagnosticError):
                    load_snapshot(snapshot)

    def test_analytic_energy_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _write_snapshot(Path(directory), [((0, 0, 0), (1, 1, 4))])
            diagnostic = compute_diagnostics(snapshot)
            self.assertAlmostEqual(diagnostic["magnetic"], 4.0 + 1.0 / (2.0 * np.pi))
            self.assertAlmostEqual(diagnostic["electron"], 60.0)
            self.assertAlmostEqual(diagnostic["ion"], 52.0)
            self.assertAlmostEqual(diagnostic["kinetic"], 10.0)
            np.testing.assert_allclose(diagnostic["mode_density"], [0.0, 0.5, 0.0])
            np.testing.assert_allclose(diagnostic["mode_transverse_b"], [0.0, 1.0, 0.0])

    def test_legacy_cli_rejects_missing_input(self):
        script = Path(__file__).with_name("legacy_compare.py")
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "missing-snapshot",
                "missing-reference.npz",
                "--config",
                "missing-config.toml",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
