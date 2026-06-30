#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import h5py
import numpy as np

from picnix import hdf5_converter


def write_json(path, obj):
    with path.open("w") as fp:
        json.dump(obj, fp)


def write_step(path, prefix, stem, dataset_name, array):
    path.mkdir(parents=True, exist_ok=True)
    raw_path = path / f"{stem}.data"
    raw_path.write_bytes(array.tobytes())
    write_json(
        path / f"{stem}.json",
        {
            "meta": {
                "endian": 1,
                "rawfile": raw_path.name,
                "layout": 1,
                "time": 0.0,
                "step": int(stem),
            },
            "dataset": {
                dataset_name: {
                    "datatype": "f8",
                    "description": prefix,
                    "offset": 0,
                    "size": array.nbytes,
                    "ndim": array.ndim,
                    "shape": list(array.shape),
                }
            },
        },
    )


def make_posix_fixture(tmp_path):
    input_dir = tmp_path / "data"
    for node_index in range(2):
        node_dir = input_dir / f"node{node_index:06d}"

        field = (np.arange(4, dtype="<f8") + node_index * 10).reshape(2, 2)
        write_step(node_dir / "field", "field", "00000000", "uf", field)

        particle = np.array(
            [
                [1.0 + node_index, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
                [7.0 + node_index, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0],
            ],
            dtype="<f8",
        )
        ids = np.array([100 + node_index * 10, 101 + node_index * 10], dtype="<u8")
        particle[:, -1] = ids.view("<f8")
        write_step(node_dir / "particle", "particle", "00000000", "up00", particle)
    (input_dir / "node000000" / "history.txt").write_text("history\n")
    return input_dir


def run_converter(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["picnix-hdf5-convert", *map(str, args)])
    hdf5_converter.main()


def test_auto_prefix_conversion_writes_manifest_and_vds(tmp_path, monkeypatch):
    input_dir = make_posix_fixture(tmp_path)

    run_converter(monkeypatch, "--input-dir", input_dir, "--overwrite")

    output_dir = input_dir / "hdf5"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert [item["prefix"] for item in manifest["prefixes"]] == [
        "field",
        "particle",
    ]
    assert manifest["defaults"]["field_dtype"] == "float32"
    assert manifest["defaults"]["compression"] == "none"
    assert manifest["verification"]["status"] == "passed"
    assert manifest["verification"]["level"] == "fast"
    assert manifest["verification"]["raw_fingerprint"]["hdf5_files"] == 2

    with h5py.File(output_dir / "field.vds.h5", "r") as h5fp:
        field = h5fp["field/00000000/uf"][...]
    np.testing.assert_allclose(
        field,
        np.array([[0, 1], [2, 3], [10, 11], [12, 13]], dtype=np.float32),
    )
    assert field.dtype == np.dtype("float32")

    with h5py.File(output_dir / "particle.vds.h5", "r") as h5fp:
        values = h5fp["particle/00000000/up00"][...]
        ids = h5fp["particle/00000000/up00_id"][...]
    assert values.shape == (4, 6)
    assert values.dtype == np.dtype("float32")
    assert ids.dtype == np.dtype("uint64")
    assert ids.tolist() == [100, 101, 110, 111]


def test_selected_prefix_and_resume(tmp_path, monkeypatch):
    input_dir = make_posix_fixture(tmp_path)
    output_dir = tmp_path / "selected"

    run_converter(
        monkeypatch,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--prefix",
        "field",
        "--overwrite",
    )
    run_converter(
        monkeypatch,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--prefix",
        "field",
        "--resume",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert [item["prefix"] for item in manifest["prefixes"]] == ["field"]
    assert (output_dir / "field" / "00000000.h5").exists()
    assert (output_dir / "field.vds.h5").exists()
    assert not (output_dir / "particle.vds.h5").exists()


def test_standalone_verify_and_remove_original(tmp_path, monkeypatch):
    input_dir = make_posix_fixture(tmp_path)

    run_converter(
        monkeypatch,
        "--input-dir",
        input_dir,
        "--overwrite",
        "--no-verify",
    )
    manifest_path = input_dir / "hdf5" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "verification" not in manifest

    run_converter(monkeypatch, "verify", "--input-dir", input_dir, "--prefix", "field")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["verification"]["status"] == "passed"

    run_converter(
        monkeypatch,
        "remove-original",
        "--input-dir",
        input_dir,
        "--prefix",
        "field",
        "--dry-run",
    )
    assert (input_dir / "node000000" / "field" / "00000000.json").exists()

    run_converter(
        monkeypatch,
        "remove-original",
        "--input-dir",
        input_dir,
        "--prefix",
        "field",
        "--yes",
    )
    assert not (input_dir / "node000000" / "field").exists()
    assert not (input_dir / "node000001" / "field").exists()
    assert (input_dir / "history.txt").read_text() == "history\n"
    assert not (input_dir / "node000000" / "history.txt").exists()
    assert (input_dir / "node000000" / "particle").exists()
    assert (input_dir / "node000001" / "particle").exists()

    run_converter(
        monkeypatch, "verify", "--input-dir", input_dir, "--prefix", "particle"
    )
    run_converter(
        monkeypatch,
        "remove-original",
        "--input-dir",
        input_dir,
        "--prefix",
        "particle",
        "--yes",
    )
    assert not (input_dir / "node000000").exists()
    assert not (input_dir / "node000001").exists()


def test_remove_original_preserves_unreferenced_data(tmp_path, monkeypatch):
    input_dir = make_posix_fixture(tmp_path)
    stale_path = input_dir / "node000000" / "field" / "stale.data"
    stale_path.write_bytes(b"not referenced by json metadata")

    run_converter(
        monkeypatch,
        "--input-dir",
        input_dir,
        "--prefix",
        "field",
        "--overwrite",
    )
    run_converter(
        monkeypatch,
        "remove-original",
        "--input-dir",
        input_dir,
        "--prefix",
        "field",
        "--yes",
    )

    assert not (input_dir / "node000000" / "field" / "00000000.json").exists()
    assert not (input_dir / "node000000" / "field" / "00000000.data").exists()
    assert stale_path.read_bytes() == b"not referenced by json metadata"
