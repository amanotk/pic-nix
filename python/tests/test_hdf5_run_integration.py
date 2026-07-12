#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import h5py
import msgpack
import numpy as np
import pytest

from picnix import Run, hdf5_converter


def write_json(path, obj):
    with path.open("w") as fp:
        json.dump(obj, fp)


def write_particle_step(path, stem, values, ids):
    path.mkdir(parents=True, exist_ok=True)
    records = np.zeros((values.shape[0], values.shape[1] + 1), dtype="<f8")
    records[:, :-1] = values
    records[:, -1] = ids.astype("<u8").view("<f8")
    raw_path = path / f"{stem}.data"
    raw_path.write_bytes(records.tobytes())
    write_json(
        path / f"{stem}.json",
        {
            "meta": {
                "endian": 1,
                "rawfile": raw_path.name,
                "layout": 1,
                "time": 1.25,
                "step": int(stem),
            },
            "dataset": {
                "up00": {
                    "datatype": "f8",
                    "description": "particle species 00",
                    "offset": 0,
                    "size": records.nbytes,
                    "ndim": records.ndim,
                    "shape": list(records.shape),
                }
            },
        },
    )


def make_run_fixture(tmp_path):
    data_dir = tmp_path / "data"
    values0 = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype="<f8")
    values1 = np.array(
        [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]], dtype="<f8"
    )
    write_particle_step(
        data_dir / "node000000" / "particle", "00000000", values0, np.array([100, 101])
    )
    write_particle_step(
        data_dir / "node000001" / "particle", "00000000", values1, np.array([200, 201])
    )

    config = {
        "application": {"basedir": "data", "iomode": "posix"},
        "parameter": {
            "Ns": 1,
            "Nx": 1,
            "Ny": 1,
            "Nz": 1,
            "Cx": 1,
            "Cy": 1,
            "Cz": 1,
            "delt": 1.0,
            "delh": 1.0,
        },
        "diagnostic": [{"name": "particle"}],
    }
    profile = data_dir / "profile.msgpack"
    with profile.open("wb") as fp:
        msgpack.pack(
            {
                "timestamp": {},
                "nprocess": 2,
                "chunkmap": {"chunkid": [[[0]]], "coord": [[0, 0, 0]]},
                "qm": None,
                "configuration": config,
            },
            fp,
        )
    return profile, data_dir


def convert_particle(monkeypatch, data_dir):
    monkeypatch.setattr(
        "sys.argv",
        [
            "picnix-hdf5-convert",
            "--input-dir",
            str(data_dir),
            "--prefix",
            "particle",
            "--overwrite",
        ],
    )
    hdf5_converter.main()


def test_run_reads_raw_particle_values_and_ids(tmp_path):
    profile, _ = make_run_fixture(tmp_path)

    run = Run(str(profile))

    assert run.get_diag_handler("particle").storage.kind == "json"
    assert run.get_step("particle").tolist() == [0]
    assert run.get_time_at("particle", 0) == 1.25
    np.testing.assert_allclose(
        run.read_at("particle", 0)["up00"],
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
            ],
            dtype="<f8",
        ),
    )
    assert run.read_particle_id_at("particle", 0)["up00"].tolist() == [
        100,
        101,
        200,
        201,
    ]


def test_run_prefers_hdf5_particle_values_and_ids(tmp_path, monkeypatch):
    profile, data_dir = make_run_fixture(tmp_path)
    convert_particle(monkeypatch, data_dir)

    run = Run(str(profile))

    assert run.get_diag_handler("particle").storage.kind == "hdf5-vds"
    np.testing.assert_allclose(
        run.read_at("particle", 0)["up00"],
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
            ],
            dtype=np.float32,
        ),
    )
    assert run.read_particle_id_at("particle", 0)["up00"].tolist() == [
        100,
        101,
        200,
        201,
    ]


def test_run_reads_raw_particle_ranges_across_nodes(tmp_path):
    profile, _ = make_run_fixture(tmp_path)

    run = Run(str(profile))

    np.testing.assert_allclose(
        run.read_particle_at("particle", 0, start=1, stop=3)["up00"],
        np.array([[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]], dtype="<f8"),
    )
    assert run.read_particle_id_at("particle", 0, start=1, stop=3)["up00"].tolist() == [
        101,
        200,
    ]
    assert run.read_particle_id_at("particle", 0, stop=2)["up00"].tolist() == [
        100,
        101,
    ]
    assert run.read_particle_id_at("particle", 0, start=-2)["up00"].tolist() == [
        200,
        201,
    ]


def test_run_reads_hdf5_particle_ranges_across_nodes(tmp_path, monkeypatch):
    profile, data_dir = make_run_fixture(tmp_path)
    convert_particle(monkeypatch, data_dir)

    run = Run(str(profile))

    np.testing.assert_allclose(
        run.read_particle_at("particle", 0, start=1, stop=3)["up00"],
        np.array(
            [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
            dtype=np.float32,
        ),
    )
    assert run.read_particle_id_at("particle", 0, start=1, stop=3)["up00"].tolist() == [
        101,
        200,
    ]
    assert run.read_particle_id_at("particle", 0, stop=2)["up00"].tolist() == [
        100,
        101,
    ]
    assert run.read_particle_id_at("particle", 0, start=-2)["up00"].tolist() == [
        200,
        201,
    ]


def test_run_rejects_invalid_existing_hdf5_vds(tmp_path):
    profile, data_dir = make_run_fixture(tmp_path)
    hdf5_dir = data_dir / "hdf5"
    hdf5_dir.mkdir()
    with h5py.File(hdf5_dir / "particle.vds.h5", "w") as h5fp:
        h5fp.attrs["picnix_hdf5_layout"] = "not-picnix"

    with pytest.raises(ValueError, match="invalid PIC-NIX HDF5 VDS layout"):
        Run(str(profile))
