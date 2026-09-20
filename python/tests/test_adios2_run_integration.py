#!/usr/bin/env python
# -*- coding: utf-8 -*-

import msgpack
import numpy as np
import pytest

adios2 = pytest.importorskip("adios2")

from picnix import Run  # noqa: E402


def write_diagnostic(path, diagnostic, prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with adios2.Stream(str(path), "w") as stream:
        stream.write_attribute("picnix_schema", "diagnostic-bp-v1")
        stream.write_attribute("diagnostic", diagnostic)
        stream.write_attribute("prefix", prefix)
        for step in range(2):
            stream.begin_step()
            stream.write("step", np.asarray([step], dtype=np.int64))
            stream.write("time", np.asarray([step * 0.5], dtype=np.float64))
            if diagnostic == "field":
                stream.write("uf", np.full((1, 1, 1, 2, 6), step, dtype=np.float64))
                stream.write("um", np.full((1, 1, 1, 2, 1, 14), step, dtype=np.float64))
            elif diagnostic == "load":
                stream.write("load", np.full((2, 2), step, dtype=np.float64))
                stream.write("rank", np.asarray([0, 1], dtype=np.int32))
            else:
                values = np.arange(12, dtype=np.float64).reshape((2, 6)) + step
                stream.write("up00", values)
                stream.write("up00_id", np.asarray([100, 101], dtype=np.uint64))
            stream.end_step()


def make_profile(tmp_path):
    data_dir = tmp_path / "data"
    write_diagnostic(data_dir / "adios" / "field.bp", "field", "field")
    write_diagnostic(data_dir / "adios" / "load.bp", "load", "load")
    write_diagnostic(data_dir / "adios" / "particle.bp", "particle", "particle")
    write_diagnostic(data_dir / "adios" / "tracer.bp", "tracer", "tracer")

    config = {
        "application": {"basedir": "data", "iomode": "adios"},
        "parameter": {
            "Ns": 1,
            "Nx": 2,
            "Ny": 1,
            "Nz": 1,
            "Cx": 1,
            "Cy": 1,
            "Cz": 1,
            "delt": 1.0,
            "delh": 1.0,
        },
        "diagnostic": [
            {"name": "field"},
            {"name": "load"},
            {"name": "particle"},
            {"name": "tracer"},
        ],
    }
    profile = data_dir / "profile.msgpack"
    with profile.open("wb") as fp:
        msgpack.pack(
            {
                "timestamp": {},
                "nprocess": 1,
                "chunkmap": {"chunkid": [[[0]]], "coord": [[0, 0, 0]]},
                "qm": None,
                "configuration": config,
            },
            fp,
        )
    return profile


def test_run_reads_adios_field_and_particle_data(tmp_path):
    run = Run(str(make_profile(tmp_path)))

    assert run.get_diag_handler("field").storage.kind == "adios"
    assert run.get_step("field").tolist() == [0, 1]
    assert run.get_time_at("field", 1) == 0.5

    field = run.read_at("field", 1)
    assert field["uf"].shape == (1, 1, 2, 6)
    assert field["um"].shape == (1, 1, 2, 1, 14)

    load = run.read_at("load", 1)
    assert load["load"].shape == (2, 2)
    np.testing.assert_array_equal(load["rank"], [0, 1])

    particle = run.read_particle_at("particle", 1, start=1, stop=2)
    particle_ids = run.read_particle_id_at("particle", 1, start=1, stop=2)
    np.testing.assert_allclose(
        particle["up00"], np.arange(6, dtype=np.float64).reshape(1, 6) + 7
    )
    assert particle_ids["up00"].tolist() == [101]

    tracer = run.read_at("tracer", 1)["up00"]
    assert tracer.shape == (2, 7)
    np.testing.assert_array_equal(
        np.frombuffer(tracer[:, -1].tobytes(), dtype=np.uint64), [100, 101]
    )
