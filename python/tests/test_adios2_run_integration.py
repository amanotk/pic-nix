#!/usr/bin/env python
# -*- coding: utf-8 -*-

import msgpack
import numpy as np
import pytest

adios2 = pytest.importorskip("adios2")

from picnix import Run  # noqa: E402


def write_diagnostic(
    path,
    diagnostic,
    prefix,
    steps=range(2),
    value_offset=0,
    segment_index=0,
    restart_step=-1,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with adios2.Stream(str(path), "w") as stream:
        stream.write_attribute("picnix_schema", "diagnostic-bp-v1")
        stream.write_attribute("diagnostic", diagnostic)
        stream.write_attribute("prefix", prefix)
        stream.write_attribute("segment_index", segment_index)
        stream.write_attribute("restart_step", restart_step)
        for step in steps:
            stream.begin_step()
            stream.write("step", np.asarray([step], dtype=np.int64))
            stream.write("time", np.asarray([step * 0.5], dtype=np.float64))
            if diagnostic == "field":
                value = step + value_offset
                stream.write("uf", np.full((1, 1, 1, 2, 6), value, dtype=np.float64))
                stream.write(
                    "um", np.full((1, 1, 1, 2, 1, 14), value, dtype=np.float64)
                )
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


def test_run_merges_adios_restart_segments(tmp_path):
    profile = make_profile(tmp_path)
    write_diagnostic(
        profile.parent / "adios" / "field.bp",
        "field",
        "field",
        steps=range(5),
    )
    write_diagnostic(
        profile.parent / "adios" / "field.part0001.bp",
        "field",
        "field",
        steps=range(4, 7),
        value_offset=100,
        segment_index=1,
        restart_step=3,
    )
    (profile.parent / "adios" / "field.part0002.bp.tmp").mkdir()

    run = Run(str(profile))

    assert run.get_step("field").tolist() == [0, 1, 2, 4, 5, 6]
    assert run.get_time("field").tolist() == [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    np.testing.assert_array_equal(run.read_at("field", 0)["uf"], 0.0)
    np.testing.assert_array_equal(run.read_at("field", 2)["uf"], 2.0)
    np.testing.assert_array_equal(run.read_at("field", 4)["uf"], 104.0)
    np.testing.assert_array_equal(run.read_at("field", 6)["uf"], 106.0)


@pytest.mark.parametrize(
    ("filename", "segment_index", "message"),
    [
        ("field.part0002.bp", 2, "missing ADIOS2 segment"),
        ("field.part1.bp", 1, "noncanonical ADIOS2 segment name"),
        ("field.part0000.bp", 0, "noncanonical ADIOS2 segment name"),
    ],
)
def test_run_rejects_invalid_adios_segment_sequences(
    tmp_path, filename, segment_index, message
):
    profile = make_profile(tmp_path)
    write_diagnostic(
        profile.parent / "adios" / filename,
        "field",
        "field",
        steps=[2],
        segment_index=segment_index,
        restart_step=2,
    )

    with pytest.raises(ValueError, match=message):
        Run(str(profile))
