import numpy as np
import pytest

from picnix.insitu import Dataset


def make_dataset():
    raw = np.arange(4 * 5 * 6 * 6, dtype=np.float64).reshape(4, 5, 6, 6)
    particles = np.zeros((2, 7), dtype=np.float64)
    particles[0, 6] = np.array([-7], dtype=np.int64).view(np.float64)[0]
    return {
        "domain_3": {
            "state": {"domain_id": 3},
            "picnix": {
                "schema_version": 1,
                "mesh": {"active_lower": [1, 1, 1], "active_upper": [2, 3, 4]},
                "raw": {
                    "uf": {
                        "values": raw,
                        "shape": list(raw.shape),
                        "strides_bytes": list(raw.strides),
                        "components": ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
                    }
                },
                "particles": {
                    "species_000": {
                        "values": particles,
                        "np_active": 1,
                        "mass": 2.0,
                        "components": ["x", "y", "z", "ux", "uy", "uz", "id_bits"],
                    }
                },
            },
            "fields": {"E": {"values": {"x": np.ones((2, 3, 4))}}},
        }
    }


def test_dataset_views_and_particle_id_bits():
    dataset = Dataset(make_dataset())
    chunk = next(dataset.local_chunks())

    assert chunk.domain_id == 3
    assert chunk.raw_field("uf").component("Ex").shape == (4, 5, 6)
    assert chunk.raw_field("uf").interior(chunk.mesh).shape == (2, 3, 4, 6)
    assert chunk.centered_field("E").component("x")[0, 0, 0] == 1.0
    assert chunk.particles().ids.tolist() == [-7]


def test_dataset_rejects_unknown_schema_version():
    data = make_dataset()
    data["domain_3"]["picnix"]["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported PIC-NIX schema version"):
        Dataset(data)


def test_missing_particle_species_is_empty():
    chunk = next(Dataset(make_dataset()).local_chunks())

    particles = chunk.particles("species_001")
    assert particles.allocated.shape == (0, 7)
    assert particles.ids.size == 0


def test_from_ascent_requires_explicit_node():
    with pytest.raises(TypeError):
        Dataset.from_ascent()

    assert next(Dataset.from_ascent(make_dataset()).local_chunks()).domain_id == 3


def test_top_level_wildcard_exports_legacy_names():
    namespace = {}
    exec("from picnix import *", namespace)
    assert "Run" in namespace
    assert "Tracer" in namespace
    assert "get_wk_spectrum" in namespace
