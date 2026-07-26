import numpy as np

from picnix.insitu import Dataset


def make_dataset():
    raw = np.arange(4 * 5 * 6 * 6, dtype=np.float64).reshape(4, 5, 6, 6)
    particles = np.zeros((2, 7), dtype=np.float64)
    particles[0, 6] = np.array([-7], dtype=np.int64).view(np.float64)[0]
    return {
        "domain_3": {
            "state": {"domain_id": 3},
            "picnix": {
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
