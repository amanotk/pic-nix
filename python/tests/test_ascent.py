import json

import numpy as np
import pytest

from picnix.ascent import Dataset


def make_domain(
    domain_id,
    dimension,
    *,
    metadata=None,
    raw=False,
    particles=False,
    mass_current=True,
    energy_momentum=False,
):
    cell_xyz = (4, 3, 2)[:dimension]
    cell_shape = (1,) * (3 - dimension) + tuple(reversed(cell_xyz))
    cell_count = int(np.prod(cell_shape))
    dims = dict(zip("ijk"[:dimension], (size + 1 for size in cell_xyz), strict=True))

    def field(components, offset):
        # Deliberately reverse insertion order to test canonical component ordering.
        return {
            "association": "element",
            "topology": "cell_mesh",
            "values": {
                name: np.arange(cell_count, dtype=np.float64) + offset + index * 100
                for index, name in reversed(list(enumerate(components)))
            },
        }

    def scalar_field(offset):
        return {
            "association": "element",
            "topology": "cell_mesh",
            "values": np.arange(cell_count, dtype=np.float64) + offset,
        }

    mass_current_components = ("M0", "Mx", "My", "Mz")
    energy_momentum_components = (
        "Ttt",
        "Txx",
        "Tyy",
        "Tzz",
        "Ttx",
        "Tty",
        "Ttz",
        "Txy",
        "Tyz",
        "Tzx",
    )
    moment_components = []
    if mass_current:
        moment_components.extend(enumerate(mass_current_components))
    if energy_momentum:
        moment_components.extend(enumerate(energy_momentum_components, start=4))
    moment_fields = {
        f"um03_{name}": scalar_field(20 + index * 100)
        for index, name in reversed(moment_components)
    }

    domain = {
        "state": {"domain_id": domain_id, "cycle": 5, "time": 1.5},
        "coordsets": {"cell_coords": {"type": "uniform", "dims": dims}},
        "topologies": {"cell_mesh": {"type": "uniform", "coordset": "cell_coords"}},
        "fields": {
            "E": field(("x", "y", "z"), 0),
            "B": field(("x", "y", "z"), 10),
            **moment_fields,
        },
        "pic": {},
    }
    if metadata is not None:
        domain["pic"].update(metadata)

    if raw:
        raw_xyz = tuple(size + 2 for size in cell_xyz)
        raw_shape = (1,) * (3 - dimension) + tuple(reversed(raw_xyz))
        raw_count = int(np.prod(raw_shape))
        domain["pic"]["raw"] = {"shape": np.asarray(raw_shape, dtype=np.int64)}
        for name, components in (
            ("uf", ("Ex", "Ey", "Ez", "Bx", "By", "Bz")),
            ("uj", ("rho", "Jx", "Jy", "Jz")),
        ):
            values = np.stack(
                [
                    np.arange(raw_count, dtype=np.float64) + index * 1000
                    for index, _ in enumerate(components)
                ],
                axis=-1,
            )
            domain["pic"]["raw"][name] = values.reshape(-1)
        domain["pic"]["neighbors"] = {
            "domain_ids": np.arange(27, dtype=np.int64),
            "neighbor_ranks": np.arange(27, dtype=np.int32) + 100,
        }
        domain["pic"]["neighbors"]["domain_ids"][13] = domain_id

    if particles:
        xu = np.arange(21, dtype=np.float64).reshape(3, 7)
        xu[:, 6] = [101.0, 102.0, 103.0]
        domain["pic"]["particles"] = {
            "particle00": {"xu": xu},
            "particle12": {"xu": np.arange(7, dtype=np.float64)},
        }
    return domain


def metadata(config=None):
    return {
        "boundary_margin": 1,
        "config": json.dumps({"simulation": {"dimension": 3}})
        if config is None
        else config,
    }


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [(1, (1, 1, 4)), (2, (1, 3, 4)), (3, (2, 3, 4))],
)
def test_topology_shapes_are_normalized_to_three_dimensions(dimension, expected):
    dataset = Dataset({"domain": make_domain(7, dimension, metadata=metadata())})
    domain = dataset.domain(7)

    assert domain.E.shape == (*expected, 3)
    assert domain.B.shape == (*expected, 3)
    assert domain.um03.shape == (*expected, 4)
    assert domain.E.component("x").shape == expected


def test_canonical_component_order_and_domain_mapping():
    dataset = Dataset(
        {
            "not_the_lowest_id": make_domain(9, 2, metadata=metadata()),
            "domain_2": make_domain(2, 2),
        }
    )
    chunks = list(dataset.local_chunks())

    assert [chunk.domain_id for chunk in chunks] == [9, 2]
    assert dataset.domain(2).domain_id == 2
    assert chunks[0].E.components == ("x", "y", "z")
    assert chunks[0].E[0, 0, 0].tolist() == [0.0, 100.0, 200.0]
    assert chunks[0].um03.components == ("M0", "Mx", "My", "Mz")
    assert chunks[0].um03[0, 0, 0].tolist() == [20.0 + 100 * i for i in range(4)]
    assert chunks[0].um03.Mx[0, 0, 0] == 120.0
    with pytest.raises(KeyError):
        chunks[0].um03.component("Ttx")
    with pytest.raises(AttributeError):
        _ = chunks[0].um03.Ttx
    with pytest.raises(KeyError):
        dataset.domain(100)


def test_energy_momentum_extends_grouped_moment_in_deposited_order():
    domain = Dataset(
        {"domain": make_domain(3, 2, metadata=metadata(), energy_momentum=True)}
    ).domain(3)

    assert domain.um03.components == (
        "M0",
        "Mx",
        "My",
        "Mz",
        "Ttt",
        "Txx",
        "Tyy",
        "Tzz",
        "Ttx",
        "Tty",
        "Ttz",
        "Txy",
        "Tyz",
        "Tzx",
    )
    assert domain.um03.shape == (1, 3, 4, 14)
    assert domain.um03.Tzx[0, 0, 0] == 1320.0
    assert domain.moment_field("um03").component("Ttx")[0, 0, 0] == 820.0


def test_energy_momentum_can_be_published_without_mass_current():
    domain = Dataset(
        {
            "domain": make_domain(
                3,
                1,
                metadata=metadata(),
                mass_current=False,
                energy_momentum=True,
            )
        }
    ).domain(3)

    assert domain.um03.components == (
        "Ttt",
        "Txx",
        "Tyy",
        "Tzz",
        "Ttx",
        "Tty",
        "Ttz",
        "Txy",
        "Tyz",
        "Tzx",
    )
    assert domain.um03.shape == (1, 1, 4, 10)
    assert domain.um03.Ttt[0, 0, 0] == 420.0


def test_metadata_owner_accepts_json_and_object_config():
    json_dataset = Dataset({"a": make_domain(8, 1, metadata=metadata())})
    object_dataset = Dataset(
        {
            "a": make_domain(
                8,
                1,
                metadata=metadata({"simulation": {"dimension": 1}, "species": [1, 2]}),
            )
        }
    )

    assert json_dataset.boundary_margin == 1
    assert json_dataset.config["simulation"]["dimension"] == 3
    assert object_dataset.config == {
        "simulation": {"dimension": 1},
        "species": [1, 2],
    }


def test_conduit_config_preserves_lists_of_objects():
    conduit = pytest.importorskip("conduit")
    config = conduit.Node()
    config["species"].append()["charge"] = -1.0
    config["species"].append()["charge"] = 1.0

    dataset = Dataset({"domain": make_domain(8, 1, metadata=metadata(config))})

    assert dataset.config == {"species": [{"charge": -1.0}, {"charge": 1.0}]}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data["a"]["pic"].pop("config"), "exactly one local owner"),
        (
            lambda data: data["b"]["pic"].update(metadata()),
            "exactly one local owner",
        ),
        (
            lambda data: data["b"]["pic"].update(
                {"config": data["a"]["pic"].pop("config")}
            ),
            "colocated",
        ),
    ],
)
def test_metadata_errors(mutate, message):
    data = {"a": make_domain(4, 1, metadata=metadata()), "b": make_domain(5, 1)}
    mutate(data)

    with pytest.raises(ValueError, match=message):
        Dataset(data)


def test_invalid_config_is_rejected_during_construction():
    data = {"domain": make_domain(4, 1, metadata=metadata("not JSON"))}

    with pytest.raises(json.JSONDecodeError):
        Dataset(data)


@pytest.mark.parametrize(
    ("dimension", "raw_shape", "owned_shape"),
    [
        (1, (1, 1, 6, 6), (1, 1, 4, 6)),
        (2, (1, 5, 6, 6), (1, 3, 4, 6)),
        (3, (4, 5, 6, 6), (2, 3, 4, 6)),
    ],
)
def test_raw_fields_preserve_order_and_crop_owned_cells(
    dimension, raw_shape, owned_shape
):
    domain = next(
        Dataset(
            {"domain": make_domain(3, dimension, metadata=metadata(), raw=True)}
        ).local_chunks()
    )

    assert domain.uf.shape == raw_shape
    assert domain.uj.shape == (*raw_shape[:-1], 4)
    assert domain.uf[0, 0, 0].tolist() == [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    assert domain.uf_owned.shape == owned_shape
    assert domain.uj_owned.shape == (*owned_shape[:-1], 4)
    assert domain.raw_field("uf").interior().shape == owned_shape


def test_raw_fields_and_components_share_flat_source_storage():
    source_domain = make_domain(3, 2, metadata=metadata(), raw=True)
    source = source_domain["pic"]["raw"]["uf"]
    field = Dataset({"domain": source_domain}).domain(3).uf

    assert np.shares_memory(field.array, source)
    assert np.shares_memory(field.component("By"), source)
    source[4] = 42.0
    assert field.component("By")[0, 0, 0] == 42.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw.update(shape=np.array([1, 2], dtype=np.int64)),
            "three integers",
        ),
        (
            lambda raw: raw.update(shape=np.array([[1, 1, 6]], dtype=np.int64)),
            "three integers",
        ),
        (
            lambda raw: raw.update(shape=np.array([1, 0, 2], dtype=np.int64)),
            "positive",
        ),
        (lambda raw: raw.update(uf=raw["uf"][:-1]), "expected"),
        (lambda raw: raw.update(uf=raw["uf"].astype(np.float32)), "float64"),
        (lambda raw: raw.update(uf=raw["uf"].reshape(1, -1)), "flat"),
    ],
)
def test_invalid_raw_storage_is_rejected(mutate, message):
    source_domain = make_domain(3, 1, metadata=metadata(), raw=True)
    mutate(source_domain["pic"]["raw"])
    domain = Dataset({"domain": source_domain}).domain(3)

    with pytest.raises(ValueError, match=message):
        _ = domain.uf.array


def test_strided_component_shape_restoration_preserves_source_view():
    domain = make_domain(3, 1, metadata=metadata())
    source = np.arange(12, dtype=np.float64).reshape(4, 3)
    domain["fields"]["E"]["values"] = {
        "x": source[:, 0],
        "y": source[:, 1],
        "z": source[:, 2],
    }
    electric = Dataset({"domain": domain}).domain(3).E

    assert electric.component("y").tolist() == [[[1.0, 4.0, 7.0, 10.0]]]
    assert np.shares_memory(electric.component("y"), source)
    source[2, 1] = 42.0
    assert electric.component("y")[0, 0, 2] == 42.0


def test_neighbors_are_paired_and_validated():
    data = {"domain": make_domain(42, 2, metadata=metadata(), raw=True)}
    domain = next(Dataset(data).local_chunks())

    assert domain.neighbor_domain_ids.shape == (27,)
    assert domain.neighbor_ranks.shape == (27,)
    assert domain.neighbor_domain_ids[13] == 42

    data["domain"]["pic"]["neighbors"].pop("neighbor_ranks")
    with pytest.raises(ValueError, match="both neighbor arrays"):
        Dataset(data)


def test_particles_are_active_float64_arrays_with_two_digit_names():
    domain = next(
        Dataset(
            {"domain": make_domain(1, 3, metadata=metadata(), particles=True)}
        ).local_chunks()
    )

    assert domain.particle00.shape == (3, 7)
    assert domain.particle00.components == ("x", "y", "z", "ux", "uy", "uz", "id")
    assert domain.particle00.ids.dtype == np.float64
    assert domain.particle00.ids.tolist() == [101.0, 102.0, 103.0]
    assert domain.particles.particle12.shape == (1, 7)
    assert domain.particle01.shape == (0, 7)


def test_particle_storage_must_be_float64():
    domain = make_domain(1, 1, metadata=metadata(), particles=True)
    domain["pic"]["particles"]["particle00"]["xu"] = np.ones((2, 7), np.float32)
    particle = next(Dataset({"domain": domain}).local_chunks()).particle00

    with pytest.raises(ValueError, match="float64"):
        _ = particle.active


def test_from_conduit_and_to_json():
    data = {"domain": make_domain(6, 1, metadata=metadata())}

    assert Dataset.from_conduit(data).domain(6).domain_id == 6
    assert json.loads(Dataset(data).to_json())["domain"]["state"]["domain_id"] == 6


def test_top_level_wildcard_exports_legacy_names():
    namespace = {}
    exec("from picnix import *", namespace)
    expected = {
        "DEFAULT_LOG_PREFIX",
        "DEFAULT_LOAD_PREFIX",
        "DEFAULT_FIELD_PREFIX",
        "DEFAULT_PARTICLE_PREFIX",
        "DEFAULT_TRACER_PREFIX",
    }
    for module_name in ("utils", "field", "ohm", "particle", "run"):
        module = __import__(f"picnix.{module_name}", fromlist=[module_name])
        expected.update(
            getattr(module, "__all__", None)
            or (name for name in vars(module) if not name.startswith("_"))
        )

    assert expected <= namespace.keys()
