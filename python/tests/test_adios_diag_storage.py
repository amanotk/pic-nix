import numpy as np
import pytest
from picnix.diag import DiagHandler
from picnix.diag_storage import AdiosDiagStorage
from picnix.run import Run


class JoinedArrayReader:
    def __init__(self):
        self.values = np.arange(12, dtype=np.float64).reshape((2, 6))
        self.ids = np.asarray([100, 101], dtype=np.uint64)
        self.reads = []

    def all_blocks_info(self, name):
        if name.endswith("_id"):
            return [[{"Count": "0"}, {"Count": "2"}]]
        return [[{"Count": "0,6"}, {"Count": "2,6"}]]

    def read(self, name, *, start, count, step_selection):
        self.reads.append((name, start, count, step_selection))
        if name.endswith("_id"):
            return self.ids[start[0] : start[0] + count[0]]
        return self.values[
            start[0] : start[0] + count[0],
            start[1] : start[1] + count[1],
        ]


def make_storage(name):
    storage = AdiosDiagStorage(name, name, ".", "adios")
    storage.reader = JoinedArrayReader()
    storage.variables = {"up00": {}, "up00_id": {}}
    storage.step = np.asarray([5], dtype=np.int64)
    storage.time = np.asarray([0.5], dtype=np.float64)
    return storage


class EmptyStorage:
    def get_step(self):
        return np.asarray([], dtype=np.int32)

    def get_time(self):
        return np.asarray([], dtype=np.float64)


@pytest.mark.parametrize(
    ("name", "expected_prefix"), [("tracker", "tracker"), ("tracer", "tracer")]
)
def test_diag_handler_accepts_tracker_alias(monkeypatch, name, expected_prefix):
    monkeypatch.setattr("picnix.diag.create_diag_storage", lambda *args: EmptyStorage())

    handler = DiagHandler.create_handler({"name": name}, ".", "posix")

    assert handler.get_name() == "tracker"
    assert handler.get_prefix() == expected_prefix


def test_run_resolves_tracker_alias_prefixes():
    run = Run.__new__(Run)
    handler = object()
    run.diag_handlers = {"tracer": handler}

    assert run.get_diag_handler("tracker") is handler


@pytest.mark.parametrize("diagnostic", ["particle", "tracker"])
def test_read_at_selects_joined_arrays_with_an_empty_writer(diagnostic):
    storage = make_storage(diagnostic)

    values = storage.read_at(5, ".*")["up00"]

    expected = storage.reader.values
    if diagnostic == "tracker":
        expected = np.concatenate(
            (expected, storage.reader.ids.view(np.float64)[:, None]), axis=1
        )
    np.testing.assert_array_equal(values, expected)
    assert storage.reader.reads[0] == ("up00", [0, 0], [2, 6], [0, 1])
    if diagnostic == "tracker":
        assert storage.reader.reads[1] == ("up00_id", [0], [2], [0, 1])


def test_read_at_returns_empty_joined_arrays_when_all_writers_are_empty():
    storage = make_storage("tracker")
    storage.reader.all_blocks_info = lambda name: [
        [{"Count": "0" if name.endswith("_id") else "0,6"}]
    ]

    values = storage.read_at(5, ".*")["up00"]

    assert values.shape == (0, 7)
    assert storage.reader.reads == []
