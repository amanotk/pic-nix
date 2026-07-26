import json
from collections.abc import Mapping

import numpy as np

SCHEMA_VERSION = 1


def _get(node, key, default=None):
    if isinstance(node, Mapping):
        return node.get(key, default)
    if hasattr(node, "has_path") and not node.has_path(key):
        return default
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        return default


def _children(node):
    if isinstance(node, Mapping):
        return node.items()
    if hasattr(node, "child_names"):
        return ((name, node[name]) for name in node.child_names())
    raise TypeError("expected a mapping or Conduit node")


def _items(node):
    if isinstance(node, Mapping):
        return node.items()
    if hasattr(node, "child_names"):
        return ((name, node[name]) for name in node.child_names())
    return ()


def _value(node):
    if hasattr(node, "number_of_children") and node.number_of_children() > 0:
        return [_value(node[index]) for index in range(node.number_of_children())]
    if hasattr(node, "value"):
        return node.value()
    return node


def _array(node):
    return np.asarray(_value(node))


class Field:
    def __init__(self, node, shape=None):
        self.node = node
        self.shape = tuple(shape) if shape is not None else None

    def _array(self, node):
        values = _array(node)
        if self.shape is not None and values.size == np.prod(self.shape):
            return values.reshape(self.shape)
        return values

    def component(self, name):
        values = _get(self.node, "values")
        children = dict(_items(values))
        if children:
            if name not in children:
                raise KeyError(name)
            return self._array(children[name])
        return self._array(values)

    @property
    def array(self):
        values = _get(self.node, "values")
        children = list(_items(values))
        if children:
            return np.stack([self._array(value) for _, value in children], axis=-1)
        return self._array(values)


class RawField(Field):
    def __init__(self, node):
        super().__init__(node)
        self.components = list(_value(_get(node, "components", [])))
        self.locations = list(_value(_get(node, "component_locations", [])))

    def component(self, name):
        try:
            index = self.components.index(name)
        except ValueError as error:
            raise KeyError(name) from error

        return self.array()[..., index]

    def array(self):
        values = _array(_get(self.node, "values"))
        shape = tuple(_value(_get(self.node, "shape", values.shape)))
        strides = tuple(_value(_get(self.node, "strides_bytes", values.strides)))
        if values.shape == shape and values.strides == strides:
            return values
        return np.ndarray(shape, dtype=np.float64, buffer=values, strides=strides)

    def interior(self, mesh):
        lower = (
            mesh.active_lower
            if hasattr(mesh, "active_lower")
            else _value(_get(mesh, "active_lower", []))
        )
        upper = (
            mesh.active_upper
            if hasattr(mesh, "active_upper")
            else _value(_get(mesh, "active_upper", []))
        )
        slices = tuple(slice(lo, hi + 1) for lo, hi in zip(lower, upper))
        return self.array()[slices]


class ParticleField:
    def __init__(self, node):
        self.node = node
        self.components = list(_value(_get(node, "components", [])))

    def _array(self):
        values = _get(self.node, "values")
        shape_value = _get(self.node, "shape")
        strides_value = _get(self.node, "strides_bytes")
        if values is None:
            return np.empty((0, 7), dtype=np.float64)

        values = _array(values)
        shape = tuple(_value(shape_value)) if shape_value is not None else values.shape
        strides = (
            tuple(_value(strides_value))
            if strides_value is not None
            else values.strides
        )
        if not shape:
            shape = (0, 7)
        if values.shape == shape and values.strides == strides:
            return values
        return np.ndarray(shape, dtype=np.float64, buffer=values, strides=strides)

    @property
    def allocated(self):
        return self._array()

    @property
    def active(self):
        count = int(_value(_get(self.node, "np_active", 0)))
        return self.allocated[:count]

    @property
    def ids(self):
        return np.frombuffer(self.active[:, 6].tobytes(), dtype=np.int64)

    def kinetic_energy(self):
        mass = float(_value(_get(self.node, "mass", 0.0)))
        velocity = self.active[:, 3:6]
        return 0.5 * mass * np.sum(velocity * velocity, axis=1)


class Domain:
    def __init__(self, node):
        self.node = node
        self.mesh = _get(_get(node, "picnix"), "mesh", {})

    @property
    def domain_id(self):
        return int(_value(_get(_get(self.node, "state"), "domain_id", 0)))

    @property
    def active_lower(self):
        return tuple(_value(_get(self.mesh, "active_lower", [])))

    @property
    def active_upper(self):
        return tuple(_value(_get(self.mesh, "active_upper", [])))

    def raw_field(self, name):
        return RawField(_get(_get(_get(self.node, "picnix"), "raw"), name))

    def centered_field(self, name):
        shape = _value(_get(self.mesh, "local_cell_shape", []))
        return Field(_get(_get(self.node, "fields"), name), shape)

    def particles(self, species=0):
        name = f"species_{species:03d}" if isinstance(species, int) else species
        return ParticleField(_get(_get(_get(self.node, "picnix"), "particles"), name))


class Dataset:
    def __init__(self, node):
        self.node = node
        self._validate_schema()

    def _validate_schema(self):
        for _, domain in _children(self.node):
            schema = _get(_get(domain, "picnix"), "schema_version")
            if schema is not None and int(_value(schema)) != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported PIC-NIX schema version: {_value(schema)}"
                )

    @classmethod
    def from_conduit(cls, node):
        return cls(node)

    @classmethod
    def from_ascent(cls, node_or_callable):
        node = node_or_callable() if callable(node_or_callable) else node_or_callable
        return cls(node)

    def local_chunks(self):
        return (Domain(node) for _, node in _children(self.node))

    def domain(self, domain_id):
        for chunk in self.local_chunks():
            if chunk.domain_id == domain_id:
                return chunk
        raise KeyError(domain_id)

    def to_json(self):
        if hasattr(self.node, "to_json"):
            return self.node.to_json()
        return json.dumps(self.node)
