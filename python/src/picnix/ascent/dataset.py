import json
import re
from collections.abc import Mapping

import numpy as np

_CENTERED_COMPONENTS = {
    "E": ("x", "y", "z"),
    "B": ("x", "y", "z"),
}
_MASS_CURRENT_COMPONENTS = ("M0", "Mx", "My", "Mz")
_ENERGY_MOMENTUM_COMPONENTS = (
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
_MOMENT_COMPONENTS = _MASS_CURRENT_COMPONENTS + _ENERGY_MOMENTUM_COMPONENTS
_RAW_COMPONENTS = {
    "uf": ("Ex", "Ey", "Ez", "Bx", "By", "Bz"),
    "uj": ("rho", "Jx", "Jy", "Jz"),
}
_PARTICLE_COMPONENTS = ("x", "y", "z", "ux", "uy", "uz", "id")
_SHARED_METADATA = ("boundary_margin", "config")


def _get(node, key, default=None):
    if node is None:
        return default
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
    if node is None:
        return ()
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


def _object(node):
    if isinstance(node, Mapping):
        return {name: _object(value) for name, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_object(value) for value in node]
    dtype_method = getattr(node, "dtype", None)
    if callable(dtype_method):
        dtype = dtype_method()
        if dtype.is_object():
            return {name: _object(value) for name, value in _items(node)}
        if dtype.is_list():
            return [_object(node[index]) for index in range(node.number_of_children())]
    value = _value(node)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Field:
    def __init__(self, node, shape, components):
        if node is None:
            raise KeyError("field is not published")
        self.node = node
        self.spatial_shape = tuple(shape)
        self.components = tuple(components)

    def _component_array(self, node):
        values = _array(node)
        expected = int(np.prod(self.spatial_shape))
        if values.size != expected:
            raise ValueError(
                f"field component has {values.size} values, expected {expected}"
            )
        if values.shape == self.spatial_shape:
            return values
        if values.ndim != 1:
            raise ValueError("field components must be flat or have the topology shape")
        element_stride = values.strides[0]
        nz, ny, nx = self.spatial_shape
        return np.lib.stride_tricks.as_strided(
            values,
            shape=self.spatial_shape,
            strides=(ny * nx * element_stride, nx * element_stride, element_stride),
            writeable=values.flags.writeable,
        )

    def component(self, name):
        if name not in self.components:
            raise KeyError(name)
        values = _get(self.node, "values")
        component = _get(values, name)
        if component is None:
            raise ValueError(f"field is missing canonical component {name!r}")
        return self._component_array(component)

    @property
    def array(self):
        return np.stack([self.component(name) for name in self.components], axis=-1)

    @property
    def shape(self):
        return (*self.spatial_shape, len(self.components))

    def __array__(self, dtype=None, copy=None):
        array = self.array
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        return self.array[key]


class RawField(Field):
    def __init__(self, node, shape, components, boundary_margin):
        super().__init__(node, shape, components)
        self.boundary_margin = boundary_margin

    @property
    def array(self):
        values = _array(self.node)
        if values.dtype != np.float64:
            raise ValueError("raw field arrays must use float64 storage")
        expected = int(np.prod(self.spatial_shape)) * len(self.components)
        if values.size != expected:
            raise ValueError(f"raw field has {values.size} values, expected {expected}")
        if values.ndim != 1:
            raise ValueError("raw field arrays must be flat")
        return values.reshape((*self.spatial_shape, len(self.components)))

    def component(self, name):
        try:
            index = self.components.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.array[..., index]

    @property
    def owned(self):
        margin = self.boundary_margin
        if margin == 0:
            return self.array
        slices = []
        for size in self.spatial_shape:
            if size == 1:
                slices.append(slice(None))
            elif size <= 2 * margin:
                raise ValueError("boundary margin leaves no owned raw cells")
            else:
                slices.append(slice(margin, -margin))
        return self.array[tuple(slices)]

    def interior(self, mesh=None):
        return self.owned


class MomentField(Field):
    def __init__(self, fields, name, shape):
        components = tuple(
            component
            for component in _MOMENT_COMPONENTS
            if _get(fields, f"{name}_{component}") is not None
        )
        if not components:
            raise KeyError("moment is not published")
        super().__init__(fields, shape, components)
        self.name = name

    def component(self, name):
        if name not in self.components:
            raise KeyError(name)
        field = _get(self.node, f"{self.name}_{name}")
        values = _get(field, "values")
        if values is None:
            raise ValueError(f"moment is missing canonical component {name!r}")
        return self._component_array(values)

    def __getattr__(self, name):
        if name in _MOMENT_COMPONENTS:
            try:
                return self.component(name)
            except KeyError as error:
                raise AttributeError(
                    f"moment component {name!r} is not published"
                ) from error
        raise AttributeError(name)


class ParticleField:
    components = _PARTICLE_COMPONENTS

    def __init__(self, node=None):
        self.node = node

    @property
    def active(self):
        if self.node is None:
            return np.empty((0, 7), dtype=np.float64)
        values = _array(self.node)
        if values.size % 7:
            raise ValueError("particle array length is not divisible by 7")
        values = values.reshape((-1, 7))
        if values.dtype != np.float64:
            raise ValueError("particle arrays must use float64 storage")
        return values

    @property
    def allocated(self):
        return self.active

    @property
    def array(self):
        return self.active

    @property
    def shape(self):
        return self.active.shape

    @property
    def xu(self):
        return self.active

    @property
    def ids(self):
        return self.active[:, 6]

    def kinetic_energy(self, mass):
        velocity = self.active[:, 3:6]
        return 0.5 * mass * np.sum(velocity * velocity, axis=1)

    def __array__(self, dtype=None, copy=None):
        array = self.active
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        return self.active[key]


class _Particles:
    def __init__(self, domain):
        self._domain = domain

    def __getattr__(self, name):
        if re.fullmatch(r"particle\d{2,}", name):
            return self._domain._particle(name)
        raise AttributeError(name)


class Domain:
    def __init__(self, node, dataset):
        self.node = node
        self.dataset = dataset

    @property
    def domain_id(self):
        value = _get(_get(self.node, "state"), "domain_id")
        if value is None:
            raise ValueError("domain is missing state/domain_id")
        return int(_value(value))

    def _topology_shape(self, topology_name):
        topology = _get(_get(self.node, "topologies"), topology_name)
        if topology is None:
            raise KeyError(topology_name)
        coordset_name = _value(_get(topology, "coordset"))
        coordset = _get(_get(self.node, "coordsets"), coordset_name)
        dims = _get(coordset, "dims")
        extents = []
        for axis in ("i", "j", "k"):
            value = _get(dims, axis)
            if value is None:
                break
            extent = int(_value(value)) - 1
            if extent < 1:
                raise ValueError(f"invalid {topology_name} coordset dimension {axis}")
            extents.append(extent)
        if not extents:
            raise ValueError(f"cannot derive shape for topology {topology_name!r}")
        return (1,) * (3 - len(extents)) + tuple(reversed(extents))

    def raw_field(self, name):
        try:
            components = _RAW_COMPONENTS[name]
        except KeyError as error:
            raise KeyError(name) from error
        raw = _get(_get(self.node, "pic"), "raw")
        node = _get(raw, name)
        if node is None:
            raise KeyError("field is not published")
        shape = _array(_get(raw, "shape"))
        if (
            shape.ndim != 1
            or shape.size != 3
            or not np.issubdtype(shape.dtype, np.integer)
        ):
            raise ValueError("pic/raw/shape must contain three integers")
        shape = tuple(int(value) for value in shape)
        if any(value < 1 for value in shape):
            raise ValueError("pic/raw/shape extents must be positive")
        return RawField(
            node,
            shape,
            components,
            self.dataset.boundary_margin,
        )

    def centered_field(self, name):
        try:
            components = _CENTERED_COMPONENTS[name]
        except KeyError as error:
            raise KeyError(name) from error
        node = _get(_get(self.node, "fields"), name)
        return Field(node, self._topology_shape("cell_mesh"), components)

    def moment_field(self, name):
        if not re.fullmatch(r"um\d{2,}", name):
            raise KeyError(name)
        fields = _get(self.node, "fields")
        return MomentField(fields, name, self._topology_shape("cell_mesh"))

    @property
    def particles(self):
        return _Particles(self)

    def _particle(self, species=0):
        name = f"particle{species:02d}" if isinstance(species, int) else species
        particle = _get(_get(_get(self.node, "pic"), "particles"), name)
        return ParticleField(_get(particle, "xu"))

    @property
    def E(self):
        return self.centered_field("E")

    @property
    def B(self):
        return self.centered_field("B")

    @property
    def uf(self):
        return self.raw_field("uf")

    @property
    def uj(self):
        return self.raw_field("uj")

    @property
    def uf_owned(self):
        return self.uf.owned

    @property
    def uj_owned(self):
        return self.uj.owned

    @property
    def neighbor_domain_ids(self):
        return self._neighbors("domain_ids")

    @property
    def neighbor_ranks(self):
        return self._neighbors("neighbor_ranks")

    def _neighbors(self, name):
        neighbors = _get(_get(self.node, "pic"), "neighbors")
        value = _get(neighbors, name)
        if value is None:
            raise KeyError(f"pic/neighbors/{name}")
        array = _array(value)
        if array.size != 27 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"neighbor {name} must contain 27 integers")
        return array.reshape(27)

    def __getattr__(self, name):
        if re.fullmatch(r"um\d{2,}", name):
            return self.moment_field(name)
        if re.fullmatch(r"particle\d{2,}", name):
            return self._particle(name)
        raise AttributeError(name)


class Dataset:
    def __init__(self, node):
        self.node = node
        self._domains = list(_children(node))
        self._metadata_node = self._validate_metadata()
        self._boundary_margin = self._parse_boundary_margin()
        self._config = self._parse_config()

    def _validate_metadata(self):
        if not self._domains:
            raise ValueError("dataset has no local domains")

        owners = {name: [] for name in _SHARED_METADATA}
        for domain_name, domain in self._domains:
            pic = _get(domain, "pic")
            for name in _SHARED_METADATA:
                if _get(pic, name) is not None:
                    owners[name].append(domain_name)

            neighbors = _get(pic, "neighbors")
            has_ids = _get(neighbors, "domain_ids") is not None
            has_ranks = _get(neighbors, "neighbor_ranks") is not None
            if has_ids != has_ranks:
                raise ValueError(
                    f"domain {domain_name!r} must publish both neighbor arrays"
                )

        if any(len(names) != 1 for names in owners.values()):
            raise ValueError("shared PIC metadata must have exactly one local owner")
        owner_names = {names[0] for names in owners.values()}
        if len(owner_names) != 1:
            raise ValueError(
                "shared PIC metadata must be colocated on one local domain"
            )

        owner_name = owner_names.pop()
        owner = dict(self._domains)[owner_name]
        return _get(owner, "pic")

    @property
    def boundary_margin(self):
        return self._boundary_margin

    def _parse_boundary_margin(self):
        margin = int(_value(_get(self._metadata_node, "boundary_margin")))
        if margin < 0:
            raise ValueError("boundary margin must be non-negative")
        return margin

    @property
    def config(self):
        return self._config

    def _parse_config(self):
        config = _object(_get(self._metadata_node, "config"))
        if isinstance(config, bytes):
            config = config.decode()
        if isinstance(config, str):
            config = json.loads(config)
        if not isinstance(config, dict):
            raise ValueError("pic/config must be a JSON object or object tree")
        return config

    @classmethod
    def from_conduit(cls, node):
        return cls(node)

    def local_chunks(self):
        return (Domain(node, self) for _, node in self._domains)

    def domain(self, domain_id):
        for chunk in self.local_chunks():
            if chunk.domain_id == domain_id:
                return chunk
        raise KeyError(domain_id)

    def to_json(self):
        if hasattr(self.node, "to_json"):
            return self.node.to_json()
        return json.dumps(_object(self.node))
