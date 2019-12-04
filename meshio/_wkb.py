"""
replace with a push parser
"""
import sys
from collections import OrderedDict
from enum import Enum, IntEnum

import numpy as np

from meshio import Mesh
from meshio._files import open_file


class Endianness(Enum):
    LITTLE = "<"
    BIG = ">"

    @classmethod
    def from_int(cls, i):
        if i == 0:
            return cls.BIG
        elif i == 1:
            return cls.LITTLE

    def __int__(self):
        cls = type(self)
        if self == cls.LITTLE:
            return 1
        else:
            return 0

    def to_uint8(self):
        return np.uint8(int(self))

    def to_bytes(self):
        return self.to_uint8().to_bytes()

    def to_buffer(self, buf):
        return buf.write(self.to_bytes())

    def to_word(self):
        return self.name.lower()

    @classmethod
    def from_buffer(cls, buf):
        i = read_num(buf, MACHINE_ENDIANNESS, "uint8")
        return cls.from_int(i)


MACHINE_ENDIANNESS = Endianness.LITTLE if sys.byteorder == "little" else Endianness.BIG


class WKBType(IntEnum):
    POLYGON = 3
    POLYGON_Z = 1003
    TIN = 16
    TIN_Z = 1016

    @classmethod
    def from_buffer(cls, buf, endianness: Endianness):
        i = read_num(buf, endianness, "uint32")
        return cls(i)

    def to_uint32(self):
        return np.uint32(self.value)

    def to_buffer(self, buf, endianness=MACHINE_ENDIANNESS):
        b = self.to_bytes(4, endianness.to_word(), signed=False)
        buf.write(b)

    def ndim(self):
        return 2 + int(self/1000)


def read_arr(buf, endianness: Endianness, count: int, dtype="float64"):
    dt = np.dtype(dtype)
    dt.newbyteorder(endianness.value)
    return np.frombuffer(buf, dt, count)


def read_num(buf, endianness, dtype="uint32"):
    return read_arr(buf, endianness, 1, dtype).item()


def push_parse_tin(b):
    end = Endianness.from_buffer(b)
    wkb_type = WKBType.from_buffer(b, end)
    if wkb_type == WKBType.TIN:
        dim = 2
    elif wkb_type == WKBType.TIN_Z:
        dim = 3
    else:
        raise ValueError("Not a TIN")
    n_polygons = read_num(b, end)
    for _ in range(n_polygons):
        yield parse_polygon(b, dim, end)


def parse_polygon(b, dim, end):
    this_end = Endianness.from_buffer(b)
    if this_end != end:
        raise ValueError("Inconsistent endianness")
    wkb_type = WKBType.from_buffer(b, end)
    if wkb_type not in (WKBType.POLYGON, WKBType.POLYGON_Z):
        raise ValueError("Not a polygon")
    if wkb_type.ndim() != dim:
        raise ValueError("Inconsistent dimensionality")
    count = read_num(b, end)
    if count != 1:
        raise ValueError("More than one ring")
    return parse_ring(b, dim, end)


def parse_ring(b, dim, end):
    count = read_num(b, end)
    arr = read_arr(b, end, count*dim).reshape((count, dim))
    if len(arr) != 4:
        raise ValueError("Ring is not a triangle")
    if not np.array_equal(arr[-1], arr[0]):
        raise ValueError("Ring is not closed")
    return [tuple(row) for row in arr[:-1]]


def triangles_to_mesh(tris):
    point_idxs = OrderedDict()
    tri_idxs = []
    for tri in tris:
        tri_point_idxs = []
        for point in tri:
            if point not in point_idxs:
                point_idxs[point] = len(point_idxs)
            point_idx = point_idxs.setdefault(point, len(point_idxs))
            tri_point_idxs.append(point_idx)

    point_arr = np.array(list(point_idxs), np.float64)
    tri_arr = np.array(tri_idxs, np.uint64)

    return Mesh(point_arr, {"triangle": tri_arr})


def read(f):
    with open_file(f, "rb"):
        return read_buffer(f)


def read_buffer(f):
    return triangles_to_mesh(
        push_parse_tin(f)
    )


def write(f, mesh):
    with open_file(f, "wb"):
        return write_buffer(f, mesh)


def write_buffer(f, mesh):
    dim = mesh.points.shape[1]
    if dim == 2:
        poly_type = WKBType.POLYGON
        tin_type = WKBType.TIN
    elif dim == 3:
        poly_type = WKBType.POLYGON_Z
        tin_type = WKBType.TIN_Z
    else:
        raise ValueError("Must be 2 or 3D")
    try:
        tri_points = mesh.points[mesh.cells["triangle"]]
    except KeyError:
        raise ValueError("Must have triangular cells")
    byteorder = tri_points.dtype.byteorder
    end = MACHINE_ENDIANNESS if byteorder == "=" else Endianness(byteorder)
    end_bytes = MACHINE_ENDIANNESS.to_bytes()

    dtype = np.dtype("float64")
    dtype.newbyteorder(end.value)
    tri_points = np.asarray(tri_points, dtype=dtype, order="C")

    f.write(end_bytes)
    tin_type.to_buffer(f)
    end_word = end.to_word()
    f.write(len(tri_points).to_bytes(4, end_word, signed=False))

    # number of rings in polygon
    one = int(1).to_bytes(4, end_word, signed=False)
    # number of points in ring
    four = int(4).to_bytes(4, end_word, signed=False)

    tri: np.ndarray
    for tri in tri_points:
        f.write(end_bytes)
        poly_type.to_buffer(f, end_bytes)
        f.write(one)  # number of rings (ash nazg)
        f.write(four)  # number of points in ring
        f.write(tri.tobytes())
        f.write(tri[-1].tobytes())
