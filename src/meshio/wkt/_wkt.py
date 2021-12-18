import re
from collections import OrderedDict
from io import StringIO

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

float_pattern = r"[+-]?(?:\d+\.?\d*|\d*\.?\d+)"
float_re = re.compile(float_pattern)

point_pattern = r"{0}\s+{0}\s+{0}(?:\s+{0})?".format(float_pattern)
point_re = re.compile(point_pattern)

triangle_pattern = r"\(\s*\(\s*({})\s*\)\s*\)".format(
    r"\s*,\s*".join(point_pattern for _ in range(4))
)
triangle_re = re.compile(triangle_pattern)

tin_pattern = fr"TIN\s*\((?:\s*{triangle_pattern}\s*,?)*\s*\)"
tin_re = re.compile(tin_pattern)


def read_str(s):
    s = s.strip()
    tin_match = tin_re.match(s)
    if tin_match is None:
        raise ReadError("Invalid WKT TIN")
    point_idxs = OrderedDict()
    tri_idxs = []
    for tri_match in triangle_re.finditer(tin_match.group()):
        tri_point_idxs = []
        for point_match in point_re.finditer(tri_match.group()):
            point = []
            for float_match in float_re.finditer(point_match.group()):
                point.append(float(float_match.group()))
            point = tuple(point)
            if point not in point_idxs:
                point_idxs[point] = len(point_idxs)
            tri_point_idxs.append(point_idxs[point])

        if tri_point_idxs[-1] != tri_point_idxs[0]:
            raise ValueError("Triangle is not a closed linestring")

        tri_idxs.append(tri_point_idxs[:-1])

    try:
        point_arr = np.array(list(point_idxs), np.float64)
    except ValueError as e:
        if len({len(p) for p in point_idxs}) > 1:
            raise ReadError("Points have mixed dimensionality")
        else:
            raise e

    tri_arr = np.array(tri_idxs, np.uint64)

    return Mesh(point_arr, [CellBlock("triangle", tri_arr)])


def arr_to_str(arr):
    return " ".join(str(item) for item in arr)


def read(filename):
    with open_file(filename) as f:
        return read_str(f.read())


def write(filename, mesh):
    with open_file(filename, "w") as f:
        write_buffer(f, mesh)


def write_buffer(f, mesh):
    skip = [c for c in mesh.cells if c.type != "triangle"]
    if skip:
        warn('WTK only supports triangle cells. Skipping {", ".join(skip)}.')

    triangles = mesh.get_cells_type("triangle")

    f.write("TIN (")
    joiner = ""
    for tri_points in mesh.points[triangles]:
        f.write(
            "{0}(({1}, {2}, {3}, {1}))".format(
                joiner, *(arr_to_str(p) for p in tri_points)
            )
        )
        joiner = ", "
    f.write(")")


def write_str(mesh):
    buf = StringIO()
    write_buffer(buf, mesh)
    buf.seek(0)
    return buf.read()


register_format("wkt", [".wkt"], read, {"wkt": write})
