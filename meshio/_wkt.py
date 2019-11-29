from collections import OrderedDict
from io import StringIO

import numpy as np

from ._exceptions import ReadError, WriteError
from ._files import open_file
from ._mesh import Mesh


def extract_float(lst):
    return float("".join(lst))


def extract_point(lst):
    return [tuple(lst)]


def extract_tri(lst):
    nums = [lst[2], lst[4], lst[6]]
    last = lst[8]
    if last != nums[0]:
        raise ReadError("Triangle did not start and end at equal points")
    return [nums]


def extract_tris(lst):
    return lst[2:-1]


def initialise_parser():
    try:
        from pyparsing import Optional, Regex, Literal, OneOrMore
    except ImportError:
        return None

    number = Regex(r"[+-]?(\d+\.?\d*|\d*\.?\d+)").setParseAction(extract_float)
    point = (number * (2, 3)).setParseAction(extract_point)
    triangle = (
        Literal("(")
        + "("
        + point
        + ","
        + point
        + ","
        + point
        + ","
        + point
        + ")"
        + ")"
        + Optional(",")
    ).setParseAction(extract_tri)
    return (Literal("TIN") + "(" + OneOrMore(triangle) + ")").setParseAction(
        extract_tris
    )


parser = initialise_parser()


def arr_to_str(arr):
    return " ".join(str(item) for item in arr)


def read(filename):
    with open_file(filename) as f:
        return read_buffer(f)


def triangles_to_mesh(tris):
    point_idxs = OrderedDict()
    tri_idxs = []
    for points in tris:
        this_tri_idxs = []
        for point in points:
            try:
                idx = point_idxs[point]
            except KeyError:
                idx = len(point_idxs)
                point_idxs[point] = idx

            this_tri_idxs.append(idx)
        tri_idxs.append(this_tri_idxs)

    points = np.array(list(point_idxs), float)
    triangles = np.array(tri_idxs, int)

    return Mesh(points, {"triangle": triangles})


def read_buffer(f):
    if parser is None:
        raise ReadError("pyparsing not installed; WKT not available")

    triangles = parser.parseFile(f, True)
    return triangles_to_mesh(triangles)


def read_str(s):
    if parser is None:
        raise ReadError("pyparsing not installed; WKT not available")

    triangles = parser.parseString(s, True)
    return triangles_to_mesh(triangles)


def write(filename, mesh):
    with open_file(filename, "w") as f:
        write_buffer(f, mesh)


def write_buffer(f, mesh):
    try:
        tris = mesh.cells["triangle"]
    except KeyError:
        raise WriteError("WKT meshes can only have triangles")
    f.write("TIN (")

    joiner = ""
    for tri_points in mesh.points[tris]:
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
