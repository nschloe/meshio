"""
I/O for the Wavefront .obj file format, cf.
<https://en.wikipedia.org/wiki/Wavefront_.obj_file>.
"""
import datetime

import numpy

from .__about__ import __version__
from ._mesh import Mesh


def read(filename):
    with open(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    points = []
    faces = []
    while True:
        line = f.readline()

        if not line:
            # EOF
            break

        strip = line.strip()

        if len(strip) == 0 or strip[0] == "#":
            continue

        split = strip.split()

        if split[0] == "v":
            # vertex
            points.append([numpy.float(item) for item in split[1:]])
        elif split[0] == "vn":
            # skip vertex normals
            pass
        elif split[0] == "s":
            # "s 1" or "s off" controls smooth shading
            pass
        elif split[0] == "f":
            faces.append([int(item.split("//")[0]) for item in split[1:]])
        else:
            # who knows
            pass

    triangle = numpy.array([f for f in faces if len(f) == 3])
    quad = numpy.array([f for f in faces if len(f) == 4])

    cells = {}
    if len(triangle) > 0:
        cells["triangle"] = triangle - 1
    if len(quad) > 0:
        cells["quad"] = quad - 1

    return Mesh(numpy.array(points), cells)


def write(filename, mesh):
    assert (
        "triangle" in mesh.cells or "quad" in mesh.cells
    ), "Wavefront .obj files can only contain triangle or quad cells."

    with open(filename, "w") as f:
        f.write(
            "# Created by meshio v{}, {}\n".format(
                __version__, datetime.datetime.now().isoformat()
            )
        )
        for p in mesh.points:
            f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        if "triangle" in mesh.cells:
            for c in mesh.cells["triangle"]:
                f.write("f {} {} {}\n".format(*(c + 1)))
        if "quad" in mesh.cells:
            for c in mesh.cells["quad"]:
                f.write("f {} {} {} {}\n".format(*(c + 1)))
    return
