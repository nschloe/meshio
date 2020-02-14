"""
I/O for the Wavefront .obj file format, cf.
<https://en.wikipedia.org/wiki/Wavefront_.obj_file>.
"""
import datetime

import numpy

from ..__about__ import __version__
from .._exceptions import WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


def read(filename):
    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    points = []
    face_groups = []
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
            dat = [int(item.split("/")[0]) for item in split[1:]]
            if len(face_groups) == 0 or (
                len(face_groups[-1]) > 0 and len(face_groups[-1][-1]) != len(dat)
            ):
                face_groups.append([])
            face_groups[-1].append(dat)
        elif split[0] == "g":
            # new group
            face_groups.append([])
        else:
            # who knows
            pass

    # convert to numpy arrays
    face_groups = [numpy.array(f) for f in face_groups]
    cells = [("triangle" if f.shape[1] == 3 else "quad", f - 1) for f in face_groups]

    return Mesh(numpy.array(points), cells)


def write(filename, mesh):

    for c in mesh.cells:
        if c.type not in ["triangle", "quad"]:
            raise WriteError(
                "Wavefront .obj files can only contain triangle or quad cells."
            )

    with open_file(filename, "w") as f:
        f.write(
            "# Created by meshio v{}, {}\n".format(
                __version__, datetime.datetime.now().isoformat()
            )
        )
        for p in mesh.points:
            f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        for cell_type, cell_array in mesh.cells:
            fmt = "f {} {} {}"
            if cell_type == "quad":
                fmt += " {}"
            for c in cell_array:
                f.write("{}\n".format(fmt).format(*(c + 1)))


register("obj", [".obj"], read, {"obj": write})
