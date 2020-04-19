"""
I/O for the Wavefront .obj file format, cf.
<https://en.wikipedia.org/wiki/Wavefront_.obj_file>.
"""
import datetime
import logging

import numpy

from ..__about__ import __version__
from .._exceptions import WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import CellBlock, Mesh


def read(filename):
    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    points = []
    vertex_normals = []
    texture_coords = []
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
            vertex_normals.append([numpy.float(item) for item in split[1:]])
        elif split[0] == "vt":
            texture_coords.append([numpy.float(item) for item in split[1:]])
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

    # There may be empty groups, too. <https://github.com/nschloe/meshio/issues/770>
    # Remove them.
    face_groups = [f for f in face_groups if len(f) > 0]

    points = numpy.array(points)
    texture_coords = numpy.array(texture_coords)
    vertex_normals = numpy.array(vertex_normals)
    point_data = {}
    if len(texture_coords) > 0:
        point_data["obj:vt"] = texture_coords
    if len(vertex_normals) > 0:
        point_data["obj:vn"] = vertex_normals

    # convert to numpy arrays
    face_groups = [numpy.array(f) for f in face_groups]
    cells = []
    for f in face_groups:
        if f.shape[1] == 3:
            cells.append(CellBlock("triangle", f - 1))
        elif f.shape[1] == 4:
            cells.append(CellBlock("quad", f - 1))
        else:
            # Anything else but triangles or quads not supported yet
            logging.warning(
                "meshio::obj only supports triangles and quads. "
                "Skipping {} polygons with {} nodes".format(f.shape[0], f.shape[1])
            )

    return Mesh(points, cells, point_data=point_data)


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

        if "obj:vn" in mesh.point_data:
            dat = mesh.point_data["obj:vn"]
            fmt = "vn " + " ".join(["{}"] * dat.shape[1]) + "\n"
            for vn in dat:
                f.write(fmt.format(*vn))

        if "obj:vt" in mesh.point_data:
            dat = mesh.point_data["obj:vt"]
            fmt = "vt " + " ".join(["{}"] * dat.shape[1]) + "\n"
            for vt in dat:
                f.write(fmt.format(*vt))

        for cell_type, cell_array in mesh.cells:
            fmt = "f " + " ".join(["{}"] * cell_array.shape[1]) + "\n"
            for c in cell_array:
                f.write(fmt.format(*(c + 1)))


register("obj", [".obj"], read, {"obj": write})
