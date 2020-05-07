"""
I/O for the TetGen file format, c.f.
<https://wias-berlin.de/software/tetgen/fformats.node.html>
"""
import logging
import pathlib

import numpy

from ..__about__ import __version__
from .._exceptions import ReadError, WriteError
from .._helpers import register
from .._mesh import CellBlock, Mesh


def read(filename):
    filename = pathlib.Path(filename)
    if filename.suffix == ".node":
        node_filename = filename
        ele_filename = filename.parent / (filename.stem + ".ele")
    elif filename.suffix == ".ele":
        node_filename = filename.parent / (filename.stem + ".node")
        ele_filename = filename
    else:
        raise ReadError()

    # read nodes
    # TODO remove as_posix
    with open(node_filename.as_posix()) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        num_points, dim, num_attrs, num_bmarkers = [
            int(item) for item in line.split(" ") if item != ""
        ]
        if dim != 3:
            raise ReadError("Need 3D points.")

        points = numpy.fromfile(
            f, dtype=float, count=(4 + num_attrs + num_bmarkers) * num_points, sep=" "
        ).reshape(num_points, 4 + num_attrs + num_bmarkers)

        node_index_base = int(points[0, 0])
        # make sure the nodes a numbered consecutively
        if not numpy.all(
            points[:, 0]
            == numpy.arange(node_index_base, node_index_base + points.shape[0])
        ):
            raise ReadError()
        # remove the leading index column, the attributes, and the boundary markers
        points = points[:, 1:4]

    # read elements
    with open(ele_filename.as_posix()) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        num_tets, num_points_per_tet, num_attrs = [
            int(item) for item in line.strip().split(" ") if item != ""
        ]
        if num_points_per_tet != 4:
            raise ReadError()
        cells = numpy.fromfile(
            f, dtype=int, count=(5 + num_attrs) * num_tets, sep=" "
        ).reshape(num_tets, 5 + num_attrs)
        # remove the leading index column and the attributes
        cells = cells[:, 1:5]
        cells -= node_index_base

    return Mesh(points, [CellBlock("tetra", cells)])


def write(filename, mesh, float_fmt=".16e"):
    filename = pathlib.Path(filename)
    if filename.suffix == ".node":
        node_filename = filename
        ele_filename = filename.parent / (filename.stem + ".ele")
    elif filename.suffix == ".ele":
        node_filename = filename.parent / (filename.stem + ".node")
        ele_filename = filename
    else:
        raise WriteError("Must specify .node or .ele file. Got {}.".format(filename))

    if mesh.points.shape[1] != 3:
        raise WriteError("Can only write 3D points")

    # write nodes
    # TODO remove .as_posix when requiring Python 3.6
    with open(node_filename.as_posix(), "w") as fh:
        fh.write("# This file was created by meshio v{}\n".format(__version__))
        fh.write("{} {} {} {}\n".format(mesh.points.shape[0], 3, 0, 0))
        fmt = "{} " + " ".join(3 * ["{:" + float_fmt + "}"]) + "\n"
        for k, pt in enumerate(mesh.points):
            fh.write(fmt.format(k, pt[0], pt[1], pt[2]))

    if not any(c.type == "tetra" for c in mesh.cells):
        raise WriteError("TegGen only supports tetrahedra")

    if any(c.type != "tetra" for c in mesh.cells):
        logging.warning(
            "TetGen only supports tetrahedra, but mesh has {}. Skipping those.".format(
                ", ".join([c.type for c in mesh.cells if c.type != "tetra"])
            )
        )

    # write cells
    # TODO remove .as_posix when requiring Python 3.6
    with open(ele_filename.as_posix(), "w") as fh:
        fh.write("# This file was created by meshio v{}\n".format(__version__))
        for cell_type, data in filter(lambda c: c.type == "tetra", mesh.cells):
            fh.write("{} {} {}\n".format(data.shape[0], 4, 0))
            for k, tet in enumerate(data):
                fh.write("{} {} {} {} {}\n".format(k, tet[0], tet[1], tet[2], tet[3]))


register("tetgen", [".ele", ".node"], read, {"tetgen": write})
