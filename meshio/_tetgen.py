"""
I/O for the TetGen file format, c.f.
<https://wias-berlin.de/software/tetgen/fformats.node.html>
"""
import logging
import os

import numpy

from .__about__ import __version__
from ._mesh import Mesh


def read(filename):
    base, ext = os.path.splitext(filename)
    if ext == ".node":
        node_filename = filename
        ele_filename = base + ".ele"
    else:
        assert ext == ".ele"
        node_filename = base + ".node"
        ele_filename = filename

    # read nodes
    with open(node_filename) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        num_points, dim, num_attrs, num_bmarkers = [
            int(item) for item in line.split(" ") if item != ""
        ]
        assert dim == 3

        points = numpy.fromfile(
            f, dtype=float, count=(4 + num_attrs + num_bmarkers) * num_points, sep=" "
        ).reshape(num_points, 4 + num_attrs + num_bmarkers)

        node_index_base = int(points[0, 0])
        # make sure the nodes a numbered consecutively
        assert numpy.all(
            points[:, 0]
            == numpy.arange(node_index_base, node_index_base + points.shape[0])
        )
        # remove the leading index column, the attributes, and the boundary markers
        points = points[:, 1:4]

    # read elements
    with open(ele_filename) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        num_tets, num_points_per_tet, num_attrs = [
            int(item) for item in line.strip().split(" ") if item != ""
        ]
        assert num_points_per_tet == 4
        cells = numpy.fromfile(
            f, dtype=int, count=(5 + num_attrs) * num_tets, sep=" "
        ).reshape(num_tets, 5 + num_attrs)
        # remove the leading index column and the attributes
        cells = cells[:, 1:5]
        cells -= node_index_base

    return Mesh(points, {"tetra": cells})


def write(filename, mesh):

    base, ext = os.path.splitext(filename)
    if ext == ".node":
        node_filename = filename
        ele_filename = base + ".ele"
    else:
        assert ext == ".ele"
        node_filename = base + ".node"
        ele_filename = filename

    assert mesh.points.shape[1] == 3

    # write nodes
    with open(node_filename, "w") as fh:
        fh.write("# This file was created by meshio v{}\n".format(__version__))
        fh.write("{} {} {} {}\n".format(mesh.points.shape[0], 3, 0, 0))
        for k, pt in enumerate(mesh.points):
            fh.write("{} {:.15e} {:.15e} {:.15e}\n".format(k, pt[0], pt[1], pt[2]))

    assert "tetra" in mesh.cells, "TegGen only supports tetrahedra"
    if len(mesh.cells) > 1:
        logging.warning(
            "TetGen only supports tetrahedra, but mesh has {}. Skipping those.".format(
                ", ".join([key for key in mesh.keys() if key != "tetra"])
            )
        )

    # write cells
    with open(ele_filename, "w") as fh:
        fh.write("# This file was created by meshio v{}\n".format(__version__))
        fh.write("{} {} {}\n".format(mesh.cells["tetra"].shape[0], 4, 0))
        for k, tet in enumerate(mesh.cells["tetra"]):
            fh.write("{} {} {} {} {}\n".format(k, tet[0], tet[1], tet[2], tet[3]))

    return
