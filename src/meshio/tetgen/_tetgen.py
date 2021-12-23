"""
I/O for the TetGen file format, c.f.
<https://wias-berlin.de/software/tetgen/fformats.node.html>
"""
import pathlib

import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh


CELL_DIM = {
    '.ele': 4,
    '.face': 3,
    '.edge': 2,
}

CELL_NAME = {
    '.ele': 'tetra',
    '.face': 'triangle',
    '.edge': 'line',
}


def read(filename):
    filename = pathlib.Path(filename)
    if filename.suffix == ".node":
        node_filename = filename
        ele_filename = filename.parent / (filename.stem + ".ele")
    elif filename.suffix in (".ele", ".face", ".edge"):
        node_filename = filename.parent / (filename.stem + ".node")
        ele_filename = filename
    else:
        raise ReadError()

    point_data = {}
    cell_data = {}

    # read nodes
    with open(node_filename) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        num_points, dim, num_attrs, num_bmarkers = (
            int(item) for item in line.split(" ") if item != ""
        )
        if dim != 3:
            raise ReadError("Need 3D points.")

        points = np.fromfile(
            f, dtype=float, count=(4 + num_attrs + num_bmarkers) * num_points, sep=" "
        ).reshape(num_points, 4 + num_attrs + num_bmarkers)

        node_index_base = int(points[0, 0])
        # make sure the nodes are numbered consecutively
        if not np.all(
            points[:, 0]
            == np.arange(node_index_base, node_index_base + points.shape[0])
        ):
            raise ReadError()
        # read point attributes
        for k in range(num_attrs):
            point_data[f"tetgen:attr{k + 1}"] = points[:, 4 + k]
        # read boundary markers, the first is "ref", the others are "ref2", "ref3", ...
        for k in range(num_bmarkers):
            flag = "" if k == 0 else str(k + 1)
            point_data["tetgen:ref" + flag] = points[:, 4 + num_attrs + k]
        # remove the leading index column, the attributes, and the boundary markers
        points = points[:, 1:4]

    cell_dim = CELL_DIM[ele_filename.suffix]
    cell_name = CELL_NAME[ele_filename.suffix]

    # read elements
    with open(ele_filename.as_posix()) as f:
        line = f.readline().strip()
        while len(line) == 0 or line[0] == "#":
            line = f.readline().strip()

        if ele_filename.suffix == '.ele':
            num_cells, num_points_per_tet, num_attrs = (
                int(item) for item in line.strip().split(" ") if item != ""
            )
            if num_points_per_tet != cell_dim:
                raise ReadError()
        else:
            num_cells, num_attrs = (
                int(item) for item in line.strip().split(" ") if item != ""
            )

        cells = np.fromfile(
            f, dtype=int, count=(1 + cell_dim + num_attrs) * num_cells, sep=" "
        ).reshape(num_cells, 1 + cell_dim + num_attrs)
        # read cell (region) attributes, the first is "ref", the others are "ref2",
        # "ref3", ...
        for k in range(num_attrs):
            flag = "" if k == 0 else str(k + 1)
            cell_data["tetgen:ref" + flag] = [cells[:, 1 + cell_dim + k]]
        # remove the leading index column and the attributes
        cells = cells[:, 1:1 + cell_dim]
        cells -= node_index_base

    return Mesh(
        points, [CellBlock(cell_name, cells)], point_data=point_data, cell_data=cell_data
    )


def write(filename, mesh, float_fmt=".16e"):
    filename = pathlib.Path(filename)
    if filename.suffix == ".node":
        node_filename = filename
        ele_filename = filename.parent / (filename.stem + ".ele")
    elif filename.suffix in (".ele", ".face", ".edge"):
        node_filename = filename.parent / (filename.stem + ".node")
        ele_filename = filename
    else:
        raise WriteError(f"Must specify .node, .ele, .face, or .edge file. Got {filename}.")

    if mesh.points.shape[1] != 3:
        raise WriteError("Can only write 3D points")

    # write nodes
    with open(node_filename, "w") as fh:
        # identify ":ref" key
        attr_keys = list(mesh.point_data.keys())
        ref_keys = [k for k in attr_keys if ":ref" in k]
        if len(attr_keys) > 0:
            if len(ref_keys) > 0:
                ref_keys = ref_keys[:1]
                attr_keys.remove(ref_keys[0])
            else:
                ref_keys = attr_keys[:1]
                attr_keys = attr_keys[1:]

        nattr, nref = len(attr_keys), len(ref_keys)
        fh.write(f"# This file was created by meshio v{__version__}\n")
        if (nattr + nref) > 0:
            fh.write(
                "# attribute and marker names: {}\n".format(
                    ", ".join(attr_keys + ref_keys)
                )
            )
        fh.write(f"{mesh.points.shape[0]} {3} {nattr} {nref}\n")
        fmt = (
            "{} "
            + " ".join((3 + nattr) * ["{:" + float_fmt + "}"])
            + "".join((nref) * [" {}"])
            + "\n"
        )
        for k, pt in enumerate(mesh.points):
            data = (
                list(pt[:3])
                + [mesh.point_data[key][k] for key in attr_keys]
                + [mesh.point_data[key][k] for key in ref_keys]
            )
            fh.write(fmt.format(k, *data))

    cell_dim = CELL_DIM[ele_filename.suffix]
    cell_name = CELL_NAME[ele_filename.suffix]

    if any(c.type != cell_name for c in mesh.cells):
        string = ", ".join([c.type for c in mesh.cells if c.type != cell_name])
        warn(f"{filename} only supports {cell_name} cells, but mesh has {string}. Skipping those.")

    # write cells
    with open(ele_filename, "w") as fh:
        attr_keys = list(mesh.cell_data.keys())
        ref_keys = [k for k in attr_keys if ":ref" in k]
        if len(attr_keys) > 0:
            if len(ref_keys) > 0:
                attr_keys.remove(ref_keys[0])
                attr_keys = ref_keys[:1] + attr_keys

        nattr = len(attr_keys)
        fh.write(f"# This file was created by meshio v{__version__}\n")
        if nattr > 0:
            fh.write("# attribute names: {}\n".format(", ".join(attr_keys)))
        for id, c in enumerate(filter(lambda c: c.type == cell_name, mesh.cells)):
            data = c.data
            fh.write(f"{data.shape[0]} {cell_dim} {nattr}\n")
            fmt = " ".join((1 + cell_dim + nattr) * ["{}"]) + "\n"
            for k, cells in enumerate(data):
                data = list(cells[:cell_dim]) + [mesh.cell_data[key][id][k] for key in attr_keys]
                fh.write(fmt.format(k, *data))


register_format("tetgen", [".face", ".edge", ".ele", ".node"], read, {"tetgen": write})
