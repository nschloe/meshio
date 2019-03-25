# -*- coding: utf-8 -*-
#
"""
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.
"""
import logging
import struct

import numpy

from ..mesh import Mesh
from .common import (
    num_nodes_per_cell,
    cell_data_from_raw,
    raw_from_cell_data,
    _gmsh_to_meshio_type,
    _meshio_to_gmsh_type,
    _read_physical_names,
    _write_physical_names,
    _read_data,
    _write_data,
)

c_int = numpy.dtype("i")
c_double = numpy.dtype("d")


def read_buffer(f, is_ascii, data_size):
    # The format is specified at
    # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the optional data fields
    points = []
    cells = {}
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}
    periodic = None
    while True:
        line = f.readline().decode("utf-8")
        if not line:
            # EOF
            break
        assert line[0] == "$"
        environ = line[1:].strip()

        if environ == "PhysicalNames":
            _read_physical_names(f, field_data)
        elif environ == "Nodes":
            points = _read_nodes(f, is_ascii, data_size)
        elif environ == "Elements":
            has_additional_tag_data, cell_tags = _read_cells(f, cells, is_ascii)
        elif environ == "Periodic":
            periodic = _read_periodic(f)
        elif environ == "NodeData":
            _read_data(f, "NodeData", point_data, data_size, is_ascii)
        elif environ == "ElementData":
            _read_data(f, "ElementData", cell_data_raw, data_size, is_ascii)
        else:
            # skip environment
            while line != "$End" + environ:
                line = f.readline().decode("utf-8").strip()

    if has_additional_tag_data:
        logging.warning("The file contains tag data that couldn't be processed.")

    cell_data = cell_data_from_raw(cells, cell_data_raw)

    # merge cell_tags into cell_data
    for key, tag_dict in cell_tags.items():
        if key not in cell_data:
            cell_data[key] = {}
        for name, item_list in tag_dict.items():
            assert name not in cell_data[key]
            cell_data[key][name] = item_list

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        gmsh_periodic=periodic,
    )


def _read_nodes(f, is_ascii, data_size):
    # The first line is the number of nodes
    line = f.readline().decode("utf-8")
    num_nodes = int(line)
    if is_ascii:
        points = numpy.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
        # The first number is the index
        points = points[:, 1:]
    else:
        # binary
        dtype = [("index", c_int), ("x", c_double, (3,))]
        data = numpy.fromfile(f, count=num_nodes, dtype=dtype)
        assert (data["index"] == range(1, num_nodes + 1)).all()
        points = numpy.ascontiguousarray(data["x"])
        line = f.readline().decode("utf-8")
        assert line == "\n"

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndNodes"
    return points


def _read_cells(f, cells, is_ascii):
    # The first line is the number of elements
    line = f.readline().decode("utf-8")
    total_num_cells = int(line)
    has_additional_tag_data = False
    cell_tags = {}
    if is_ascii:
        _read_cells_ascii(f, cells, cell_tags, total_num_cells)
    else:
        _read_cells_binary(f, cells, cell_tags, total_num_cells)

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndElements"

    # Subtract one to account for the fact that python indices are
    # 0-based.
    for key in cells:
        cells[key] -= 1

    # restrict to the standard two data items (physical, geometrical)
    output_cell_tags = {}
    for key in cell_tags:
        output_cell_tags[key] = {"gmsh:physical": [], "gmsh:geometrical": []}
        for item in cell_tags[key]:
            if len(item) > 0:
                output_cell_tags[key]["gmsh:physical"].append(item[0])
            if len(item) > 1:
                output_cell_tags[key]["gmsh:geometrical"].append(item[1])
            if len(item) > 2:
                has_additional_tag_data = True
        output_cell_tags[key]["gmsh:physical"] = numpy.array(
            output_cell_tags[key]["gmsh:physical"], dtype=int
        )
        output_cell_tags[key]["gmsh:geometrical"] = numpy.array(
            output_cell_tags[key]["gmsh:geometrical"], dtype=int
        )

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]

    return has_additional_tag_data, output_cell_tags


def _read_cells_ascii(f, cells, cell_tags, total_num_cells):
    for _ in range(total_num_cells):
        line = f.readline().decode("utf-8")
        data = [int(k) for k in filter(None, line.split())]
        t = _gmsh_to_meshio_type[data[1]]
        num_nodes_per_elem = num_nodes_per_cell[t]

        if t not in cells:
            cells[t] = []
        cells[t].append(data[-num_nodes_per_elem:])

        # data[2] gives the number of tags. The gmsh manual
        # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>
        # says:
        # >>>
        # By default, the first tag is the number of the physical entity to
        # which the element belongs; the second is the number of the
        # elementary geometrical entity to which the element belongs; the
        # third is the number of mesh partitions to which the element
        # belongs, followed by the partition ids (negative partition ids
        # indicate ghost cells). A zero tag is equivalent to no tag. Gmsh
        # and most codes using the MSH 2 format require at least the first
        # two tags (physical and elementary tags).
        # <<<
        num_tags = data[2]
        if num_tags > 0:
            if t not in cell_tags:
                cell_tags[t] = []
            cell_tags[t].append(data[3 : 3 + num_tags])

    # convert to numpy arrays
    for key in cells:
        cells[key] = numpy.array(cells[key], dtype=int)
    # Cannot convert cell_tags[key] to numpy array: There may be a
    # different number of tags for each cell.

    return


def _read_cells_binary(f, cells, cell_tags, total_num_cells):
    int_size = struct.calcsize("i")
    num_elems = 0
    while num_elems < total_num_cells:
        # read element header
        elem_type = struct.unpack("i", f.read(int_size))[0]
        t = _gmsh_to_meshio_type[elem_type]
        num_nodes_per_elem = num_nodes_per_cell[t]
        num_elems0 = struct.unpack("i", f.read(int_size))[0]
        num_tags = struct.unpack("i", f.read(int_size))[0]
        # assert num_tags >= 2

        # read element data
        shape = (num_elems0, 1 + num_tags + num_nodes_per_elem)
        count = shape[0] * shape[1]
        data = numpy.fromfile(f, count=count, dtype=c_int).reshape(shape)

        if t not in cells:
            cells[t] = []
        cells[t].append(data[:, -num_nodes_per_elem:])

        if t not in cell_tags:
            cell_tags[t] = []
        cell_tags[t].append(data[:, 1 : num_tags + 1])

        num_elems += num_elems0

    # collect cells
    for key in cells:
        cells[key] = numpy.vstack(cells[key])

    # collect cell tags
    for key in cell_tags:
        cell_tags[key] = numpy.vstack(cell_tags[key])

    line = f.readline().decode("utf-8")
    assert line == "\n"
    return


def _read_periodic(f):
    periodic = []
    num_periodic = int(f.readline().decode("utf-8"))
    for _ in range(num_periodic):
        line = f.readline().decode("utf-8")
        edim, stag, mtag = [int(s) for s in line.split()]
        line = f.readline().decode("utf-8").strip()
        if line.startswith("Affine"):
            affine = line.replace("Affine", "", 1)
            affine = numpy.fromstring(affine, float, sep=" ")
            num_nodes = int(f.readline().decode("utf-8"))
        else:
            affine = None
            num_nodes = int(line)
        slave_master = []
        for _ in range(num_nodes):
            line = f.readline().decode("utf-8")
            snode, mnode = [int(s) for s in line.split()]
            slave_master.append([snode, mnode])
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master -= 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), affine, slave_master])
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPeriodic"
    return periodic


def write(filename, mesh, write_binary=True):
    """Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    """
    if mesh.points.shape[1] == 2:
        logging.warning(
            "msh2 requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    if write_binary:
        for key, value in mesh.cells.items():
            if value.dtype != c_int:
                logging.warning(
                    "Binary Gmsh needs 32-bit integers (got %s). Converting.",
                    value.dtype,
                )
                mesh.cells[key] = numpy.array(value, dtype=c_int)

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    cells = mesh.cells.copy()
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]
        ]

    with open(filename, "wb") as fh:
        mode_idx = 1 if write_binary else 0
        size_of_double = 8
        fh.write(
            ("$MeshFormat\n2.2 {} {}\n".format(mode_idx, size_of_double)).encode(
                "utf-8"
            )
        )
        if write_binary:
            fh.write(struct.pack("i", 1))
            fh.write("\n".encode("utf-8"))
        fh.write("$EndMeshFormat\n".encode("utf-8"))

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        # Split the cell data: gmsh:physical and gmsh:geometrical are tags, the
        # rest is actual cell data.
        tag_data = {}
        other_data = {}
        for cell_type, a in mesh.cell_data.items():
            tag_data[cell_type] = {}
            other_data[cell_type] = {}
            for key, data in a.items():
                if key in ["gmsh:physical", "gmsh:geometrical"]:
                    tag_data[cell_type][key] = data.astype(c_int)
                else:
                    other_data[cell_type][key] = data

        _write_nodes(fh, mesh.points, write_binary)
        _write_elements(fh, cells, tag_data, write_binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodeData", name, dat, write_binary)
        cell_data_raw = raw_from_cell_data(other_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, write_binary)

    return


def _write_nodes(fh, points, write_binary):
    fh.write("$Nodes\n".encode("utf-8"))
    fh.write("{}\n".format(len(points)).encode("utf-8"))
    if write_binary:
        dtype = [("index", c_int), ("x", c_double, (3,))]
        tmp = numpy.empty(len(points), dtype=dtype)
        tmp["index"] = 1 + numpy.arange(len(points))
        tmp["x"] = points
        fh.write(tmp.tostring())
        fh.write("\n".encode("utf-8"))
    else:
        for k, x in enumerate(points):
            fh.write(
                "{} {!r} {!r} {!r}\n".format(k + 1, x[0], x[1], x[2]).encode("utf-8")
            )
    fh.write("$EndNodes\n".encode("utf-8"))
    return


def _write_elements(fh, cells, tag_data, write_binary):
    # write elements
    fh.write("$Elements\n".encode("utf-8"))
    # count all cells
    total_num_cells = sum([data.shape[0] for _, data in cells.items()])
    fh.write("{}\n".format(total_num_cells).encode("utf-8"))

    consecutive_index = 0
    for cell_type, node_idcs in cells.items():
        tags = []
        for key in ["gmsh:physical", "gmsh:geometrical"]:
            try:
                tags.append(tag_data[cell_type][key])
            except KeyError:
                pass
        fcd = numpy.concatenate([tags]).T

        if len(fcd) == 0:
            fcd = numpy.empty((len(node_idcs), 0), dtype=c_int)

        if write_binary:
            # header
            fh.write(struct.pack("i", _meshio_to_gmsh_type[cell_type]))
            fh.write(struct.pack("i", node_idcs.shape[0]))
            fh.write(struct.pack("i", fcd.shape[1]))
            # actual data
            a = numpy.arange(len(node_idcs), dtype=c_int)[:, numpy.newaxis]
            a += 1 + consecutive_index
            array = numpy.hstack([a, fcd, node_idcs + 1])
            assert array.dtype == c_int
            fh.write(array.tostring())
        else:
            form = (
                "{} "
                + str(_meshio_to_gmsh_type[cell_type])
                + " "
                + str(fcd.shape[1])
                + " {} {}\n"
            )
            for k, c in enumerate(node_idcs):
                fh.write(
                    form.format(
                        consecutive_index + k + 1,
                        " ".join([str(val) for val in fcd[k]]),
                        " ".join([str(cc + 1) for cc in c]),
                    ).encode("utf-8")
                )

        consecutive_index += len(node_idcs)
    if write_binary:
        fh.write("\n".encode("utf-8"))
    fh.write("$EndElements\n".encode("utf-8"))
    return


def _write_periodic(fh, periodic):
    fh.write("$Periodic\n".encode("utf-8"))
    fh.write("{}\n".format(len(periodic)).encode("utf-8"))
    for dim, (stag, mtag), affine, slave_master in periodic:
        fh.write("{} {} {}\n".format(dim, stag, mtag).encode("utf-8"))
        if affine is not None:
            fh.write("Affine ".encode("utf-8"))
            affine = numpy.array(affine, dtype=float)
            affine = numpy.atleast_2d(affine.ravel())
            numpy.savetxt(fh, affine, "%.16g")
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 0-based
        fh.write("{}\n".format(len(slave_master)).encode("utf-8"))
        for snode, mnode in slave_master:
            fh.write("{} {}\n".format(snode, mnode).encode("utf-8"))
    fh.write("$EndPeriodic\n".encode("utf-8"))
    return
