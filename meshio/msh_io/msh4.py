# -*- coding: utf-8 -*-
#
"""
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.
"""
import logging
import struct

import numpy

from .common import (
    _read_physical_names,
    _read_periodic,
    _gmsh_to_meshio_type,
    _meshio_to_gmsh_type,
)
from ..mesh import Mesh
from ..common import raw_from_cell_data, num_nodes_per_cell, cell_data_from_raw


def _read_nodes(f, is_ascii, int_size, data_size):
    # first line: numEntityBlocks(unsigned long) numNodes(unsigned long)
    line = f.readline().decode("utf-8")
    num_entity_blocks, total_num_nodes = [int(k) for k in line.split()]

    points = numpy.empty((total_num_nodes, 3), dtype=float)
    tags = numpy.empty(total_num_nodes, dtype=int)

    idx = 0
    for k in range(num_entity_blocks):
        # first line in the entity block:
        # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
        line = f.readline().decode("utf-8")
        tag_entity, dim_entity, type_node, num_nodes = map(int, line.split())
        for i in range(num_nodes):
            # tag(int) x(double) y(double) z(double)
            line = f.readline().decode("utf-8")
            tag, x, y, z = line.split()
            points[idx] = [float(x), float(y), float(z)]
            tags[idx] = tag
            idx += 1

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndNodes"
    return points, tags


def _read_cells(f, point_tags, int_size, is_ascii):
    # numEntityBlocks(unsigned long) numElements(unsigned long)
    line = f.readline().decode("utf-8")
    num_entity_blocks, total_num_elements = [int(k) for k in line.split()]

    # invert tags array
    m = numpy.max(point_tags + 1)
    itags = -numpy.ones(m, dtype=int)
    for k, tag in enumerate(point_tags):
        itags[tag] = k

    data = []
    for k in range(num_entity_blocks):
        line = f.readline().decode("utf-8")
        # tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
        tag_entity, dim_entity, type_ele, num_elements = [int(k) for k in line.split()]
        tpe = _gmsh_to_meshio_type[type_ele]
        num_nodes_per_ele = num_nodes_per_cell[tpe]
        d = numpy.empty((num_elements, num_nodes_per_ele), dtype=int)
        idx = 0
        for i in range(num_elements):
            # tag(int) numVert[...](int)
            line = f.readline().decode("utf-8")
            items = line.split()
            assert len(items) == num_nodes_per_ele + 1
            d[idx] = [int(item) for item in items[1:]]
            idx += 1
        assert idx == num_elements
        data.append((tpe, itags[d]))

    cells = {}
    for item in data:
        key, values = item
        if key in cells:
            cells[key] = numpy.concatenate([cells[key], values])
        else:
            cells[key] = values

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndElements"

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]

    return cells


def _read_data(f, tag, data_dict, int_size, data_size, is_ascii):
    # Read string tags
    num_string_tags = int(f.readline().decode("utf-8"))
    string_tags = [
        f.readline().decode("utf-8").strip().replace('"', "")
        for _ in range(num_string_tags)
    ]
    # The real tags typically only contain one value, the time.
    # Discard it.
    num_real_tags = int(f.readline().decode("utf-8"))
    for _ in range(num_real_tags):
        f.readline()
    num_integer_tags = int(f.readline().decode("utf-8"))
    integer_tags = [int(f.readline().decode("utf-8")) for _ in range(num_integer_tags)]
    num_components = integer_tags[1]
    num_items = integer_tags[2]
    if is_ascii:
        data = numpy.fromfile(
            f, count=num_items * (1 + num_components), sep=" "
        ).reshape((num_items, 1 + num_components))
        # The first number is the index
        data = data[:, 1:]
    else:
        # binary
        assert numpy.int32(0).nbytes == int_size
        assert numpy.float64(0.0).nbytes == data_size
        dtype = [("index", numpy.int32), ("values", numpy.float64, (num_components,))]
        data = numpy.fromfile(f, count=num_items, dtype=dtype)
        assert (data["index"] == range(1, num_items + 1)).all()
        data = numpy.ascontiguousarray(data["values"])
        line = f.readline().decode("utf-8")
        assert line == "\n"

    line = f.readline().decode("utf-8")
    assert line.strip() == "$End{}".format(tag)

    # The gmsh format cannot distingiush between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data
    return


def read_buffer(f, is_ascii, int_size, data_size):
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
            points, point_tags = _read_nodes(f, is_ascii, int_size, data_size)
        elif environ == "Elements":
            cells = _read_cells(f, point_tags, int_size, is_ascii)
        elif environ == "Periodic":
            periodic = _read_periodic(f)
        elif environ == "NodeData":
            _read_data(f, "NodeData", point_data, int_size, data_size, is_ascii)
        elif environ == "ElementData":
            _read_data(f, "ElementData", cell_data_raw, int_size, data_size, is_ascii)
        else:
            # From
            # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-_0028version-4_0029>:
            # ```
            # Any section with an unrecognized header is simply ignored: you can thus
            # add comments in a .msh file by putting them e.g. inside a
            # $Comments/$EndComments section.
            # ```
            # skip environment
            while line != "$End" + environ:
                line = f.readline().decode("utf-8").strip()

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


def _write_physical_names(fh, field_data):
    # Write physical names
    entries = []
    for phys_name in field_data:
        try:
            phys_num, phys_dim = field_data[phys_name]
            phys_num, phys_dim = int(phys_num), int(phys_dim)
            entries.append((phys_dim, phys_num, phys_name))
        except (ValueError, TypeError):
            logging.warning("Field data contains entry that cannot be processed.")
    entries.sort()
    if entries:
        fh.write("$PhysicalNames\n".encode("utf-8"))
        fh.write("{}\n".format(len(entries)).encode("utf-8"))
        for entry in entries:
            fh.write('{} {} "{}"\n'.format(*entry).encode("utf-8"))
        fh.write("$EndPhysicalNames\n".encode("utf-8"))
    return


def _write_nodes(fh, points, write_binary):
    fh.write("$Nodes\n".encode("utf-8"))
    fh.write("{}\n".format(len(points)).encode("utf-8"))
    if write_binary:
        dtype = [("index", numpy.int32), ("x", numpy.float64, (3,))]
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
            fcd = numpy.empty((len(node_idcs), 0), dtype=numpy.int32)

        if write_binary:
            # header
            fh.write(struct.pack("i", _meshio_to_gmsh_type[cell_type]))
            fh.write(struct.pack("i", node_idcs.shape[0]))
            fh.write(struct.pack("i", fcd.shape[1]))
            # actual data
            a = numpy.arange(len(node_idcs), dtype=numpy.int32)[:, numpy.newaxis]
            a += 1 + consecutive_index
            array = numpy.hstack([a, fcd, node_idcs + 1])
            assert array.dtype == numpy.int32
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
    for dim, (stag, mtag), transform, slave_master in periodic:
        fh.write("{} {} {}\n".format(dim, stag, mtag).encode("utf-8"))
        if transform is not None:
            fh.write("{}\n".format(transform).encode("utf-8"))
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 0-based
        fh.write("{}\n".format(len(slave_master)).encode("utf-8"))
        for snode, mnode in slave_master:
            fh.write("{} {}\n".format(snode, mnode).encode("utf-8"))
    fh.write("$EndPeriodic\n".encode("utf-8"))


def _write_data(fh, tag, name, data, write_binary):
    fh.write("${}\n".format(tag).encode("utf-8"))
    # <http://gmsh.info/doc/texinfo/gmsh.html>:
    # > Number of string tags.
    # > gives the number of string tags that follow. By default the first
    # > string-tag is interpreted as the name of the post-processing view and
    # > the second as the name of the interpolation scheme. The interpolation
    # > scheme is provided in the $InterpolationScheme section (see below).
    fh.write("{}\n".format(1).encode("utf-8"))
    fh.write('"{}"\n'.format(name).encode("utf-8"))
    fh.write("{}\n".format(1).encode("utf-8"))
    fh.write("{}\n".format(0.0).encode("utf-8"))
    # three integer tags:
    fh.write("{}\n".format(3).encode("utf-8"))
    # time step
    fh.write("{}\n".format(0).encode("utf-8"))
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1
    assert num_components in [
        1,
        3,
        9,
    ], "Gmsh only permits 1, 3, or 9 components per data field."

    # Cut off the last dimension in case it's 1. This avoids problems with
    # writing the data.
    if len(data.shape) > 1 and data.shape[1] == 1:
        data = data[:, 0]

    fh.write("{}\n".format(num_components).encode("utf-8"))
    # num data items
    fh.write("{}\n".format(data.shape[0]).encode("utf-8"))
    # actually write the data
    if write_binary:
        dtype = [("index", numpy.int32), ("data", numpy.float64, num_components)]
        tmp = numpy.empty(len(data), dtype=dtype)
        tmp["index"] = 1 + numpy.arange(len(data))
        tmp["data"] = data
        fh.write(tmp.tostring())
        fh.write("\n".encode("utf-8"))
    else:
        fmt = " ".join(["{}"] + ["{!r}"] * num_components) + "\n"
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, x).encode("utf-8"))
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, *x).encode("utf-8"))

    fh.write("$End{}\n".format(tag).encode("utf-8"))
    return


def write(filename, mesh, write_binary=True):
    """Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    """
    if write_binary:
        for key, value in mesh.cells.items():
            if value.dtype != numpy.int32:
                logging.warning(
                    "Binary Gmsh needs 32-bit integers (got %s). Converting.",
                    value.dtype,
                )
                mesh.cells[key] = numpy.array(value, dtype=numpy.int32)

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
                    tag_data[cell_type][key] = data.astype(numpy.int32)
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
