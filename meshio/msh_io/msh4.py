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
    _write_physical_names,
    _write_periodic,
)
from ..mesh import Mesh
from ..common import num_nodes_per_cell, cell_data_from_raw


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


def write(filename, mesh, write_binary=True):
    """Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    """
    # TODO respect binary writes
    write_binary = False

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
            ("$MeshFormat\n4 {} {}\n".format(mode_idx, size_of_double)).encode("utf-8")
        )
        if write_binary:
            fh.write(struct.pack("i", 1))
            fh.write("\n".encode("utf-8"))
        fh.write("$EndMeshFormat\n".encode("utf-8"))

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_nodes(fh, mesh.points, write_binary)
        _write_elements(fh, cells, write_binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic)

    return


def _write_nodes(fh, points, write_binary):
    # TODO respect write_binary
    fh.write("$Nodes\n".encode("utf-8"))
    # write all points as one big block

    # numEntityBlocks(unsigned long) numNodes(unsigned long)
    fh.write("{} {}\n".format(1, len(points)).encode("utf-8"))

    # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
    # TODO not sure what dimEntity is supposed to say
    fh.write("{} {} {} {}\n".format(1, 0, 0, len(points)).encode("utf-8"))

    for k, x in enumerate(points):
        # tag(int) x(double) y(double) z(double)
        fh.write("{} {!r} {!r} {!r}\n".format(k + 1, x[0], x[1], x[2]).encode("utf-8"))

    fh.write("$EndNodes\n".encode("utf-8"))
    return


def _write_elements(fh, cells, write_binary):
    # TODO respect write_binary
    # write elements
    fh.write("$Elements\n".encode("utf-8"))
    # count all cells
    total_num_cells = sum([data.shape[0] for _, data in cells.items()])
    fh.write("{} {}\n".format(len(cells), total_num_cells).encode("utf-8"))

    consecutive_index = 0
    for cell_type, node_idcs in cells.items():
        # tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
        fh.write(
            "{} {} {} {}\n".format(
                1,  # tag
                _geometric_dimension[cell_type],
                _meshio_to_gmsh_type[cell_type],
                node_idcs.shape[0],
            ).encode("utf-8")
        )
        form = " ".join(["{}"] * (num_nodes_per_cell[cell_type] + 1)) + "\n"

        # increment indices by one to conform with gmsh standard
        idcs = node_idcs + 1

        for idx in idcs:
            fh.write(form.format(consecutive_index, *idx).encode("utf-8"))
            consecutive_index += 1
        consecutive_index += len(node_idcs)

    fh.write("$EndElements\n".encode("utf-8"))
    return


_geometric_dimension = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
    "line3": 1,
    "triangle6": 2,
    "quad9": 2,
    "tetra10": 3,
    "hexahedron27": 3,
    "wedge18": 3,
    "pyramid14": 3,
    "vertex": 0,
    "quad8": 2,
    "hexahedron20": 3,
    "triangle10": 2,
    "triangle15": 2,
    "triangle21": 2,
    "line4": 1,
    "line5": 1,
    "line6": 1,
    "tetra20": 3,
    "tetra35": 3,
    "tetra56": 3,
    "quad16": 2,
    "quad25": 2,
    "quad36": 2,
    "triangle28": 2,
    "triangle36": 2,
    "triangle45": 2,
    "triangle55": 2,
    "triangle66": 2,
    "quad49": 2,
    "quad64": 2,
    "quad81": 2,
    "quad100": 2,
    "quad121": 2,
    "line7": 1,
    "line8": 1,
    "line9": 1,
    "line10": 1,
    "line11": 1,
    "tetra84": 3,
    "tetra120": 3,
    "tetra165": 3,
    "tetra220": 3,
    "tetra286": 3,
    "wedge40": 3,
    "wedge75": 3,
    "hexahedron64": 3,
    "hexahedron125": 3,
    "hexahedron216": 3,
    "hexahedron343": 3,
    "hexahedron512": 3,
    "hexahedron729": 3,
    "hexahedron1000": 3,
    "wedge126": 3,
    "wedge196": 3,
    "wedge288": 3,
    "wedge405": 3,
    "wedge550": 3,
}
