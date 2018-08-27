# -*- coding: utf-8 -*-
#
"""
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.
"""
import logging
import shlex
import struct

import numpy

from ..mesh import Mesh
from ..common import raw_from_cell_data, num_nodes_per_cell, cell_data_from_raw


# Translate meshio types to gmsh codes
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-version-2
_gmsh_to_meshio_type = {
    1: "line",
    2: "triangle",
    3: "quad",
    4: "tetra",
    5: "hexahedron",
    6: "wedge",
    7: "pyramid",
    8: "line3",
    9: "triangle6",
    10: "quad9",
    11: "tetra10",
    12: "hexahedron27",
    13: "wedge18",
    14: "pyramid14",
    15: "vertex",
    16: "quad8",
    17: "hexahedron20",
    21: "triangle10",
    23: "triangle15",
    25: "triangle21",
    26: "line4",
    27: "line5",
    28: "line6",
    29: "tetra20",
    30: "tetra35",
    31: "tetra56",
    36: "quad16",
    37: "quad25",
    38: "quad36",
    42: "triangle28",
    43: "triangle36",
    44: "triangle45",
    45: "triangle55",
    46: "triangle66",
    47: "quad49",
    48: "quad64",
    49: "quad81",
    50: "quad100",
    51: "quad121",
    62: "line7",
    63: "line8",
    64: "line9",
    65: "line10",
    66: "line11",
    71: "tetra84",
    72: "tetra120",
    73: "tetra165",
    74: "tetra220",
    75: "tetra286",
    90: "wedge40",
    91: "wedge75",
    92: "hexahedron64",
    93: "hexahedron125",
    94: "hexahedron216",
    95: "hexahedron343",
    96: "hexahedron512",
    97: "hexahedron729",
    98: "hexahedron1000",
    106: "wedge126",
    107: "wedge196",
    108: "wedge288",
    109: "wedge405",
    110: "wedge550",
}
_meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}


def _read_physical_names(f, field_data):
    line = f.readline().decode("utf-8")
    num_phys_names = int(line)
    for _ in range(num_phys_names):
        line = shlex.split(f.readline().decode("utf-8"))
        key = line[2]
        value = numpy.array(line[1::-1], dtype=int)
        field_data[key] = value
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPhysicalNames"
    return


def _read_nodes(f, is_ascii, int_size, data_size):
    # first line: numEntityBlocks(unsigned long) numNodes(unsigned long)
    line = f.readline().decode("utf-8")
    num_entity_blocks, total_num_nodes = [int(k) for k in line.split()]

    points = numpy.empty((total_num_nodes, 3), dtype=float)

    idx = 0
    for k in range(num_entity_blocks):
        # first line in the entity block:
        # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
        line = f.readline().decode("utf-8")
        tag_entity, dim_entity, type_node, num_nodes = [int(k) for k in line.split()]
        for i in range(num_nodes):
            # tag(int) x(double) y(double) z(double)
            line = f.readline().decode("utf-8")
            tag, x, y, z = line.split()
            points[idx] = [float(x), float(y), float(z)]
            idx += 1

    assert idx == total_num_nodes

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndNodes"
    return points


def _read_cells(f, int_size, is_ascii):
    # numEntityBlocks(unsigned long) numElements(unsigned long)
    line = f.readline().decode("utf-8")
    num_entity_blocks, total_num_elements = [int(k) for k in line.split()]

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
        data.append((tpe, d))

    cells = {}
    for item in data:
        key, values = item
        if key in cells:
            cells[key] = numpy.concatenate([cells[key], values])
        else:
            cells[key] = values

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndElements"

    # Subtract one to account for the fact that python indices are
    # 0-based.
    for key in cells:
        cells[key] -= 1

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]

    return cells


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


def _read_cells_binary(f, cells, cell_tags, total_num_cells, int_size):
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
        data = numpy.fromfile(f, count=count, dtype=numpy.int32).reshape(shape)

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
            transform = line
            num_nodes = int(f.readline().decode("utf-8"))
        else:
            transform = None
            num_nodes = int(line)
        slave_master = []
        for _ in range(num_nodes):
            line = f.readline().decode("utf-8")
            snode, mnode = [int(s) for s in line.split()]
            slave_master.append([snode, mnode])
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master -= 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), transform, slave_master])
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPeriodic"
    return periodic


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
            points = _read_nodes(f, is_ascii, int_size, data_size)
        elif environ == "Elements":
            cells = _read_cells(f, int_size, is_ascii)
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
