"""
I/O for Gmsh's msh format (version 4.0, as used by Gmsh 4.1.5), cf.
<http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-_0028version-4_0029>.
"""
import logging
from functools import partial

import numpy

from .._common import (
    _topological_dimension,
    cell_data_from_raw,
    num_nodes_per_cell,
    raw_from_cell_data,
)
from .._exceptions import ReadError, WriteError
from .._mesh import CellBlock, Mesh
from .common import (
    _fast_forward_to_end_block,
    _gmsh_to_meshio_order,
    _gmsh_to_meshio_type,
    _meshio_to_gmsh_order,
    _meshio_to_gmsh_type,
    _read_data,
    _read_physical_names,
    _write_data,
    _write_physical_names,
)

c_int = numpy.dtype("i")
c_long = numpy.dtype("l")
c_ulong = numpy.dtype("L")
c_double = numpy.dtype("d")


def read_buffer(f, is_ascii, data_size):
    # Initialize the optional data fields
    points = []
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}
    physical_tags = None
    periodic = None
    while True:
        line = f.readline().decode("utf-8")
        if not line:
            # EOF
            break
        if line[0] != "$":
            raise ReadError
        environ = line[1:].strip()

        if environ == "PhysicalNames":
            _read_physical_names(f, field_data)
        elif environ == "Entities":
            physical_tags = _read_entities(f, is_ascii, data_size)
        elif environ == "Nodes":
            points, point_tags = _read_nodes(f, is_ascii, data_size)
        elif environ == "Elements":
            cells, cell_tags = _read_elements(
                f, point_tags, physical_tags, is_ascii, data_size
            )
        elif environ == "Periodic":
            periodic = _read_periodic(f, is_ascii)
        elif environ == "NodeData":
            _read_data(f, "NodeData", point_data, data_size, is_ascii)
        elif environ == "ElementData":
            _read_data(f, "ElementData", cell_data_raw, data_size, is_ascii)
        else:
            # From
            # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-_0028version-4_0029>:
            # ```
            # Any section with an unrecognized header is simply ignored: you can thus
            # add comments in a .msh file by putting them e.g. inside a
            # $Comments/$EndComments section.
            # ```
            # skip environment
            _fast_forward_to_end_block(f, environ)

    cell_data = cell_data_from_raw(cells, cell_data_raw)
    cell_data.update(cell_tags)

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        gmsh_periodic=periodic,
    )


def _read_entities(f, is_ascii, data_size):
    physical_tags = tuple({} for _ in range(4))  # dims 0, 1, 2, 3
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    number = fromfile(f, c_ulong, 4)  # dims 0, 1, 2, 3

    for d, n in enumerate(number):
        for _ in range(n):
            tag = int(fromfile(f, c_int, 1)[0])
            fromfile(f, c_double, 6)  # discard boxMinX...boxMaxZ
            num_physicals = int(fromfile(f, c_ulong, 1)[0])
            physical_tags[d][tag] = list(fromfile(f, c_int, num_physicals))
            if d > 0:  # discard tagBREP{Vert,Curve,Surfaces}
                num_BREP = int(fromfile(f, c_ulong, 1)[0])
                fromfile(f, c_int, num_BREP)
    _fast_forward_to_end_block(f, "Entities")
    return physical_tags


def _read_nodes(f, is_ascii, data_size):
    if is_ascii:
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
    else:
        # numEntityBlocks(unsigned long) numNodes(unsigned long)
        num_entity_blocks, _ = numpy.fromfile(f, count=2, dtype=c_ulong)

        points = []
        tags = []
        for _ in range(num_entity_blocks):
            # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
            numpy.fromfile(f, count=3, dtype=c_int)
            num_nodes = numpy.fromfile(f, count=1, dtype=c_ulong)[0]
            dtype = [("tag", c_int), ("x", c_double, (3,))]
            data = numpy.fromfile(f, count=num_nodes, dtype=dtype)
            tags.append(data["tag"])
            points.append(data["x"])

        tags = numpy.concatenate(tags)
        points = numpy.concatenate(points)

        line = f.readline().decode("utf-8")
        if line != "\n":
            raise ReadError()

    _fast_forward_to_end_block(f, "Nodes")
    return points, tags


def _read_elements(f, point_tags, physical_tags, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")

    # numEntityBlocks(unsigned long) numElements(unsigned long)
    num_entity_blocks, total_num_elements = fromfile(f, c_ulong, 2)

    data = []

    for k in range(num_entity_blocks):
        # tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
        tag_entity, dim_entity, type_ele = fromfile(f, c_int, 3)
        (num_ele,) = fromfile(f, c_ulong, 1)
        tpe = _gmsh_to_meshio_type[type_ele]
        num_nodes_per_ele = num_nodes_per_cell[tpe]
        d = fromfile(f, c_int, int(num_ele * (1 + num_nodes_per_ele))).reshape(
            (num_ele, -1)
        )
        if physical_tags is None:
            data.append((None, tag_entity, tpe, d))
        else:
            data.append((physical_tags[dim_entity][tag_entity], tag_entity, tpe, d))

    _fast_forward_to_end_block(f, "Elements")

    # The msh4 elements array refers to the nodes by their tag, not the index. All other
    # mesh formats use the index, which is far more efficient, too. Hence,
    # unfortunately, we have to do a fairly expensive conversion here.
    m = numpy.max(point_tags + 1)
    itags = -numpy.ones(m, dtype=int)
    itags[point_tags] = numpy.arange(len(point_tags))

    # Note that the first column in the data array is the element tag; discard it.
    data = [
        (physical_tag, geom_tag, tpe, itags[d[:, 1:]])
        for physical_tag, geom_tag, tpe, d in data
    ]

    cells = []
    cell_data = {}
    for physical_tag, geom_tag, key, values in data:
        cells.append((key, values))
        if physical_tag:
            if "gmsh:physical" not in cell_data:
                cell_data["gmsh:physical"] = []
            cell_data["gmsh:physical"].append(
                numpy.full(len(values), physical_tag[0], int)
            )
        if "gmsh:geometrical" not in cell_data:
            cell_data["gmsh:geometrical"] = []
        cell_data["gmsh:geometrical"].append(numpy.full(len(values), geom_tag, int))
    cells[:] = _gmsh_to_meshio_order(cells)

    return cells, cell_data


def _read_periodic(f, is_ascii):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    periodic = []
    num_periodic = int(fromfile(f, c_int, 1)[0])
    for _ in range(num_periodic):
        edim, stag, mtag = fromfile(f, c_int, 3)
        if is_ascii:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("Affine"):
                affine = line.replace("Affine", "", 1)
                affine = numpy.fromstring(affine, float, sep=" ")
                num_nodes = int(f.readline().decode("utf-8"))
            else:
                affine = None
                num_nodes = int(line)
        else:
            num_nodes = int(fromfile(f, c_long, 1)[0])
            if num_nodes < 0:
                affine = fromfile(f, c_double, 16)
                num_nodes = int(fromfile(f, c_ulong, 1)[0])
            else:
                affine = None
        slave_master = fromfile(f, c_int, num_nodes * 2).reshape(-1, 2)
        slave_master = slave_master - 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), affine, slave_master])

    _fast_forward_to_end_block(f, "Periodic")
    return periodic


def write(filename, mesh, float_fmt=".16e", binary=True):
    """Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    """
    if mesh.points.shape[1] == 2:
        logging.warning(
            "msh4 requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    if binary:
        for k, (key, value) in enumerate(mesh.cells):
            if value.dtype != c_int:
                logging.warning(
                    "Binary Gmsh needs c_int (typically numpy.int32) integers "
                    "(got %s). Converting.",
                    value.dtype,
                )
                mesh.cells[k] = CellBlock(key, numpy.array(value, dtype=c_int))

    cells = _meshio_to_gmsh_order(mesh.cells)

    with open(filename, "wb") as fh:
        mode_idx = 1 if binary else 0
        size_of_double = 8
        fh.write(f"$MeshFormat\n4.0 {mode_idx} {size_of_double}\n".encode("utf-8"))
        if binary:
            numpy.array([1], dtype=c_int).tofile(fh)
            fh.write(b"\n")
        fh.write(b"$EndMeshFormat\n")

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_nodes(fh, mesh.points, float_fmt, binary)
        _write_elements(fh, cells, binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic, float_fmt, binary)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodeData", name, dat, binary)
        cell_data_raw = raw_from_cell_data(mesh.cell_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, binary)


def _write_nodes(fh, points, float_fmt, binary):
    fh.write(b"$Nodes\n")

    # TODO not sure what dimEntity is supposed to say
    dim_entity = 0
    type_node = 0

    if binary:
        # write all points as one big block
        # numEntityBlocks(unsigned long) numNodes(unsigned long)
        # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
        # tag(int) x(double) y(double) z(double)
        numpy.array([1, points.shape[0]], dtype=c_ulong).tofile(fh)
        numpy.array([1, dim_entity, type_node], dtype=c_int).tofile(fh)
        numpy.array([points.shape[0]], dtype=c_ulong).tofile(fh)
        dtype = [("index", c_int), ("x", c_double, (3,))]
        tmp = numpy.empty(len(points), dtype=dtype)
        tmp["index"] = 1 + numpy.arange(len(points))
        tmp["x"] = points
        tmp.tofile(fh)
        fh.write(b"\n")
    else:
        # write all points as one big block
        # numEntityBlocks(unsigned long) numNodes(unsigned long)
        fh.write("{} {}\n".format(1, len(points)).encode("utf-8"))

        # tagEntity(int) dimEntity(int) typeNode(int) numNodes(unsigned long)
        fh.write(
            "{} {} {} {}\n".format(1, dim_entity, type_node, len(points)).encode(
                "utf-8"
            )
        )

        fmt = "{} " + " ".join(3 * ["{:" + float_fmt + "}"]) + "\n"
        for k, x in enumerate(points):
            # tag(int) x(double) y(double) z(double)
            fh.write(fmt.format(k + 1, x[0], x[1], x[2]).encode("utf-8"))

    fh.write(b"$EndNodes\n")


def _write_elements(fh, cells, binary):
    # TODO respect binary
    # write elements
    fh.write(b"$Elements\n")

    if binary:
        total_num_cells = sum([data.shape[0] for _, data in cells])
        numpy.array([len(cells), total_num_cells], dtype=c_ulong).tofile(fh)

        consecutive_index = 0
        for cell_type, node_idcs in cells:
            # tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
            numpy.array(
                [1, _topological_dimension[cell_type], _meshio_to_gmsh_type[cell_type]],
                dtype=c_int,
            ).tofile(fh)
            numpy.array([node_idcs.shape[0]], dtype=c_ulong).tofile(fh)

            if node_idcs.dtype != c_int:
                raise WriteError()
            data = numpy.column_stack(
                [
                    numpy.arange(
                        consecutive_index,
                        consecutive_index + len(node_idcs),
                        dtype=c_int,
                    ),
                    # increment indices by one to conform with gmsh standard
                    node_idcs + 1,
                ]
            )
            data.tofile(fh)
            consecutive_index += len(node_idcs)

        fh.write(b"\n")
    else:
        # count all cells
        total_num_cells = sum([data.shape[0] for _, data in cells])
        fh.write("{} {}\n".format(len(cells), total_num_cells).encode("utf-8"))

        consecutive_index = 0
        for cell_type, node_idcs in cells:
            # tagEntity(int) dimEntity(int) typeEle(int) numElements(unsigned long)
            fh.write(
                "{} {} {} {}\n".format(
                    1,  # tag
                    _topological_dimension[cell_type],
                    _meshio_to_gmsh_type[cell_type],
                    node_idcs.shape[0],
                ).encode("utf-8")
            )
            # increment indices by one to conform with gmsh standard
            idcs = node_idcs + 1

            fmt = " ".join(["{}"] * (num_nodes_per_cell[cell_type] + 1)) + "\n"
            for idx in idcs:
                fh.write(fmt.format(consecutive_index, *idx).encode("utf-8"))
                consecutive_index += 1

    fh.write(b"$EndElements\n")


def _write_periodic(fh, periodic, float_fmt, binary):
    def tofile(fh, value, dtype, **kwargs):
        ary = numpy.array(value, dtype=dtype)
        if binary:
            ary.tofile(fh)
        else:
            ary = numpy.atleast_2d(ary)
            fmt = float_fmt if dtype == c_double else "d"
            fmt = "%" + kwargs.pop("fmt", fmt)
            numpy.savetxt(fh, ary, fmt=fmt, **kwargs)

    fh.write(b"$Periodic\n")
    tofile(fh, len(periodic), c_int)
    for dim, (stag, mtag), affine, slave_master in periodic:
        tofile(fh, [dim, stag, mtag], c_int)
        if affine is not None and len(affine) > 0:
            tofile(fh, -1, c_long)
            tofile(fh, affine, c_double, fmt=float_fmt)
        slave_master = numpy.array(slave_master, dtype=c_int)
        slave_master = slave_master.reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 1-based
        tofile(fh, len(slave_master), c_int)
        tofile(fh, slave_master, c_int)
    if binary:
        fh.write(b"\n")
    fh.write(b"$EndPeriodic\n")
