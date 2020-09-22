"""
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.
"""
import logging

import numpy

from .._common import cell_data_from_raw, num_nodes_per_cell, raw_from_cell_data
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
c_double = numpy.dtype("d")


def read_buffer(f, is_ascii, data_size):
    # The format is specified at
    # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the optional data fields
    points = []
    cells = []
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
        if line[0] != "$":
            raise ReadError()
        environ = line[1:].strip()

        if environ == "PhysicalNames":
            _read_physical_names(f, field_data)
        elif environ == "Nodes":
            points, point_tags = _read_nodes(f, is_ascii, data_size)
        elif environ == "Elements":
            has_additional_tag_data, cell_tags = _read_cells(
                f, cells, point_tags, is_ascii
            )
        elif environ == "Periodic":
            periodic = _read_periodic(f)
        elif environ == "NodeData":
            _read_data(f, "NodeData", point_data, data_size, is_ascii)
        elif environ == "ElementData":
            _read_data(f, "ElementData", cell_data_raw, data_size, is_ascii)
        else:
            _fast_forward_to_end_block(f, environ)

    if has_additional_tag_data:
        logging.warning("The file contains tag data that couldn't be processed.")

    cell_data = cell_data_from_raw(cells, cell_data_raw)

    # merge cell_tags into cell_data
    for tag_name, tag_dict in cell_tags.items():
        if tag_name not in cell_data:
            cell_data[tag_name] = []
        offset = {}
        for cell_type, cell_array in cells:
            start = offset.setdefault(cell_type, 0)
            end = start + len(cell_array)
            offset[cell_type] = end
            tags = tag_dict.get(cell_type, [])
            tags = numpy.array(tags[start:end], dtype=c_int)
            cell_data[tag_name].append(tags)

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
        point_tags = points[:, 0]
        points = points[:, 1:]
    else:
        # binary
        dtype = [("index", c_int), ("x", c_double, (3,))]
        data = numpy.fromfile(f, count=num_nodes, dtype=dtype)
        if not (data["index"] == range(1, num_nodes + 1)).all():
            raise ReadError()
        points = numpy.ascontiguousarray(data["x"])
        point_tags = data["index"]

    _fast_forward_to_end_block(f, "Nodes")
    return points, point_tags


def _read_cells(f, cells, point_tags, is_ascii):
    # The first line is the number of elements
    line = f.readline().decode("utf-8")
    total_num_cells = int(line)
    has_additional_tag_data = False
    cell_tags = {}
    if is_ascii:
        _read_cells_ascii(f, cells, cell_tags, total_num_cells)
    else:
        _read_cells_binary(f, cells, cell_tags, total_num_cells)
    cells[:] = _gmsh_to_meshio_order(cells)

    point_tags = numpy.asarray(point_tags, dtype=numpy.int32) - 1
    remap = -numpy.ones((numpy.max(point_tags) + 1,), dtype=numpy.int32)
    remap[point_tags] = numpy.arange(point_tags.shape[0])

    for ic, (ct, cd) in enumerate(cells):
        cells[ic] = (ct, remap[cd])

    _fast_forward_to_end_block(f, "Elements")

    # restrict to the standard two data items (physical, geometrical)
    output_cell_tags = {}
    for cell_type in cell_tags:
        physical = []
        geometrical = []
        for item in cell_tags[cell_type]:
            if len(item) > 0:
                physical.append(item[0])
            if len(item) > 1:
                geometrical.append(item[1])
            if len(item) > 2:
                has_additional_tag_data = True
        physical = numpy.array(physical, dtype=c_int)
        geometrical = numpy.array(geometrical, dtype=c_int)
        if len(physical) > 0:
            if "gmsh:physical" not in output_cell_tags:
                output_cell_tags["gmsh:physical"] = {}
            output_cell_tags["gmsh:physical"][cell_type] = physical
        if len(geometrical) > 0:
            if "gmsh:geometrical" not in output_cell_tags:
                output_cell_tags["gmsh:geometrical"] = {}
            output_cell_tags["gmsh:geometrical"][cell_type] = geometrical

    return has_additional_tag_data, output_cell_tags


def _read_cells_ascii(f, cells, cell_tags, total_num_cells):
    for _ in range(total_num_cells):
        line = f.readline().decode("utf-8")
        data = [int(k) for k in filter(None, line.split())]
        t = _gmsh_to_meshio_type[data[1]]
        num_nodes_per_elem = num_nodes_per_cell[t]

        if len(cells) == 0 or t != cells[-1][0]:
            cells.append((t, []))
        cells[-1][1].append(data[-num_nodes_per_elem:])

        # data[2] gives the number of tags. The gmsh manual
        # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>
        # says:
        # >>>
        # By default, the first tag is the number of the physical entity to which the
        # element belongs; the second is the number of the elementary geometrical entity
        # to which the element belongs; the third is the number of mesh partitions to
        # which the element belongs, followed by the partition ids (negative partition
        # ids indicate ghost cells). A zero tag is equivalent to no tag. Gmsh and most
        # codes using the MSH 2 format require at least the first two tags (physical and
        # elementary tags).
        # <<<
        num_tags = data[2]
        if t not in cell_tags:
            cell_tags[t] = []
        cell_tags[t].append(data[3 : 3 + num_tags])

    # convert to numpy arrays
    # Subtract one to account for the fact that python indices are 0-based.
    for k, c in enumerate(cells):
        cells[k] = (c[0], numpy.array(c[1], dtype=c_int) - 1)
    # Cannot convert cell_tags[key] to numpy array: There may be a different number of
    # tags for each cell.


def _read_cells_binary(f, cells, cell_tags, total_num_cells):
    num_elems = 0
    while num_elems < total_num_cells:
        # read element header
        elem_type, num_elems0, num_tags = numpy.fromfile(f, count=3, dtype=c_int)
        t = _gmsh_to_meshio_type[elem_type]
        num_nodes_per_elem = num_nodes_per_cell[t]

        # read element data
        shape = (num_elems0, 1 + num_tags + num_nodes_per_elem)
        count = shape[0] * shape[1]
        data = numpy.fromfile(f, count=count, dtype=c_int).reshape(shape)

        if len(cells) == 0 or t != cells[-1][0]:
            cells.append((t, []))
        cells[-1][1].append(data[:, -num_nodes_per_elem:])

        if t not in cell_tags:
            cell_tags[t] = []
        cell_tags[t].append(data[:, 1 : num_tags + 1])

        num_elems += num_elems0

    # collect cells
    for k, c in enumerate(cells):
        cells[k] = (c[0], numpy.vstack(c[1]) - 1)

    # collect cell tags
    for key in cell_tags:
        cell_tags[key] = numpy.vstack(cell_tags[key])


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
        slave_master = numpy.array(slave_master, dtype=c_int).reshape(-1, 2)
        slave_master -= 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), affine, slave_master])
    _fast_forward_to_end_block(f, "Periodic")
    return periodic


def write(filename, mesh, float_fmt=".16e", binary=True):
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

    # Split the cell data: gmsh:physical and gmsh:geometrical are tags, the rest is
    # actual cell data.
    tag_data = {}
    other_data = {}
    for key, d in mesh.cell_data.items():
        if key in ["gmsh:physical", "gmsh:geometrical", "cell_tags"]:
            tag_data[key] = d
        else:
            other_data[key] = d

    # Always include the physical and geometrical tags. See also the quoted excerpt from
    # the gmsh documentation in the _read_cells_ascii function above.
    for tag in ["gmsh:physical", "gmsh:geometrical"]:
        if tag not in tag_data:
            logging.warning(
                "Appending zeros to replace the missing {} tag data.".format(tag[5:])
            )
            tag_data[tag] = [numpy.zeros(len(x.data), dtype=c_int) for x in mesh.cells]

    if binary:
        for k, (key, value) in enumerate(mesh.cells):
            if value.dtype != c_int:
                logging.warning(
                    "Binary Gmsh needs 32-bit integers (got %s). Converting.",
                    value.dtype,
                )
                mesh.cells[k] = CellBlock(key, numpy.array(value, dtype=c_int))

    cells = _meshio_to_gmsh_order(mesh.cells)

    with open(filename, "wb") as fh:
        mode_idx = 1 if binary else 0
        size_of_double = 8
        fh.write(f"$MeshFormat\n2.2 {mode_idx} {size_of_double}\n".encode("utf-8"))
        if binary:
            numpy.array([1], dtype=c_int).tofile(fh)
            fh.write(b"\n")
        fh.write(b"$EndMeshFormat\n")

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_nodes(fh, mesh.points, float_fmt, binary)
        _write_elements(fh, cells, tag_data, binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic, float_fmt)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodeData", name, dat, binary)

        cell_data_raw = raw_from_cell_data(other_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, binary)


def _write_nodes(fh, points, float_fmt, binary):
    fh.write(b"$Nodes\n")
    fh.write("{}\n".format(len(points)).encode("utf-8"))
    if binary:
        dtype = [("index", c_int), ("x", c_double, (3,))]
        tmp = numpy.empty(len(points), dtype=dtype)
        tmp["index"] = 1 + numpy.arange(len(points))
        tmp["x"] = points
        tmp.tofile(fh)
        fh.write(b"\n")
    else:
        fmt = "{} " + " ".join(3 * ["{:" + float_fmt + "}"]) + "\n"
        for k, x in enumerate(points):
            fh.write(fmt.format(k + 1, x[0], x[1], x[2]).encode("utf-8"))
    fh.write(b"$EndNodes\n")


def _write_elements(fh, cells, tag_data, binary):
    # write elements
    fh.write(b"$Elements\n")
    # count all cells
    total_num_cells = sum([c.shape[0] for _, c in cells])
    fh.write(f"{total_num_cells}\n".encode("utf-8"))

    consecutive_index = 0
    for k, (cell_type, node_idcs) in enumerate(cells):
        tags = []
        for name in ["gmsh:physical", "gmsh:geometrical", "cell_tags"]:
            if name in tag_data:
                tags.append(tag_data[name][k])
        fcd = numpy.concatenate([tags]).astype(c_int).T

        if len(fcd) == 0:
            fcd = numpy.empty((len(node_idcs), 0), dtype=c_int)

        if binary:
            # header
            header = [_meshio_to_gmsh_type[cell_type], node_idcs.shape[0], fcd.shape[1]]
            numpy.array(header, dtype=c_int).tofile(fh)
            # actual data
            a = numpy.arange(len(node_idcs), dtype=c_int)[:, numpy.newaxis]
            a += 1 + consecutive_index
            array = numpy.hstack([a, fcd, node_idcs + 1])
            if array.dtype != c_int:
                raise WriteError(f"Wrong dtype (require c_int, got {array.dtype})")
            array.tofile(fh)
        else:
            form = (
                "{} "
                + str(_meshio_to_gmsh_type[cell_type])
                + " "
                + str(fcd.shape[1])
                + " {} {}\n"
            )
            for i, c in enumerate(node_idcs):
                fh.write(
                    form.format(
                        consecutive_index + i + 1,
                        " ".join([str(val) for val in fcd[i]]),
                        # a bit clumsy for `c+1`, but if c is uint64, c+1 is float64
                        " ".join([str(cc) for cc in c + numpy.array(1, dtype=c.dtype)]),
                    ).encode("utf-8")
                )

        consecutive_index += len(node_idcs)
    if binary:
        fh.write(b"\n")
    fh.write(b"$EndElements\n")


def _write_periodic(fh, periodic, float_fmt):
    fh.write(b"$Periodic\n")
    fh.write("{}\n".format(len(periodic)).encode("utf-8"))
    for dim, (stag, mtag), affine, slave_master in periodic:
        fh.write(f"{dim} {stag} {mtag}\n".encode("utf-8"))
        if affine is not None:
            fh.write(b"Affine ")
            affine = numpy.array(affine, dtype=float)
            affine = numpy.atleast_2d(affine.ravel())
            numpy.savetxt(fh, affine, fmt="%" + float_fmt)
        slave_master = numpy.array(slave_master, dtype=c_int).reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 0-based
        fh.write("{}\n".format(len(slave_master)).encode("utf-8"))
        for snode, mnode in slave_master:
            fh.write(f"{snode} {mnode}\n".encode("utf-8"))
    fh.write(b"$EndPeriodic\n")
