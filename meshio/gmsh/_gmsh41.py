"""
I/O for Gmsh's msh format (version 4.1, as used by Gmsh 4.2.2+), cf.
<http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.
"""
import logging
from functools import partial

import numpy

from .._common import _geometric_dimension, cell_data_from_raw, raw_from_cell_data
from .._exceptions import ReadError, WriteError
from .._mesh import CellBlock, Mesh
from .common import (
    _gmsh_to_meshio_order,
    _gmsh_to_meshio_type,
    _meshio_to_gmsh_order,
    _meshio_to_gmsh_type,
    _read_data,
    _read_physical_names,
    _write_data,
    _write_physical_names,
    num_nodes_per_cell,
)

c_int = numpy.dtype("i")
c_size_t = numpy.dtype("P")
c_double = numpy.dtype("d")


def _size_type(data_size):
    return numpy.dtype("u{}".format(data_size))


def read_buffer(f, is_ascii, data_size):
    # The format is specified at
    # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.

    # Initialize the optional data fields
    points = []
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}
    physical_tags = None
    cell_sets = {}
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
        elif environ == "Entities":
            physical_tags = _read_entities(f, is_ascii, data_size)
        elif environ == "Nodes":
            points, point_tags = _read_nodes(f, is_ascii, data_size)
        elif environ == "Elements":
            cells, cell_tags, cell_sets = _read_elements(
                f, point_tags, physical_tags, is_ascii, data_size, field_data
            )
        elif environ == "Periodic":
            periodic = _read_periodic(f, is_ascii, data_size)
        elif environ == "NodeData":
            _read_data(f, "NodeData", point_data, data_size, is_ascii)
        elif environ == "ElementData":
            _read_data(f, "ElementData", cell_data_raw, data_size, is_ascii)
        else:
            # From
            # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>:
            # ```
            # Any section with an unrecognized header is simply ignored: you can thus
            # add comments in a .msh file by putting them e.g. inside a
            # $Comments/$EndComments section.
            # ```
            # skip environment
            while line != "$End" + environ:
                line = f.readline()
                # Skip binary strings, but try to recognize text strings
                # to catch the end of the environment
                # See also https://github.com/nschloe/pygalmesh/issues/34
                try:
                    line = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    pass

    cell_data = cell_data_from_raw(cells, cell_data_raw)
    cell_data.update(cell_tags)

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        cell_sets=cell_sets,
        gmsh_periodic=periodic,
    )


def _read_entities(f, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)
    physical_tags = tuple({} for _ in range(4))  # dims 0, 1, 2, 3
    number = fromfile(f, c_size_t, 4)  # dims 0, 1, 2, 3

    for d, n in enumerate(number):
        for _ in range(n):
            (tag,) = fromfile(f, c_int, 1)
            fromfile(f, c_double, 3 if d == 0 else 6)  # discard bounding-box
            (num_physicals,) = fromfile(f, c_size_t, 1)
            physical_tags[d][tag] = list(fromfile(f, c_int, num_physicals))
            if d > 0:  # discard tagBREP{Vert,Curve,Surfaces}
                (num_BREP_,) = fromfile(f, c_size_t, 1)
                fromfile(f, c_int, num_BREP_)

    if not is_ascii:
        line = f.readline().decode("utf-8").strip()
        if line != "":
            raise ReadError()
    line = f.readline().decode("utf-8").strip()
    if line != "$EndEntities":
        raise ReadError()
    return physical_tags


def _read_nodes(f, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numNodes minNodeTag maxNodeTag (all size_t)
    num_entity_blocks, total_num_nodes, min_node_tag, max_node_tag = fromfile(
        f, c_size_t, 4
    )

    points = numpy.empty((total_num_nodes, 3), dtype=float)
    tags = numpy.empty(total_num_nodes, dtype=int)

    idx = 0
    for k in range(num_entity_blocks):
        # entityDim(int) entityTag(int) parametric(int) numNodes(size_t)
        _, __, parametric = fromfile(f, c_int, 3)
        if parametric != 0:
            raise ReadError("parametric nodes not implemented")
        num_nodes = int(fromfile(f, c_size_t, 1)[0])

        # From <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>:
        # > [...] tags can be "sparse", i.e., do not have to constitute a continuous
        # > list of numbers (the format even allows them to not be ordered).
        #
        # Following https://github.com/nschloe/meshio/issues/388, we read the tags and
        # populate the points array accordingly, thereby preserving the order of indices
        # of nodes/points.
        ixx = slice(idx, idx + num_nodes)
        tags[ixx] = fromfile(f, c_size_t, num_nodes) - 1

        # Store the point densely and in the order in which they appear in the file.
        # x(double) y(double) z(double) (* numNodes)
        points[ixx] = fromfile(f, c_double, num_nodes * 3).reshape((num_nodes, 3))
        idx += num_nodes

    if not is_ascii:
        line = f.readline().decode("utf-8")
        if line != "\n":
            raise ReadError()

    line = f.readline().decode("utf-8")
    if line.strip() != "$EndNodes":
        raise ReadError()

    return points, tags


def _read_elements(f, point_tags, physical_tags, is_ascii, data_size, field_data):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numElements minElementTag maxElementTag (all size_t)
    num_entity_blocks, total_num_elements, min_ele_tag, max_ele_tag = fromfile(
        f, c_size_t, 4
    )

    data = []
    cell_data = {}
    cell_sets = {k: [None] * num_entity_blocks for k in field_data.keys()}

    for k in range(num_entity_blocks):
        # entityDim(int) entityTag(int) elementType(int) numElements(size_t)
        dim_entity, tag_entity, type_ele = fromfile(f, c_int, 3)
        (num_ele,) = fromfile(f, c_size_t, 1)
        for physical_name, cell_set in cell_sets.items():
            cell_set[k] = numpy.arange(
                num_ele
                if (
                    physical_tags
                    and field_data[physical_name][1] == dim_entity
                    and field_data[physical_name][0]
                    in physical_tags[dim_entity][tag_entity]
                )
                else 0,
                dtype=type(num_ele),
            )
        tpe = _gmsh_to_meshio_type[type_ele]
        num_nodes_per_ele = num_nodes_per_cell[tpe]
        d = fromfile(f, c_size_t, int(num_ele * (1 + num_nodes_per_ele))).reshape(
            (num_ele, -1)
        )
        if physical_tags is None:
            data.append((None, tpe, d))
        else:
            data.append((physical_tags[dim_entity][tag_entity], tpe, d))

    if not is_ascii:
        line = f.readline().decode("utf-8")
        if line != "\n":
            raise ReadError()
    line = f.readline().decode("utf-8")
    if line.strip() != "$EndElements":
        raise ReadError()

    # Inverse point tags
    inv_tags = numpy.full(numpy.max(point_tags) + 1, -1, dtype=int)
    inv_tags[point_tags] = numpy.arange(len(point_tags))

    # Note that the first column in the data array is the element tag; discard it.
    data = [
        (physical_tag, tpe, inv_tags[d[:, 1:] - 1]) for physical_tag, tpe, d in data
    ]

    cells = []
    for physical_tag, key, values in data:
        cells.append((key, values))
        if physical_tag:
            if "gmsh:physical" not in cell_data:
                cell_data["gmsh:physical"] = []
            cell_data["gmsh:physical"].append(
                physical_tag[0] * numpy.ones(len(values), int)
            )
    cells[:] = _gmsh_to_meshio_order(cells)

    return cells, cell_data, cell_sets


def _read_periodic(f, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)
    periodic = []
    # numPeriodicLinks(size_t)
    num_periodic = int(fromfile(f, c_size_t, 1)[0])
    for _ in range(num_periodic):
        # entityDim(int) entityTag(int) entityTagMaster(int)
        edim, stag, mtag = fromfile(f, c_int, 3)
        # numAffine(size_t) value(double) ...
        num_affine = int(fromfile(f, c_size_t, 1)[0])
        affine = fromfile(f, c_double, num_affine)
        # numCorrespondingNodes(size_t)
        num_nodes = int(fromfile(f, c_size_t, 1)[0])
        # nodeTag(size_t) nodeTagMaster(size_t) ...
        slave_master = fromfile(f, c_size_t, num_nodes * 2).reshape(-1, 2)
        slave_master = slave_master - 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), affine, slave_master])
    if not is_ascii:
        line = f.readline().decode("utf-8")
        if line != "\n":
            raise ReadError()
    line = f.readline().decode("utf-8")
    if line.strip() != "$EndPeriodic":
        raise ReadError()
    return periodic


def write(filename, mesh, float_fmt=".16e", binary=True):
    write4_1(filename, mesh, float_fmt=float_fmt, binary=binary)


def write4_1(filename, mesh, float_fmt=".16e", binary=True):
    """Writes msh files, cf.
    <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.
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
                    "Binary Gmsh needs c_size_t (got %s). Converting.", value.dtype
                )
                mesh.cells[k] = CellBlock(key, value.astype(c_size_t))

    cells = _meshio_to_gmsh_order(mesh.cells)

    with open(filename, "wb") as fh:
        file_type = 1 if binary else 0
        data_size = c_size_t.itemsize
        fh.write(b"$MeshFormat\n")
        fh.write("4.1 {} {}\n".format(file_type, data_size).encode("utf-8"))
        if binary:
            numpy.array([1], dtype=c_int).tofile(fh)
            fh.write(b"\n")
        fh.write(b"$EndMeshFormat\n")

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_entities(fh, cells, binary)
        _write_nodes(fh, mesh.points, mesh.cells, float_fmt, binary)
        _write_elements(fh, cells, binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic, float_fmt, binary)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodeData", name, dat, binary)

        cell_data_raw = raw_from_cell_data(mesh.cell_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, binary)


def _write_entities(fh, cells, binary):
    """write the $Entities block

    specified as

    numPoints(size_t) numCurves(size_t)
      numSurfaces(size_t) numVolumes(size_t)
    pointTag(int) X(double) Y(double) Z(double)
      numPhysicalTags(size_t) physicalTag(int) ...
    ...
    curveTag(int) minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundingPoints(size_t) pointTag(int) ...
    ...
    surfaceTag(int) minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundingCurves(size_t) curveTag(int) ...
    ...
    volumeTag(int) minX(double) minY(double) minZ(double)
      maxX(double) maxY(double) maxZ(double)
      numPhysicalTags(size_t) physicalTag(int) ...
      numBoundngSurfaces(size_t) surfaceTag(int) ...
    ...

    """
    # fh.write("$Entities\n".encode("utf-8"))
    # raise NotImplementedError
    # fh.write("$EndEntities\n".encode("utf-8"))
    return


def _write_nodes(fh, points, cells, float_fmt, binary):
    fh.write(b"$Nodes\n")

    # The entity_dim and entity_tag in the $Elements section must correspond to an
    # entity_dim and entity_tag array in the $Nodes section.
    # TODO Not sure what to do if there are multiple element types present.
    if len(cells) != 1:
        raise WriteError("Can only deal with one cell type for now")
    dim_entity = _geometric_dimension[cells[0][0]]
    entity_tag = 0

    # write all points as one big block
    #
    # $Nodes
    #   numEntityBlocks(size_t) numNodes(size_t) minNodeTag(size_t) maxNodeTag(size_t)
    #   entityDim(int) entityTag(int) parametric(int; 0 or 1)
    #   numNodesInBlock(size_t)
    #     nodeTag(size_t)
    #     ...
    #     x(double) y(double) z(double)
    #        < u(double; if parametric and entityDim >= 1) >
    #        < v(double; if parametric and entityDim >= 2) >
    #        < w(double; if parametric and entityDim == 3) >
    #     ...
    #   ...
    # $EndNodes
    #
    n = points.shape[0]
    num_blocks = 1
    min_tag = 1
    max_tag = n
    is_parametric = 0
    if binary:
        if points.dtype != c_double:
            logging.warning(
                "Binary Gmsh needs c_double points (got %s). Converting.", points.dtype
            )
            points = points.astype(c_double)
        numpy.array([num_blocks, n, min_tag, max_tag], dtype=c_size_t).tofile(fh)
        numpy.array([dim_entity, entity_tag, is_parametric], dtype=c_int).tofile(fh)
        numpy.array([n], dtype=c_size_t).tofile(fh)
        numpy.arange(1, 1 + n, dtype=c_size_t).tofile(fh)
        points.tofile(fh)
        fh.write(b"\n")
    else:
        fh.write(
            "{} {} {} {}\n".format(num_blocks, n, min_tag, max_tag).encode("utf-8")
        )
        fh.write(
            "{} {} {} {}\n".format(dim_entity, entity_tag, is_parametric, n).encode(
                "utf-8"
            )
        )
        numpy.arange(1, 1 + n, dtype=c_size_t).tofile(fh, "\n", "%d")
        fh.write(b"\n")
        numpy.savetxt(fh, points, delimiter=" ", fmt="%" + float_fmt)

    fh.write(b"$EndNodes\n")
    return


def _write_elements(fh, cells, binary):
    """write the $Elements block

    $Elements
      numEntityBlocks(size_t)
      numElements(size_t) minElementTag(size_t) maxElementTag(size_t)
      entityDim(int) entityTag(int) elementType(int; see below) numElementsInBlock(size_t)
        elementTag(size_t) nodeTag(size_t) ...
        ...
      ...
    $EndElements
    """
    fh.write(b"$Elements\n")

    total_num_cells = sum(len(c) for _, c in cells)
    num_blocks = len(cells)
    min_element_tag = 1
    max_element_tag = total_num_cells
    if binary:
        numpy.array(
            [num_blocks, total_num_cells, min_element_tag, max_element_tag],
            dtype=c_size_t,
        ).tofile(fh)

        tag0 = 1
        for cell_type, node_idcs in cells:
            # entityDim(int) entityTag(int) elementType(int)
            # numElementsBlock(size_t)
            dim = _geometric_dimension[cell_type]
            entity_tag = 0
            cell_type = _meshio_to_gmsh_type[cell_type]
            numpy.array([dim, entity_tag, cell_type], dtype=c_int).tofile(fh)
            n = node_idcs.shape[0]
            numpy.array([n], dtype=c_size_t).tofile(fh)

            if node_idcs.dtype != c_size_t:
                logging.warning(
                    "Binary Gmsh cells need c_size_t (got %s). Converting.",
                    node_idcs.dtype,
                )
                node_idcs = node_idcs.astype(c_size_t)

            numpy.column_stack(
                [
                    numpy.arange(tag0, tag0 + n, dtype=c_size_t),
                    # increment indices by one to conform with gmsh standard
                    node_idcs + 1,
                ]
            ).tofile(fh)
            tag0 += n

        fh.write(b"\n")
    else:
        fh.write(
            "{} {} {} {}\n".format(
                num_blocks, total_num_cells, min_element_tag, max_element_tag
            ).encode("utf-8")
        )

        tag0 = 1
        for cell_type, node_idcs in cells:
            # entityDim(int) entityTag(int) elementType(int) numElementsBlock(size_t)
            dim = _geometric_dimension[cell_type]
            entity_tag = 0
            cell_type = _meshio_to_gmsh_type[cell_type]
            n = node_idcs.shape[0]
            fh.write(
                "{} {} {} {}\n".format(dim, entity_tag, cell_type, n).encode("utf-8")
            )

            numpy.savetxt(
                fh,
                # Gmsh indexes from 1 not 0
                numpy.column_stack([numpy.arange(tag0, tag0 + n), node_idcs + 1]),
                "%d",
                " ",
            )
            tag0 += n

    fh.write(b"$EndElements\n")
    return


def _write_periodic(fh, periodic, float_fmt, binary):
    """write the $Periodic block

    specified as

    $Periodic
      numPeriodicLinks(size_t)
      entityDim(int) entityTag(int) entityTagMaster(int)
      numAffine(size_t) value(double) ...
      numCorrespondingNodes(size_t)
        nodeTag(size_t) nodeTagMaster(size_t)
        ...
      ...
    $EndPeriodic

    """

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
    tofile(fh, len(periodic), c_size_t)
    for dim, (stag, mtag), affine, slave_master in periodic:
        tofile(fh, [dim, stag, mtag], c_int)
        if affine is None or len(affine) == 0:
            tofile(fh, 0, c_size_t)
        else:
            tofile(fh, len(affine), c_size_t, newline=" ")
            tofile(fh, affine, c_double, fmt=float_fmt)
        slave_master = numpy.array(slave_master, dtype=c_size_t)
        slave_master = slave_master.reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 1-based
        tofile(fh, len(slave_master), c_size_t)
        tofile(fh, slave_master, c_size_t)
    if binary:
        fh.write(b"\n")
    fh.write(b"$EndPeriodic\n")
