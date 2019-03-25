# -*- coding: utf-8 -*-
#
"""
I/O for Gmsh's msh format (version 4.1, as used by Gmsh 4.2.2), cf.
<http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.
"""
from functools import partial
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
from .msh2 import write as write2  # revert where necessary; TODO: drop this

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
    cells = {}
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
        assert line[0] == "$"
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
                line = f.readline().decode("utf-8").strip()

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
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)
    physical_tags = tuple({} for _ in range(4))  # dims 0, 1, 2, 3
    number = fromfile(f, c_size_t, 4)  # dims 0, 1, 2, 3

    for d, n in enumerate(number):
        for _ in range(n):
            tag, = fromfile(f, c_int, 1)
            fromfile(f, c_double, 3 if d == 0 else 6)  # discard bounding-box
            num_physicals, = fromfile(f, c_size_t, 1)
            physical_tags[d][tag] = list(fromfile(f, c_int, num_physicals))
            if d > 0:  # discard tagBREP{Vert,Curve,Surfaces}
                num_BREP_, = fromfile(f, c_size_t, 1)
                fromfile(f, c_int, num_BREP_)

    if not is_ascii:
        line = f.readline().decode("utf-8").strip()
        assert line == ""
    line = f.readline().decode("utf-8").strip()
    assert line == "$EndEntities"
    return physical_tags


def _read_nodes(f, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numNodes minNodeTag maxNodeTag (all size_t)
    num_entity_blocks, total_num_nodes, min_node_tag, max_node_tag = fromfile(
        f, c_size_t, 4
    )
    is_dense = min_node_tag == 1 and max_node_tag == total_num_nodes

    points = numpy.empty((total_num_nodes, 3), dtype=float)
    tags = (numpy.arange if is_dense else numpy.empty)(total_num_nodes, dtype=int)

    idx = 0
    for k in range(num_entity_blocks):
        # entityDim(int) entityTag(int) parametric(int) numNodes(size_t)
        _, __, parametric = fromfile(f, c_int, 3)
        assert parametric == 0, "parametric nodes not implemented"
        num_nodes = int(fromfile(f, c_size_t, 1)[0])
        ixx = slice(idx, idx + num_nodes)

        if is_dense:
            fromfile(f, c_size_t, num_nodes)
        else:
            tags[ixx] = fromfile(f, c_size_t, num_nodes) - 1
        points[ixx] = fromfile(f, c_double, num_nodes * 3).reshape((num_nodes, 3))
        idx += num_nodes

    if not is_ascii:

        line = f.readline().decode("utf-8")
        assert line == "\n"

    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndNodes"

    return points, tags


def _read_elements(f, point_tags, physical_tags, is_ascii, data_size):
    fromfile = partial(numpy.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numElements minElementTag maxElementTag (all size_t)
    num_entity_blocks, total_num_elements, min_ele_tag, max_ele_tag = fromfile(
        f, c_size_t, 4
    )
    is_dense = min_ele_tag == 1 and max_ele_tag == total_num_elements

    data = []
    cell_data = {}

    for k in range(num_entity_blocks):
        # entityDim(int) entityTag(int) elementType(int) numElements(size_t)
        dim_entity, tag_entity, type_ele = fromfile(f, c_int, 3)
        num_ele, = fromfile(f, c_size_t, 1)
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
        assert line == "\n"
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndElements"

    if is_dense:
        itags = point_tags
    else:
        m = numpy.max(point_tags + 1)
        itags = -numpy.ones(m, dtype=int)
        itags[point_tags] = numpy.arange(len(point_tags))

    # Note that the first column in the data array is the element tag; discard it.
    data = [(physical_tag, tpe, itags[d[:, 1:] - 1]) for physical_tag, tpe, d in data]

    cells = {}
    for physical_tag, key, values in data:
        if key in cells:
            cells[key] = numpy.concatenate([cells[key], values])
            if physical_tag:
                if key not in cell_data:
                    cell_data[key] = {}
                cell_data[key]["gmsh:physical"] = numpy.concatenate(
                    [
                        cell_data[key]["gmsh:physical"],
                        physical_tag[0] * numpy.ones(len(values), int),
                    ]
                )
        else:
            cells[key] = values
            if physical_tag:
                if key not in cell_data:
                    cell_data[key] = {}
                cell_data[key]["gmsh:physical"] = physical_tag[0] * numpy.ones(
                    len(values), int
                )

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]

    return cells, cell_data


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
        assert line == "\n"
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPeriodic"
    return periodic


def write(filename, mesh, write_binary=True):
    logging.warning("Writing MSH4.1 unimplemented, falling back on MSH2")
    write2(filename, mesh, write_binary=write_binary)


def write4_1(filename, mesh, write_binary=True):
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

    if write_binary:
        for key, value in mesh.cells.items():
            if value.dtype != c_int:
                logging.warning(
                    "Binary Gmsh needs c_size_t (got %s). Converting.", value.dtype
                )
                mesh.cells[key] = value.astype(c_size_t)

    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    cells = mesh.cells.copy()
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]
        ]

    with open(filename, "wb") as fh:
        file_type = 1 if write_binary else 0
        data_size = c_size_t.itemsize
        fh.write("$MeshFormat\n".encode("utf-8"))
        fh.write("4.1 {} {}\n".format(file_type, data_size).encode("utf-8"))
        if write_binary:
            fh.write(struct.pack("i", 1))
            fh.write("\n".encode("utf-8"))
        fh.write("$EndMeshFormat\n".encode("utf-8"))

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_entities(fh, cells, write_binary)
        _write_nodes(fh, mesh.points, write_binary)
        _write_elements(fh, cells, write_binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic, write_binary)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodeData", name, dat, write_binary)
        cell_data_raw = raw_from_cell_data(mesh.cell_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, write_binary)


def _write_entities(fh, cells, write_binary):
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
    fh.write("$Entities\n".encode("utf-8"))
    raise NotImplementedError
    fh.write("$EndEntities\n".encode("utf-8"))


def _write_nodes(fh, points, write_binary):
    fh.write("$Nodes\n".encode("utf-8"))

    # TODO not sure what dimEntity is supposed to say
    dim_entity = 0

    # write all points as one big block
    # numEntityBlocks(size_t) numNodes(size_t) minNodeTag(size_t) maxNodeTag(size_t)
    # entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesBlock(size_t)
    #   nodeTag(size_t)
    #   ...
    #   x(double) y(double) z(double)
    #     < u(double; if parametric and entityDim = 1 or entityDim = 2) >
    #     < v(double; if parametric and entityDim = 2) >
    #   ...
    # ...
    if write_binary:
        fh.write(
            numpy.array([1, len(points), 1, len(points)], dtype=c_size_t).tostring()
        )

        fh.write(numpy.array([dim_entity, 1, 0], dtype=c_int).tostring())
        fh.write(numpy.array([len(points)], dtype=c_size_t).tostring())

        fh.write(numpy.arange(1, 1 + len(points), dtype=c_size_t).tostring())
        fh.write(points.tostring())

        fh.write("\n".encode("utf-8"))
    else:
        fh.write("{} {} {} {}\n".format(1, len(points), 1, len(points)).encode("utf-8"))
        fh.write("{} {} {} {}\n".format(dim_entity, 1, 0, len(points)).encode("utf-8"))
        numpy.arange(1, 1 + len(points), dtype=c_size_t).tofile(fh, "\n", "%d")
        fh.write("\n".encode("utf-8"))
        numpy.savetxt(fh, points, delimiter=" ")

    fh.write("$EndNodes\n".encode("utf-8"))
    return


def _write_elements(fh, cells, write_binary):
    """write the $Elements block

    $Elements
      numEntityBlocks(size_t) numElements(size_t)
        minElementTag(size_t) maxElementTag(size_t)
      entityDim(int) entityTag(int) elementType(int) numElementsBlock(size_t)
        elementTag(size_t) nodeTag(size_t) ...
        ...
      ...
    $EndElements

    """
    fh.write("$Elements\n".encode("utf-8"))

    total_num_cells = sum(map(len, cells.values()))
    if write_binary:
        fh.write(
            numpy.array(
                [len(cells), total_num_cells, 1, total_num_cells], dtype=c_size_t
            ).tostring()
        )

        first_element_tag_in_entity = 0
        for entity_tag, (cell_type, node_idcs) in enumerate(cells.items(), 1):
            # entityDim(int) entityTag(int) elementType(int) numElementsBlock(size_t)
            fh.write(
                numpy.array(
                    [
                        _geometric_dimension[cell_type],
                        entity_tag,
                        _meshio_to_gmsh_type[cell_type],
                    ],
                    dtype=c_int,
                ).tostring()
            )
            fh.write(numpy.array([node_idcs.shape[0]], dtype=c_size_t).tostring())

            assert node_idcs.dtype == c_size_t
            data = numpy.column_stack(
                [
                    numpy.arange(
                        first_element_tag_in_entity,
                        first_element_tag_in_entity + len(node_idcs),
                        dtype=c_int,
                    ),
                    # increment indices by one to conform with gmsh standard
                    node_idcs + 1,
                ]
            )
            fh.write(data.tostring())
            first_element_tag_in_entity += len(node_idcs)

        fh.write("\n".encode("utf-8"))
    else:
        fh.write(
            "{} {} {} {}\n".format(
                len(cells), total_num_cells, 1, total_num_cells
            ).encode("utf-8")
        )

        first_element_tag_in_entity = 1
        for entity_tag, (cell_type, node_idcs) in enumerate(cells.items(), 1):
            # entityDim(int) entityTag(int) elementType(int) numElementsBlock(size_t)
            fh.write(
                "{} {} {} {}\n".format(
                    _geometric_dimension[cell_type],
                    entity_tag,
                    _meshio_to_gmsh_type[cell_type],
                    node_idcs.shape[0],
                ).encode("utf-8")
            )

            numpy.savetxt(
                fh,
                numpy.column_stack(
                    [
                        first_element_tag_in_entity + numpy.arange(len(node_idcs)),
                        node_idcs + 1,  # Gmsh indexes from 1 not 0
                    ]
                ).astype(c_size_t),
                "%d",
                " ",
            )
            first_element_tag_in_entity += len(node_idcs)

    fh.write("$EndElements\n".encode("utf-8"))
    return


def _write_periodic(fh, periodic, write_binary):
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
        if write_binary:
            ary.tofile(fh)
        else:
            ary = numpy.atleast_2d(ary)
            fmt = "%.16g" if dtype == c_double else "%d"
            fmt = kwargs.pop("fmt", fmt)
            numpy.savetxt(fh, ary, fmt, **kwargs)

    fh.write("$Periodic\n".encode("utf-8"))
    tofile(fh, len(periodic), c_size_t)
    for dim, (stag, mtag), affine, slave_master in periodic:
        tofile(fh, [dim, stag, mtag], c_int)
        if affine is None or len(affine) == 0:
            tofile(fh, 0, c_size_t)
        else:
            tofile(fh, len(affine), c_size_t, newline=" ")
            tofile(fh, affine, c_double, fmt="%.16g")
        slave_master = numpy.array(slave_master, dtype=c_size_t)
        slave_master = slave_master.reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 1-based
        tofile(fh, len(slave_master), c_size_t)
        tofile(fh, slave_master, c_size_t)
    if write_binary:
        fh.write("\n".encode("utf-8"))
    fh.write("$EndPeriodic\n".encode("utf-8"))


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
