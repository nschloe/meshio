"""
I/O for Gmsh's msh format (version 4.1, as used by Gmsh 4.2.2+), cf.
<http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.
"""
from functools import partial

import numpy as np

from .._common import cell_data_from_raw, num_nodes_per_cell, raw_from_cell_data, warn
from .._exceptions import ReadError, WriteError
from .._mesh import CellBlock, Mesh
from .common import (
    _fast_forward_over_blank_lines,
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

c_int = np.dtype("i")
c_size_t = np.dtype("P")
c_double = np.dtype("d")


def _size_type(data_size):
    return np.dtype(f"u{data_size}")


def read_buffer(f, is_ascii: bool, data_size):
    # The format is specified at
    # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.

    # Initialize the optional data fields
    points = []
    cells = None
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}
    physical_tags = None
    bounding_entities = None
    cell_sets = {}
    periodic = None
    while True:
        # fast-forward over blank lines
        line, is_eof = _fast_forward_over_blank_lines(f)
        if is_eof:
            break

        if line[0] != "$":
            raise ReadError(f"Unexpected line {repr(line)}")

        environ = line[1:].strip()

        if environ == "PhysicalNames":
            _read_physical_names(f, field_data)
        elif environ == "Entities":
            # Read physical tags and information on bounding entities.
            # The information is passed to the processing of elements.
            physical_tags, bounding_entities = _read_entities(f, is_ascii, data_size)
        elif environ == "Nodes":
            points, point_tags, point_entities = _read_nodes(f, is_ascii, data_size)
        elif environ == "Elements":
            cells, cell_tags, cell_sets = _read_elements(
                f,
                point_tags,
                physical_tags,
                bounding_entities,
                is_ascii,
                data_size,
                field_data,
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
            _fast_forward_to_end_block(f, environ)

    if cells is None:
        raise ReadError("$Element section not found.")
    cell_data = cell_data_from_raw(cells, cell_data_raw)
    cell_data.update(cell_tags)

    # Add node entity information to the point data
    point_data.update({"gmsh:dim_tags": point_entities})

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        cell_sets=cell_sets,
        gmsh_periodic=periodic,
    )


def _read_entities(f, is_ascii: bool, data_size):
    # Read the entity section. Return physical tags of the entities, and (for
    # entities of dimension > 0) the bounding entities (so points that form
    # the boundary of a line etc).
    # Note that the bounding box of the entities is disregarded. Adding this
    # is not difficult, but for the moment, the entropy of adding more data
    # does not seem warranted.

    fromfile = partial(np.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)
    physical_tags = ({}, {}, {}, {})
    bounding_entities = ({}, {}, {}, {})
    number = fromfile(f, c_size_t, 4)  # dims 0, 1, 2, 3
    for d, n in enumerate(number):
        for _ in range(n):
            (tag,) = fromfile(f, c_int, 1)
            fromfile(f, c_double, 3 if d == 0 else 6)  # discard bounding-box
            (num_physicals,) = fromfile(f, c_size_t, 1)
            physical_tags[d][tag] = list(fromfile(f, c_int, num_physicals))
            if d > 0:
                # Number of bounding entities
                num_BREP_ = fromfile(f, c_size_t, 1)[0]
                # Store bounding entities
                bounding_entities[d][tag] = fromfile(f, c_int, num_BREP_)

    _fast_forward_to_end_block(f, "Entities")
    return physical_tags, bounding_entities


def _read_nodes(f, is_ascii: bool, data_size):
    # Read node data: Node coordinates and tags.
    # Also find the entities of the nodes, and store this as point_data.
    # Note that entity tags are 1-offset within each dimension, thus it is
    # necessary to keep track of both tag and dimension of the entity

    fromfile = partial(np.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numNodes minNodeTag maxNodeTag (all size_t)
    num_entity_blocks, total_num_nodes, _, _ = fromfile(f, c_size_t, 4)

    points = np.empty((total_num_nodes, 3), dtype=float)
    tags = np.empty(total_num_nodes, dtype=int)
    dim_tags = np.empty((total_num_nodes, 2), dtype=int)

    # To save the entity block id for each node, initialize an array here,
    # populate it with num_nodes
    idx = 0
    for _ in range(num_entity_blocks):
        # entityDim(int) entityTag(int) parametric(int) numNodes(size_t)
        dim, entity_tag, parametric = fromfile(f, c_int, 3)
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

        # Entity tag and entity dimension of the nodes. Stored as point-data.
        dim_tags[ixx, 0] = dim
        dim_tags[ixx, 1] = entity_tag
        idx += num_nodes

    _fast_forward_to_end_block(f, "Nodes")

    return points, tags, dim_tags


def _read_elements(
    f, point_tags, physical_tags, bounding_entities, is_ascii, data_size, field_data
):
    fromfile = partial(np.fromfile, sep=" " if is_ascii else "")
    c_size_t = _size_type(data_size)

    # numEntityBlocks numElements minElementTag maxElementTag (all size_t)
    num_entity_blocks, _, _, _ = fromfile(f, c_size_t, 4)

    data = []
    cell_data = {}
    cell_sets = {k: [None] * num_entity_blocks for k in field_data.keys()}

    for k in range(num_entity_blocks):
        # entityDim(int) entityTag(int) elementType(int) numElements(size_t)
        dim, tag, type_ele = fromfile(f, c_int, 3)
        (num_ele,) = fromfile(f, c_size_t, 1)
        for physical_name, cell_set in cell_sets.items():
            cell_set[k] = np.arange(
                num_ele
                if (
                    physical_tags
                    and field_data[physical_name][1] == dim
                    and field_data[physical_name][0] in physical_tags[dim][tag]
                )
                else 0,
                dtype=type(num_ele),
            )
        tpe = _gmsh_to_meshio_type[type_ele]
        num_nodes_per_ele = num_nodes_per_cell[tpe]
        d = fromfile(f, c_size_t, int(num_ele * (1 + num_nodes_per_ele))).reshape(
            (num_ele, -1)
        )

        # Find physical tag, if defined; else it is None.
        pt = None if not physical_tags else physical_tags[dim][tag]
        # Bounding entities (of lower dimension) if defined. Else it is None.
        if dim > 0 and bounding_entities:  # Points have no boundaries
            be = bounding_entities[dim][tag]
        else:
            be = None
        data.append((pt, be, tag, tpe, d))

    _fast_forward_to_end_block(f, "Elements")

    # Inverse point tags
    inv_tags = np.full(np.max(point_tags) + 1, -1, dtype=int)
    inv_tags[point_tags] = np.arange(len(point_tags))

    # Note that the first column in the data array is the element tag; discard it.
    data = [
        (physical_tag, bound_entity, geom_tag, tpe, inv_tags[d[:, 1:] - 1])
        for physical_tag, bound_entity, geom_tag, tpe, d in data
    ]

    cells = []
    for physical_tag, bound_entity, geom_tag, key, values in data:
        cells.append(CellBlock(key, _gmsh_to_meshio_order(key, values)))
        if physical_tag:
            if "gmsh:physical" not in cell_data:
                cell_data["gmsh:physical"] = []
            cell_data["gmsh:physical"].append(
                np.full(len(values), physical_tag[0], int)
            )
        if "gmsh:geometrical" not in cell_data:
            cell_data["gmsh:geometrical"] = []
        cell_data["gmsh:geometrical"].append(np.full(len(values), geom_tag, int))

        # The bounding entities is stored in the cell_sets.
        if bounding_entities:
            if "gmsh:bounding_entities" not in cell_sets:
                cell_sets["gmsh:bounding_entities"] = []
            cell_sets["gmsh:bounding_entities"].append(bound_entity)

    return cells, cell_data, cell_sets


def _read_periodic(f, is_ascii, data_size):
    fromfile = partial(np.fromfile, sep=" " if is_ascii else "")
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

    _fast_forward_to_end_block(f, "Periodic")
    return periodic


def write(filename, mesh, float_fmt=".16e", binary=True):
    """Writes msh files, cf.
    <http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format>.
    """
    # Filter the point data: gmsh:dim_tags are tags, the rest is actual point data.
    point_data = {}
    for key, d in mesh.point_data.items():
        if key not in ["gmsh:dim_tags"]:
            point_data[key] = d

    # Split the cell data: gmsh:physical and gmsh:geometrical are tags, the rest is
    # actual cell data.
    tag_data = {}
    cell_data = {}
    for key, d in mesh.cell_data.items():
        if key in ["gmsh:physical", "gmsh:geometrical", "cell_tags"]:
            tag_data[key] = d
        else:
            cell_data[key] = d

    with open(filename, "wb") as fh:
        file_type = 1 if binary else 0
        data_size = c_size_t.itemsize
        fh.write(b"$MeshFormat\n")
        fh.write(f"4.1 {file_type} {data_size}\n".encode())
        if binary:
            np.array([1], dtype=c_int).tofile(fh)
            fh.write(b"\n")
        fh.write(b"$EndMeshFormat\n")

        if mesh.field_data:
            _write_physical_names(fh, mesh.field_data)

        _write_entities(
            fh, mesh.cells, tag_data, mesh.cell_sets, mesh.point_data, binary
        )
        _write_nodes(fh, mesh.points, mesh.cells, mesh.point_data, float_fmt, binary)
        _write_elements(fh, mesh.cells, tag_data, binary)
        if mesh.gmsh_periodic is not None:
            _write_periodic(fh, mesh.gmsh_periodic, float_fmt, binary)

        for name, dat in point_data.items():
            _write_data(fh, "NodeData", name, dat, binary)
        cell_data_raw = raw_from_cell_data(cell_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, "ElementData", name, dat, binary)


def _write_entities(fh, cells, tag_data, cell_sets, point_data, binary):
    """Write entity section in a .msh file.

    The entity section links up to three kinds of information:
        1) The geometric objects represented in the mesh.
        2) Physical tags of geometric objects. This data will be a subset
           of that represented in 1)
        3) Which geometric objects form the boundary of this object.
           The boundary is formed of objects with dimension 1 less than
           the current one. A boundary can only be specified for objects of
           dimension at least 1.

    The entities of all geometric objects is pulled from
    point_data['gmsh:dim_tags']. For details, see the function _write_nodes().

    Physical tags are specified as tag_data, while the boundary of a geometric
    object is specified in cell_sets.

    """

    # The data format for the entities section is
    #
    #    numPoints(size_t) numCurves(size_t)
    #      numSurfaces(size_t) numVolumes(size_t)
    #    pointTag(int) X(double) Y(double) Z(double)
    #      numPhysicalTags(size_t) physicalTag(int) ...
    #    ...
    #    curveTag(int) minX(double) minY(double) minZ(double)
    #      maxX(double) maxY(double) maxZ(double)
    #      numPhysicalTags(size_t) physicalTag(int) ...
    #      numBoundingPoints(size_t) pointTag(int) ...
    #    ...
    #    surfaceTag(int) minX(double) minY(double) minZ(double)
    #      maxX(double) maxY(double) maxZ(double)
    #      numPhysicalTags(size_t) physicalTag(int) ...
    #      numBoundingCurves(size_t) curveTag(int) ...
    #    ...
    #    volumeTag(int) minX(double) minY(double) minZ(double)
    #      maxX(double) maxY(double) maxZ(double)
    #      numPhysicalTags(size_t) physicalTag(int) ...
    #      numBoundngSurfaces(size_t) surfaceTag(int) ...

    # Both nodes and cells have entities, but the cell entities are a subset of
    # the nodes. The reason is (if the inner workings of Gmsh has been correctly
    # understood) that node entities are assigned to all
    # objects necessary to specify the geometry whereas only cells of Physical
    # objcets (gmsh jargon) are present among the cell entities.
    # The entities section must therefore be built on the node-entities, if
    # these are available. If this is not the case, we leave this section blank.
    # TODO: Should this give a warning?
    if "gmsh:dim_tags" not in point_data:
        return

    fh.write(b"$Entities\n")

    # Array of entity tag (first row) and dimension (second row) per node.
    # We need to combine the two, since entity tags are reset for each dimension.
    # Uniquify, so that each row in node_dim_tags represent a unique entity
    node_dim_tags = np.unique(point_data["gmsh:dim_tags"], axis=0)

    # Write number of entities per dimension
    num_occ = np.bincount(node_dim_tags[:, 0], minlength=4)
    if num_occ.size > 4:
        raise ValueError("Encountered entity with dimension > 3")

    if binary:
        num_occ.astype(c_size_t).tofile(fh)
    else:
        fh.write(f"{num_occ[0]} {num_occ[1]} {num_occ[2]} {num_occ[3]}\n".encode())

    # Array of dimension and entity tag per cell. Will be compared with the
    # similar not array.
    cell_dim_tags = np.empty((len(cells), 2), dtype=int)
    for ci, cell_block in enumerate(cells):
        cell_dim_tags[ci] = [
            cell_block.dim,
            tag_data["gmsh:geometrical"][ci][0],
        ]

    # We will only deal with bounding entities if this information is available
    has_bounding_elements = "gmsh:bounding_entities" in cell_sets

    # The node entities form a superset of cell entities. Write entity information
    # based on nodes, supplement with cell information when there is a matcihng
    # cell block.
    for dim, tag in node_dim_tags:
        # Find the matching cell block, if it exists
        matching_cell_block = np.where(
            np.logical_and(cell_dim_tags[:, 0] == dim, cell_dim_tags[:, 1] == tag)
        )[0]
        if matching_cell_block.size > 1:
            # It is not 100% clear if this is not permissible, but the current
            # implementation for sure does not allow it.
            raise ValueError("Encountered non-unique CellBlock dim_tag")

        # The information to be written varies according to entity dimension,
        # whether entity has a physical tag, and between ascii and binary.
        # The resulting code is a bit ugly, but no simpler and clean option
        # seems possible.

        # Entity tag
        if binary:
            np.array([tag], dtype=c_int).tofile(fh)
        else:
            fh.write(f"{tag} ".encode())

        # Min-max coordinates for the entity. For now, simply put zeros here,
        # and hope that gmsh does not complain. To expand this, the point
        # coordinates must be made available to this function; the bounding
        # box can then be found by a min-max over the points of the matching
        # cell.
        if dim == 0:
            # Bounding box is a point
            if binary:
                np.zeros(3, dtype=c_double).tofile(fh)
            else:
                fh.write(b"0 0 0 ")
        else:
            # Bounding box has six coordinates
            if binary:
                np.zeros(6, dtype=c_double).tofile(fh)
            else:
                fh.write(b"0 0 0 0 0 0 ")

        # If there is a corresponding cell block, write physical tags (if any)
        # and bounding entities (if any)
        if matching_cell_block.size > 0:
            # entity has a physical tag, write this
            # ASSUMPTION: There is a single physical tag for this
            physical_tag = tag_data["gmsh:physical"][matching_cell_block[0]][0]
            if binary:
                np.array([1], dtype=c_size_t).tofile(fh)
                np.array([physical_tag], dtype=c_int).tofile(fh)
            else:
                fh.write(f"1 {physical_tag} ".encode())
        else:
            # The number of physical tags is zero
            if binary:
                np.array([0], dtype=c_size_t).tofile(fh)
            else:
                fh.write(b"0 ")

        if dim > 0:
            # Entities not of the lowest dimension can have their
            # bounding elements (of dimension one less) specified
            if has_bounding_elements and matching_cell_block.size > 0:
                # The bounding element should be a list
                bounds = cell_sets["gmsh:bounding_entities"][matching_cell_block[0]]
                num_bounds = len(bounds)
                if num_bounds > 0:
                    if binary:
                        np.array(num_bounds, dtype=c_size_t).tofile(fh)
                        np.array(bounds, dtype=c_int).tofile(fh)
                    else:
                        fh.write(f"{num_bounds} ".encode())
                        for bi in bounds:
                            fh.write(f"{bi} ".encode())
                        fh.write(b"\n")
                else:
                    # Register that there are no bounding elements
                    if binary:
                        np.array([0], dtype=c_size_t).tofile(fh)
                    else:
                        fh.write(b"0\n")

            else:
                # Register that there are no bounding elements
                if binary:
                    np.array([0], dtype=c_size_t).tofile(fh)
                else:
                    fh.write(b"0\n")
        else:
            # If ascii, enforce line change
            if not binary:
                fh.write(b"\n")

    if binary:
        fh.write(b"\n")
    # raise NotImplementedError
    fh.write(b"$EndEntities\n")


def _write_nodes(fh, points, cells, point_data, float_fmt, binary):
    """Write node information.

    If data on dimension and tags of the geometric entities which the nodes belong to
    is available available, the nodes will be grouped accordingly. This data is
    specified as point_data, using the key 'gmsh:dim_tags' and data as an
    num_points x 2 numpy array (first column is the dimension of the geometric entity
    of this node, second is the tag).

    If dim_tags are not available, all nodes will be assigned the same tag of 0. This
    only makes sense if a single cell block is present in the mesh; an error will be
    raised if len(cells) > 1.

    """
    if points.shape[1] == 2:
        # msh4 requires 3D points, but 2D points given.
        # Appending 0 third component.
        points = np.column_stack([points, np.zeros_like(points[:, 0])])

    fh.write(b"$Nodes\n")

    # The format for the nodes section is
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
    min_tag = 1
    max_tag = n
    is_parametric = 0

    # If node (entity) tag and dimension is available, we make a list of unique
    # combinations thereof, and a map from the full node set to the unique
    # set.
    if "gmsh:dim_tags" in point_data:
        # reverse_index_map maps from all nodes to their respective representation in
        # (the uniquified) node_dim_tags. This approach works for general orderings of
        # the nodes
        node_dim_tags, reverse_index_map = np.unique(
            point_data["gmsh:dim_tags"],
            axis=0,
            return_inverse=True,
        )
    else:
        # If entity information is not provided, we will assign the same entity for all
        # nodes. This only makes sense if the cells are of a single type
        if len(cells) != 1:
            raise WriteError(
                "Specify entity information (gmsh:dim_tags in point_data) "
                + "to deal with more than one cell type. "
            )

        dim = cells[0].dim
        tag = 0
        node_dim_tags = np.array([[dim, tag]])
        # All nodes map to the (single) dimension-entity object
        reverse_index_map = np.full(n, 0, dtype=int)

    num_blocks = node_dim_tags.shape[0]

    # First write preamble
    if binary:
        if points.dtype != c_double:
            warn(f"Binary Gmsh needs c_double points (got {points.dtype}). Converting.")
            points = points.astype(c_double)
        np.array([num_blocks, n, min_tag, max_tag], dtype=c_size_t).tofile(fh)
    else:
        fh.write(f"{num_blocks} {n} {min_tag} {max_tag}\n".encode())

    for j in range(num_blocks):
        dim, tag = node_dim_tags[j]

        node_tags = np.where(reverse_index_map == j)[0]
        num_points_this = node_tags.size

        if binary:
            np.array([dim, tag, is_parametric], dtype=c_int).tofile(fh)
            np.array([num_points_this], dtype=c_size_t).tofile(fh)
            (node_tags + 1).astype(c_size_t).tofile(fh)
            points[node_tags].tofile(fh)
        else:
            fh.write(f"{dim} {tag} {is_parametric} {num_points_this}\n".encode())
            (node_tags + 1).astype(c_size_t).tofile(fh, "\n", "%d")
            fh.write(b"\n")
            np.savetxt(fh, points[node_tags], delimiter=" ", fmt="%" + float_fmt)

    if binary:
        fh.write(b"\n")

    fh.write(b"$EndNodes\n")


def _write_elements(fh, cells, tag_data, binary: bool) -> None:
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

    total_num_cells = sum(len(c) for c in cells)
    num_blocks = len(cells)
    min_element_tag = 1
    max_element_tag = total_num_cells
    if binary:
        np.array(
            [num_blocks, total_num_cells, min_element_tag, max_element_tag],
            dtype=c_size_t,
        ).tofile(fh)

        tag0 = 1
        for ci, cell_block in enumerate(cells):
            node_idcs = _meshio_to_gmsh_order(cell_block.type, cell_block.data)
            if node_idcs.dtype != c_size_t:
                # Binary Gmsh needs c_size_t. Converting."
                node_idcs = node_idcs.astype(c_size_t)

            # entityDim(int) entityTag(int) elementType(int)
            # numElementsBlock(size_t)

            # The entity tag should be equal within a CellBlock
            if "gmsh:geometrical" in tag_data:
                entity_tag = tag_data["gmsh:geometrical"][ci][0]
            else:
                entity_tag = 0

            cell_type = _meshio_to_gmsh_type[cell_block.type]
            np.array([cell_block.dim, entity_tag, cell_type], dtype=c_int).tofile(fh)
            n = node_idcs.shape[0]
            np.array([n], dtype=c_size_t).tofile(fh)

            if node_idcs.dtype != c_size_t:
                warn(
                    f"Binary Gmsh cells need c_size_t (got {node_idcs.dtype}). "
                    + "Converting."
                )
                node_idcs = node_idcs.astype(c_size_t)

            np.column_stack(
                [
                    np.arange(tag0, tag0 + n, dtype=c_size_t),
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
            ).encode()
        )

        tag0 = 1
        for ci, cell_block in enumerate(cells):
            node_idcs = _meshio_to_gmsh_order(cell_block.type, cell_block.data)

            # entityDim(int) entityTag(int) elementType(int) numElementsBlock(size_t)

            # The entity tag should be equal within a CellBlock
            if "gmsh:geometrical" in tag_data:
                entity_tag = tag_data["gmsh:geometrical"][ci][0]
            else:
                entity_tag = 0

            cell_type = _meshio_to_gmsh_type[cell_block.type]
            n = len(cell_block.data)
            fh.write(f"{cell_block.dim} {entity_tag} {cell_type} {n}\n".encode())
            np.savetxt(
                fh,
                # Gmsh indexes from 1 not 0
                np.column_stack([np.arange(tag0, tag0 + n), node_idcs + 1]),
                "%d",
                " ",
            )
            tag0 += n

    fh.write(b"$EndElements\n")


def _write_periodic(fh, periodic, float_fmt: str, binary: bool) -> None:
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
        ary = np.array(value, dtype=dtype)
        if binary:
            ary.tofile(fh)
        else:
            ary = np.atleast_2d(ary)
            fmt = float_fmt if dtype == c_double else "d"
            fmt = "%" + kwargs.pop("fmt", fmt)
            np.savetxt(fh, ary, fmt=fmt, **kwargs)

    fh.write(b"$Periodic\n")
    tofile(fh, len(periodic), c_size_t)
    for dim, (stag, mtag), affine, slave_master in periodic:
        tofile(fh, [dim, stag, mtag], c_int)
        if affine is None or len(affine) == 0:
            tofile(fh, 0, c_size_t)
        else:
            tofile(fh, len(affine), c_size_t, newline=" ")
            tofile(fh, affine, c_double, fmt=float_fmt)
        slave_master = np.array(slave_master, dtype=c_size_t)
        slave_master = slave_master.reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 1-based
        tofile(fh, len(slave_master), c_size_t)
        tofile(fh, slave_master, c_size_t)
    if binary:
        fh.write(b"\n")
    fh.write(b"$EndPeriodic\n")
