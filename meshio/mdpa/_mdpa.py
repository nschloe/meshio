"""
I/O for KratosMultiphysics's mdpa format, cf.
<https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.

The MDPA format is unsuitable for fast consumption, this is why:
<https://github.com/KratosMultiphysics/Kratos/issues/5365>.
"""
import logging

import numpy

from .._common import num_nodes_per_cell, raw_from_cell_data
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import CellBlock, Mesh

## We check if we can read/write the mesh natively from Kratos
# TODO: Implement native reading

# Translate meshio types to KratosMultiphysics codes
# Kratos uses the same node numbering of GiD pre and post processor
# http://www-opale.inrialpes.fr/Aerochina/info/en/html-version/gid_11.html
# https://github.com/KratosMultiphysics/Kratos/wiki/Mesh-node-ordering
_mdpa_to_meshio_type = {
    "Line2D2": "line",
    "Line3D2": "line",
    "Triangle2D3": "triangle",
    "Triangle3D3": "triangle",
    "Quadrilateral2D4": "quad",
    "Quadrilateral3D4": "quad",
    "Tetrahedra3D4": "tetra",
    "Hexahedra3D8": "hexahedron",
    "Prism3D6": "wedge",
    "Line2D3": "line3",
    "Triangle2D6": "triangle6",
    "Triangle3D6": "triangle6",
    "Quadrilateral2D9": "quad9",
    "Quadrilateral3D9": "quad9",
    "Tetrahedra3D10": "tetra10",
    "Hexahedra3D27": "hexahedron27",
    "Point2D": "vertex",
    "Point3D": "vertex",
    "Quadrilateral2D8": "quad8",
    "Quadrilateral3D8": "quad8",
    "Hexahedra3D20": "hexahedron20",
}

_meshio_to_mdpa_type = {
    "line": "Line2D2",
    "triangle": "Triangle2D3",
    "quad": "Quadrilateral2D4",
    "tetra": "Tetrahedra3D4",
    "hexahedron": "Hexahedra3D8",
    "wedge": "Prism3D6",
    "line3": "Line2D3",
    "triangle6": "Triangle2D6",
    "quad9": "Quadrilateral2D9",
    "tetra10": "Tetrahedra3D10",
    "hexahedron27": "Hexahedra3D27",
    "vertex": "Point2D",
    "quad8": "Quadrilateral2D8",
    "hexahedron20": "Hexahedra3D20",
}
inverse_num_nodes_per_cell = {v: k for k, v in num_nodes_per_cell.items()}

local_dimension_types = {
    "Line2D2": 1,
    "Line3D2": 1,
    "Triangle2D3": 2,
    "Triangle3D3": 2,
    "Quadrilateral2D4": 2,
    "Quadrilateral3D4": 2,
    "Tetrahedra3D4": 3,
    "Hexahedra3D8": 3,
    "Prism3D6": 3,
    "Line2D3": 1,
    "Triangle2D6": 2,
    "Triangle3D6": 2,
    "Quadrilateral2D9": 2,
    "Quadrilateral3D9": 2,
    "Tetrahedra3D10": 3,
    "Hexahedra3D27": 3,
    "Point2D": 0,
    "Point3D": 0,
    "Quadrilateral2D8": 2,
    "Quadrilateral3D8": 2,
    "Hexahedra3D20": 3,
}


def read(filename):
    """Reads a KratosMultiphysics mdpa file."""
    # if (have_kratos is True): # TODO: Implement natively
    # pass
    # else:
    with open_file(filename, "rb") as f:
        mesh = read_buffer(f)
    return mesh


def _read_nodes(f, is_ascii, data_size):
    # Count the number of nodes. This is _extremely_ ugly; we first read the _entire_
    # file until "End Nodes". The crazy thing is that first counting the lines, then
    # skipping back to pos, and using fromfile there is _faster_ than accumulating the
    # points into a list and converting them to a numpy array afterwards. A point count
    # would be _really_ helpful here, but yeah, that's a fallacy of the format.
    # <https://github.com/KratosMultiphysics/Kratos/issues/5365>
    pos = f.tell()
    num_nodes = 0
    while True:
        line = f.readline().decode("utf-8")
        if "End Nodes" in line:
            break
        num_nodes += 1
    f.seek(pos)

    points = numpy.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
    # The first number is the index
    points = points[:, 1:]

    line = f.readline().decode("utf-8")
    if line.strip() != "End Nodes":
        raise ReadError()
    return points


def _read_cells(f, cells, is_ascii, cell_tags, environ=None):
    if not is_ascii:
        raise ReadError()
    # First we try to identify the entity
    t = None
    if environ is not None:
        if "Begin Elements" in environ:
            entity_name = environ.replace("Begin Elements", "")
            for key in _mdpa_to_meshio_type:
                if key in entity_name:
                    t = _mdpa_to_meshio_type[key]
                    break
        elif "Begin Conditions" in environ:
            entity_name = environ.replace("Begin Conditions", "")
            for key in _mdpa_to_meshio_type:
                if key in entity_name:
                    t = _mdpa_to_meshio_type[key]
                    break

    while True:
        line = f.readline().decode("utf-8")
        if "End Elements" in line or "End Conditions" in line:
            break
        # data[0] gives the entity id
        # data[1] gives the property id
        # The rest are the ids of the nodes
        data = [int(k) for k in filter(None, line.split())]
        num_nodes_per_elem = len(data) - 2

        # We use this in case not alternative
        if t is None:
            t = inverse_num_nodes_per_cell[num_nodes_per_elem]

        if len(cells) == 0 or t != cells[-1].type:
            cells.append(CellBlock(t, []))
        # Subtract one to account for the fact that python indices are 0-based.
        cells[-1].data.append(numpy.array(data[-num_nodes_per_elem:]) - 1)

        # Using the property id as tag
        if t not in cell_tags:
            cell_tags[t] = []
        cell_tags[t].append([data[1]])

    # convert to numpy arrays
    for k, c in enumerate(cells):
        cells[k] = CellBlock(c.type, numpy.array(c.data, dtype=int))

    # Cannot convert cell_tags[key] to numpy array: There may be a
    # different number of tags for each cell.

    if line.strip() not in ["End Elements", "End Conditions"]:
        raise ReadError()


def _prepare_cells(cells, cell_tags):
    # Declaring has additional data tag
    has_additional_tag_data = False

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

    # Kratos cells are mostly ordered like VTK, with a few exceptions:
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 16, 19, 18, 17, 12, 13, 14, 15]
        ]
    if "hexahedron27" in cells:
        cells["hexahedron27"] = cells["hexahedron27"][
            :,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                11,
                10,
                9,
                16,
                19,
                18,
                17,
                12,
                13,
                14,
                15,
                22,
                24,
                21,
                23,
                20,
                25,
                26,
            ],
        ]

    return has_additional_tag_data


def _read_data(f, tag, data_dict, data_size, is_ascii, environ=None):
    if not is_ascii:
        raise ReadError()
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

    # Creating data
    data = numpy.fromfile(f, count=num_items * (1 + num_components), sep=" ").reshape(
        (num_items, 1 + num_components)
    )
    # The first number is the index
    data = data[:, 1:]

    line = f.readline().decode("utf-8")
    if line.strip() != f"End {tag}":
        raise ReadError()

    # The gmsh format cannot distingiush between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data


def read_buffer(f):
    # The format is specified at
    # <https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.

    # Initialize the optional data fields
    points = []
    cells = []
    field_data = {}
    cell_data = {}
    # cell_data_raw = {}
    cell_tags = {}
    point_data = {}

    is_ascii = True
    data_size = None

    # Definition of cell tags
    cell_tags = {}

    # Saving position
    # pos = f.tell()
    # Read mesh
    while True:
        line = f.readline().decode("utf-8")
        if not line:
            # EOF
            break
        environ = line.strip()

        if "Begin Nodes" in environ:
            points = _read_nodes(f, is_ascii, data_size)
        elif "Begin Elements" in environ or "Begin Conditions" in environ:
            _read_cells(f, cells, is_ascii, cell_tags, environ)

    # We finally prepare the cells
    has_additional_tag_data = _prepare_cells(cells, cell_tags)

    # Reverting to the original position
    # f.seek(pos)
    # Read data
    # TODO: To implement
    # while False:
    #     line = f.readline().decode("utf-8")
    #     if not line:
    #         # EOF
    #         break
    #     # elif "NodalData" in environ and cells_prepared:
    #     # _read_data(f, "NodalData", point_data, data_size, is_ascii)
    #     # elif "Begin ElementalData" in environ:
    #     # _read_data(f, "ElementalData", cell_data_raw, data_size, is_ascii)
    #     # elif "Begin ConditionalData" in environ:
    #     # _read_data(f, "ConditionalData", cell_data_raw, data_size, is_ascii)

    if has_additional_tag_data:
        logging.warning("The file contains tag data that couldn't be processed.")

    # cell_data = cell_data_from_raw(cells, cell_data_raw)

    ## Merge cell_tags into cell_data
    # for key, tag_dict in cell_tags.items():
    # if key not in cell_data:
    # cell_data[key] = {}
    # for name, item_list in tag_dict.items():
    # assert name not in cell_data[key]
    # cell_data[key][name] = item_list

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def cell_data_from_raw(cells, cell_data_raw):
    cell_data = {k: {} for k in cells}
    for key in cell_data_raw:
        d = cell_data_raw[key]
        r = 0
        for k in cells:
            cell_data[k][key] = d[r : r + len(cells[k])]
            r += len(cells[k])

    return cell_data


def _write_nodes(fh, points, float_fmt, binary=False):
    fh.write(b"Begin Nodes\n")
    if binary:
        raise WriteError()

    for k, x in enumerate(points):
        fmt = " {} " + " ".join(3 * ["{:" + float_fmt + "}"]) + "\n"
        fh.write(fmt.format(k + 1, x[0], x[1], x[2]).encode("utf-8"))
    fh.write(b"End Nodes\n\n")


def _write_elements_and_conditions(fh, cells, tag_data, binary=False, dimension=2):
    if binary:
        raise WriteError()
    # write elements
    entity = "Elements"
    dimension_name = str(dimension) + "D"
    wrong_dimension_name = "3D" if dimension == 2 else "2D"
    consecutive_index = 0
    aux_cell_type = None
    for cell_type, node_idcs in cells:
        # NOTE: The names of the dummy conditions are not regular, require extra work
        # local_dimension = local_dimension_types[cell_type]
        # if (local_dimension < dimension):
        # entity = "Conditions"

        if aux_cell_type is None:
            aux_cell_type = cell_type
            fh.write(
                (
                    "Begin "
                    + entity
                    + " "
                    + _meshio_to_mdpa_type[cell_type].replace(
                        wrong_dimension_name, dimension_name
                    )
                    + "\n"
                ).encode("utf-8")
            )
        elif aux_cell_type != cell_type:
            fh.write(("End " + entity + "\n\n").encode("utf-8"))
            fh.write(
                (
                    "Begin "
                    + entity
                    + " "
                    + _meshio_to_mdpa_type[cell_type].replace(
                        wrong_dimension_name, dimension_name
                    )
                    + "\n"
                ).encode("utf-8")
            )

        # TODO: Add proper tag recognition in the future
        fcd = numpy.empty((len(node_idcs), 0), dtype=numpy.int32)

        form = "{} " + str(fcd.shape[1]) + " {} {}\n"
        for k, c in enumerate(node_idcs):
            fh.write(
                form.format(
                    " " + str(consecutive_index + k + 1),
                    " ".join([str(val) for val in fcd[k]]),
                    " ".join([str(cc) for cc in c + 1]),
                ).encode("utf-8")
            )

        consecutive_index += len(node_idcs)

    fh.write(b"End Elements\n\n")


def _write_data(fh, tag, name, data, binary):
    if binary:
        raise WriteError()
    fh.write(("Begin " + tag + " " + name + "\n\n").encode("utf-8"))
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1

    # Cut off the last dimension in case it's 1. This avoids problems with
    # writing the data.
    if len(data.shape) > 1 and data.shape[1] == 1:
        data = data[:, 0]

    # Actually write the data
    fmt = " ".join(["{}"] + ["{!r}"] * num_components) + "\n"
    # TODO unify
    if num_components == 1:
        for k, x in enumerate(data):
            fh.write(fmt.format(k + 1, x).encode("utf-8"))
    else:
        for k, x in enumerate(data):
            fh.write(fmt.format(k + 1, *x).encode("utf-8"))

    fh.write(("End " + tag + " " + name + "\n\n").encode("utf-8"))


def write(filename, mesh, float_fmt=".16e", binary=False):
    """Writes mdpa files, cf.
    <https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.
    """
    if binary:
        raise WriteError()
    if mesh.points.shape[1] == 2:
        logging.warning(
            "mdpa requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    # Kratos cells are mostly ordered like VTK, with a few exceptions:
    cells = mesh.cells.copy()
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 16, 17, 18, 19, 12, 15, 14, 13]
        ]
    if "hexahedron27" in cells:
        cells["hexahedron27"] = cells["hexahedron27"][
            :,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                11,
                10,
                9,
                16,
                17,
                18,
                19,
                12,
                15,
                14,
                13,
                22,
                24,
                21,
                23,
                20,
                25,
                26,
            ],
        ]

    with open_file(filename, "wb") as fh:
        # Write some additional info
        fh.write(b"Begin ModelPartData\n")
        fh.write(b"//  VARIABLE_NAME value\n")
        fh.write(b"End ModelPartData\n\n")
        fh.write(b"Begin Properties 0\n")
        fh.write(b"End Properties\n\n")

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

        # We identity which dimension are we
        dimension = 2
        for c in cells:
            name_elem = _meshio_to_mdpa_type[c.type]
            if local_dimension_types[name_elem] == 3:
                dimension = 3
                break

        # We identify the entities
        _write_nodes(fh, mesh.points, float_fmt, binary)
        _write_elements_and_conditions(fh, cells, tag_data, binary, dimension)
        for name, dat in mesh.point_data.items():
            _write_data(fh, "NodalData", name, dat, binary)
        cell_data_raw = raw_from_cell_data(other_data)
        for (
            name,
            dat,
        ) in (
            cell_data_raw.items()
        ):  # NOTE: We will assume always when writing that the components are elements (for now)
            _write_data(fh, "ElementalData", name, dat, binary)


register("mdpa", [".mdpa"], read, {"mdpa": write})
