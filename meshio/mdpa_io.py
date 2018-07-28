# -*- coding: utf-8 -*-
#
"""
I/O for KratosMultiphysics's mdpa format, cf.
<https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.
"""
import logging
import struct

import numpy

from .mesh import Mesh
from .vtk_io import raw_from_cell_data
from .gmsh_io import num_nodes_per_cell

# We check if we can read/write the mesh natively from Kratos # TODO: Implement natively
try:
    import KratosMultiphysics
    have_kratos = True
except ImportError:
    have_kratos = False

# Translate meshio types to KratosMultiphysics codes
# Kratos uses the same node numbering of GiD pre and post processor
# http://www-opale.inrialpes.fr/Aerochina/info/en/html-version/gid_11.html
# https://github.com/KratosMultiphysics/Kratos/wiki/Mesh-node-ordering
_mdpa_to_meshio_type = {
    1: "line",
    2: "triangle",
    3: "quad",
    4: "tetra",
    5: "hexahedron",
    6: "wedge",
    8: "line3",
    9: "triangle6",
    10: "quad9",
    11: "tetra10",
    12: "hexahedron27",
    15: "vertex",
    16: "quad8",
    17: "hexahedron20"
}
_meshio_to_mdpa_type = {v: k for k, v in _gmsh_to_meshio_type.items()}
#_meshio_to_mdpa_type = {
    #1: "Line2D2",
    #2: "Triangle2D3",
    #3: "Quadrilateral2D4",
    #4: "Tetrahedra3D4",
    #5: "hexahedron",
    #6: "Prism3D6",
    #8: "Line2D3",
    #9: "Triangle2D6",
    #10: "Quadrilateral2D9",
    #11: "Tetrahedra3D10",
    #12: "Hexahedra3D27",
    #15: "Point3D",
    #16: "Quadrilateral2D8",
    #17: "Hexahedra3D20"
#}

def read(filename):
    """Reads a KratosMultiphysics mdpa file.
    """
    with open(filename, "rb") as f:
        mesh = read_buffer(f)
    return mesh

def _convert_string_to_list_float(line, space = " ", endline = ""):
    """This function converts a string into a list of floats
    """
    list_values = []
    string_values = (line.replace(endline,"")).split(space)
    for string in string_values:
        list_values.append(float(string))

    return list_values

def _convert_string_to_list_int_float(line, space = " ", endline = ""):
    """This function converts a string into a id and list of floats
    """
    id = 0
    list_values = []
    string_values = (line.replace(endline,"")).split(space)
    count = 0
    for string in string_values:
        if (count == 0):
            id = int(string)
        else:
            list_values.append(float(string))
        count += 1

    return id, list_values

def _read_nodes(f, is_ascii, int_size, data_size):
    # The first line is the number of nodes
    aux_f = f.copy()
    num_nodes = 0
    known_number_of_nodes = False
    while (known_number_of_nodes == False):
        line = aux_f.readline().decode("utf-8")
        if ("End Nodes" in line):
            known_number_of_nodes = True
        num_nodes += 1
    del(aux_f)

    points = numpy.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
    # The first number is the index
    points = points[:, 1:]

    line = f.readline().decode("utf-8")
    assert line.strip() == "End Nodes"
    return points

def _read_cells(f, cells, int_size, is_ascii):
    # FIrst we compute the number of entities
    aux_f = f.copy()
    total_num_cells = 0
    known_number_of_cells = False
    while (known_number_of_cells == False):
        line = aux_f.readline().decode("utf-8")
        if ("End Elements" in line or "End Conditions" in line):
            known_number_of_cells = True
        total_num_cells += 1
    del(aux_f)
    has_additional_tag_data = False
    cell_tags = {}
    if is_ascii:
        for _ in range(total_num_cells):
            line = f.readline().decode("utf-8")
            # data[0] gives the entity id
            # data[1] gives the property id
            # The rest are the ids of the nodes
            data = [int(k) for k in filter(None, line.split())]
            t = _mdpa_to_meshio_type[len(data) - 2]
            num_nodes_per_elem = num_nodes_per_cell[t]

            if t not in cells:
                cells[t] = []
            cells[t].append(data[-num_nodes_per_elem:])

            # Using the property id as tag
            if t not in cell_tags:
                cell_tags[t] = []
            cell_tags[t].append(data[1])

        # convert to numpy arrays
        for key in cells:
            cells[key] = numpy.array(cells[key], dtype=int)
        # Cannot convert cell_tags[key] to numpy array: There may be a
        # different number of tags for each cell.
    else:
        raise NameError('Only ASCII mdpa supported')

    line = f.readline().decode("utf-8")
    assert line.strip() == "End Elements"
    assert line.strip() == "End Conditions"

    # Subtract one to account for the fact that python indices are
    # 0-based.
    for key in cells:
        cells[key] -= 1

    # restrict to the standard two data items (physical, geometrical)
    output_cell_tags = {}
    for key in cell_tags:
        output_cell_tags[key] = {"mdpa:physical": [], "mdpa:geometrical": []}
        for item in cell_tags[key]:
            if len(item) > 0:
                output_cell_tags[key]["mdpa:physical"].append(item[0])
            if len(item) > 1:
                output_cell_tags[key]["mdpa:geometrical"].append(item[1])
            if len(item) > 2:
                has_additional_tag_data = True
        output_cell_tags[key]["mdpa:physical"] = numpy.array(
            output_cell_tags[key]["mdpa:physical"], dtype=int
        )
        output_cell_tags[key]["mdpa:geometrical"] = numpy.array(
            output_cell_tags[key]["mdpa:geometrical"], dtype=int
        )

    # Kratos cells are mostly ordered like VTK, with a few exceptions:
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]

    return has_additional_tag_data, output_cell_tags

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
        raise NameError('Only ASCII mdpa supported')

    line = f.readline().decode("utf-8")
    assert line.strip() == "End{}".format(tag)

    # The gmsh format cannot distingiush between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data
    return


def read_buffer(f):
    # The format is specified at
    # <https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.

    # Initialize the optional data fields
    points = []
    cells = {}
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}
    periodic = None

    is_ascii = None
    int_size = 4
    data_size = None
    while True:
        line = f.readline().decode("utf-8")
        if not line:
            # EOF
            break
        environ = line.strip()

        if "Begin Nodes" in environ:
            points = _read_nodes(f, is_ascii, int_size, data_size)
        elif "Begin Elements" in environ or "Begin Conditions" in environ :
            has_additional_tag_data, cell_tags = _read_cells(
                f, cells, int_size, is_ascii
            )
        #elif environ == "NodeData":
            #_read_data(f, "NodeData", point_data, int_size, data_size, is_ascii)
        #elif "Begin ElementalData" in environ or "Begin ConditionalData" in environ:
            #_read_data(f, "ElementData", cell_data_raw, int_size, data_size, is_ascii)

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
        field_data=field_data
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

def _write_nodes(fh, points, write_binary):
    fh.write("Begin Nodes\n".encode("utf-8"))
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
    fh.write("End Nodes\n".encode("utf-8"))
    return


def _write_elements(fh, cells, tag_data, write_binary):
    # write elements
    fh.write("Begin Elements\n".encode("utf-8"))
    # count all cells
    total_num_cells = sum([data.shape[0] for _, data in cells.items()])
    fh.write("{}\n".format(total_num_cells).encode("utf-8"))

    consecutive_index = 0
    for cell_type, node_idcs in cells.items():
        tags = []
        for key in ["mdpa:physical", "mdpa:geometrical"]:
            try:
                tags.append(tag_data[cell_type][key])
            except KeyError:
                pass
        fcd = numpy.concatenate([tags]).T

        if len(fcd) == 0:
            fcd = numpy.empty((len(node_idcs), 0), dtype=numpy.int32)

        if write_binary:
            raise NameError('Only ASCII mdpa supported')
        else:
            form = (
                "{} "
                + str(_meshio_to_mdpa_type[cell_type])
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
    fh.write("End Elements\n".encode("utf-8"))
    return

def _write_data(fh, tag, name, data, write_binary):
    fh.write("${}\n".format(tag).encode("utf-8"))
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

    # Cut off the last dimension in case it's 1. This avoids problems with
    # writing the data.
    if len(data.shape) > 1 and data.shape[1] == 1:
        data = data[:, 0]

    fh.write("{}\n".format(num_components).encode("utf-8"))
    # num data items
    fh.write("{}\n".format(data.shape[0]).encode("utf-8"))
    # actually write the data
    if write_binary:
        raise NameError('Only ASCII mdpa supported')
    else:
        fmt = " ".join(["{}"] + ["{!r}"] * num_components) + "\n"
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, x).encode("utf-8"))
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, *x).encode("utf-8"))

    fh.write("End{}\n".format(tag).encode("utf-8"))
    return


def write(filename, mesh, write_binary=False):
    """Writes mdpa files, cf.
    <https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>.
    """
    if write_binary:
        raise NameError('Only ASCII mdpa supported')

    # Kratos cells are mostly ordered like VTK, with a few exceptions:
    cells = mesh.cells.copy()
    if "tetra10" in cells:
        cells["tetra10"] = cells["tetra10"][:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
    if "hexahedron20" in cells:
        cells["hexahedron20"] = cells["hexahedron20"][
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]
        ]

    with open(filename, "wb") as fh:
        # Split the cell data: mdpa:physical and mdpa:geometrical are tags, the
        # rest is actual cell data.
        tag_data = {}
        other_data = {}
        for cell_type, a in mesh.cell_data.items():
            tag_data[cell_type] = {}
            other_data[cell_type] = {}
            for key, data in a.items():
                if key in ["mdpa:physical", "mdpa:geometrical"]:
                    tag_data[cell_type][key] = data.astype(numpy.int32)
                else:
                    other_data[cell_type][key] = data

        _write_nodes(fh, mesh.points, write_binary)
        _write_elements(fh, cells, tag_data, write_binary)
        #for name, dat in mesh.point_data.items():
            #_write_data(fh, "NodalData", name, dat, write_binary)
        #cell_data_raw = raw_from_cell_data(other_data)
        #for name, dat in cell_data_raw.items(): # We will assume always when writing that the components are elements
            #_write_data(fh, "ElementalData", name, dat, write_binary)

    return
