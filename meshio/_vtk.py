"""
I/O for VTK <https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf>.
"""
import logging
from functools import reduce

import numpy

from .__about__ import __version__
from ._common import raw_from_cell_data
from ._mesh import Mesh

# https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
vtk_to_meshio_type = {
    0: "empty",
    1: "vertex",
    # 2: 'poly_vertex',
    3: "line",
    # 4: 'poly_line',
    5: "triangle",
    # 6: 'triangle_strip',
    7: "polygon",
    # 8: 'pixel',
    9: "quad",
    10: "tetra",
    # 11: 'voxel',
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
    15: "penta_prism",
    16: "hexa_prism",
    21: "line3",
    22: "triangle6",
    23: "quad8",
    24: "tetra10",
    25: "hexahedron20",
    26: "wedge15",
    27: "pyramid13",
    28: "quad9",
    29: "hexahedron27",
    30: "quad6",
    31: "wedge12",
    32: "wedge18",
    33: "hexahedron24",
    34: "triangle7",
    35: "line4",
    #
    # 60: VTK_HIGHER_ORDER_EDGE,
    # 61: VTK_HIGHER_ORDER_TRIANGLE,
    # 62: VTK_HIGHER_ORDER_QUAD,
    # 63: VTK_HIGHER_ORDER_POLYGON,
    # 64: VTK_HIGHER_ORDER_TETRAHEDRON,
    # 65: VTK_HIGHER_ORDER_WEDGE,
    # 66: VTK_HIGHER_ORDER_PYRAMID,
    # 67: VTK_HIGHER_ORDER_HEXAHEDRON,
}
meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}

vtk_type_to_numnodes = {
    0: 0,  # empty
    1: 1,  # vertex
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quad
    10: 4,  # tetra
    12: 8,  # hexahedron
    13: 6,  # wedge
    14: 5,  # pyramid
    15: 10,  # penta_prism
    16: 12,  # hexa_prism
    21: 3,  # line3
    22: 6,  # triangle6
    23: 8,  # quad8
    24: 10,  # tetra10
    25: 20,  # hexahedron20
    26: 15,  # wedge15
    27: 13,  # pyramid13
    28: 9,  # quad9
    29: 27,  # hexahedron27
    30: 6,  # quad6
    31: 12,  # wedge12
    32: 18,  # wedge18
    33: 24,  # hexahedron24
    34: 7,  # triangle7
    35: 4,  # line4
}


# These are all VTK data types. One sometimes finds 'vtktypeint64', but
# this is ill-formed.
vtk_to_numpy_dtype_name = {
    "bit": "bool",
    "unsigned_char": "uint8",
    "char": "int8",
    "unsigned_short": "uint16",
    "short": "int16",
    "unsigned_int": "uint32",
    "int": "int32",
    "unsigned_long": "int64",
    "long": "int64",
    "float": "float32",
    "double": "float64",
}

numpy_to_vtk_dtype = {v: k for k, v in vtk_to_numpy_dtype_name.items()}

# supported vtk dataset types
vtk_dataset_types = [
    "UNSTRUCTURED_GRID",
    "STRUCTURED_POINTS",
    "STRUCTURED_GRID",
    "RECTILINEAR_GRID",
]
# additional infos per dataset type
vtk_dataset_infos = {
    "UNSTRUCTURED_GRID": [],
    "STRUCTURED_POINTS": [
        "DIMENSIONS",
        "ORIGIN",
        "SPACING",
        "ASPECT_RATIO",  # alternative for SPACING in version 1.0 and 2.0
    ],
    "STRUCTURED_GRID": ["DIMENSIONS"],
    "RECTILINEAR_GRID": [
        "DIMENSIONS",
        "X_COORDINATES",
        "Y_COORDINATES",
        "Z_COORDINATES",
    ],
}

# all main sections in vtk
vtk_sections = [
    "METADATA",
    "DATASET",
    "POINTS",
    "CELLS",
    "CELL_TYPES",
    "POINT_DATA",
    "CELL_DATA",
    "LOOKUP_TABLE",
]


class Info(object):
    """Info Container for the VTK reader."""

    def __init__(self):
        self.points = None
        self.field_data = {}
        self.cell_data_raw = {}
        self.point_data = {}
        self.dataset = {}
        self.c = None
        self.ct = None
        self.active = None
        self.is_ascii = False
        self.split = []
        self.num_items = 0
        # One of the problem in reading VTK files are POINT_DATA and CELL_DATA fields. They
        # can contain a number of SCALARS+LOOKUP_TABLE tables, without giving and indication
        # of how many there are. Hence, SCALARS must be treated like a first-class section.
        # To associate it with POINT/CELL_DATA, we store the `active` section in this
        # variable.
        self.section = None


def read(filename):
    """Reads a VTK vtk file.
    """
    with open(filename, "rb") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # initialize output data
    info = Info()

    # skip header and title
    f.readline()
    f.readline()

    data_type = f.readline().decode("utf-8").strip().upper()
    assert data_type in ["ASCII", "BINARY"], "Unknown VTK data type '{}'.".format(
        data_type
    )
    info.is_ascii = data_type == "ASCII"

    while True:
        line = f.readline().decode("utf-8")
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0:
            continue

        info.split = line.split()
        info.section = info.split[0].upper()

        if info.section in vtk_sections:
            _read_section(f, info)
        else:
            _read_sub_section(f, info)

    _check_mesh(info)
    cells, cell_data = translate_cells(info.c, info.ct, info.cell_data_raw)

    return Mesh(
        info.points,
        cells,
        point_data=info.point_data,
        cell_data=cell_data,
        field_data=info.field_data,
    )


def _read_section(f, info):
    if info.section == "METADATA":
        _skip_meta(f)

    elif info.section == "DATASET":
        info.active = "DATASET"
        info.dataset["type"] = info.split[1].upper()
        assert (
            info.dataset["type"] in vtk_dataset_types
        ), "Only VTK '{}' supported (not {}).".format(
            "', '".join(vtk_dataset_types), info.dataset["type"]
        )

    elif info.section == "POINTS":
        info.active = "POINTS"
        info.num_points = int(info.split[1])
        data_type = info.split[2].lower()
        info.points = _read_points(f, data_type, info.is_ascii, info.num_points)

    elif info.section == "CELLS":
        info.active = "CELLS"
        info.num_items = int(info.split[2])
        info.c = _read_cells(f, info.is_ascii, info.num_items)

    elif info.section == "CELL_TYPES":
        info.active = "CELL_TYPES"
        info.num_items = int(info.split[1])
        info.ct = _read_cell_types(f, info.is_ascii, info.num_items)

    elif info.section == "POINT_DATA":
        info.active = "POINT_DATA"
        info.num_items = int(info.split[1])

    elif info.section == "CELL_DATA":
        info.active = "CELL_DATA"
        info.num_items = int(info.split[1])

    elif info.section == "LOOKUP_TABLE":
        info.num_items = int(info.split[2])
        data = numpy.fromfile(f, count=info.num_items * 4, sep=" ", dtype=float)
        rgba = data.reshape((info.num_items, 4))  # noqa F841


def _read_sub_section(f, info):
    if info.active == "POINT_DATA":
        d = info.point_data
    elif info.active == "CELL_DATA":
        d = info.cell_data_raw
    elif info.active == "DATASET":
        d = info.dataset
    else:
        d = info.field_data

    if info.section in vtk_dataset_infos[info.dataset["type"]]:
        if info.section[1:] == "_COORDINATES":
            info.num_points = int(info.split[1])
            data_type = info.split[2].lower()
            d[info.section] = _read_coords(f, data_type, info.is_ascii, info.num_points)
        else:
            if info.section == "DIMENSIONS":
                d[info.section] = list(map(int, info.split[1:]))
            else:
                d[info.section] = list(map(float, info.split[1:]))
            assert (
                len(d[info.section]) == 3
            ), "Wrong number of info in section '{}'. Need 3, got {}.".format(
                info.section, len(d[info.section])
            )
    elif info.section == "SCALARS":
        d.update(_read_scalar_field(f, info.num_items, info.split))
    elif info.section == "VECTORS":
        d.update(_read_field(f, info.num_items, info.split, [3]))
    elif info.section == "TENSORS":
        d.update(_read_field(f, info.num_items, info.split, [3, 3]))
    else:
        assert info.section == "FIELD", "Unknown section '{}'.".format(info.section)
        d.update(_read_fields(f, int(info.split[2]), info.is_ascii))


def _check_mesh(info):
    if info.dataset["type"] == "UNSTRUCTURED_GRID":
        assert info.c is not None, "Required section CELLS not found."
        assert info.ct is not None, "Required section CELL_TYPES not found."
    elif info.dataset["type"] == "STRUCTURED_POINTS":
        dim = info.dataset["DIMENSIONS"]
        ori = info.dataset["ORIGIN"]
        spa = (
            info.dataset["SPACING"]
            if "SPACING" in info.dataset
            else info.dataset["ASPECT_RATIO"]
        )
        axis = [
            numpy.linspace(ori[i], ori[i] + (dim[i] - 1.0) * spa[i], dim[i])
            for i in range(3)
        ]
        info.points = _generate_points(axis)
        info.c, info.ct = _generate_cells(dim=info.dataset["DIMENSIONS"])
    elif info.dataset["type"] == "RECTILINEAR_GRID":
        axis = [
            info.dataset["X_COORDINATES"],
            info.dataset["Y_COORDINATES"],
            info.dataset["Z_COORDINATES"],
        ]
        info.points = _generate_points(axis)
        info.c, info.ct = _generate_cells(dim=info.dataset["DIMENSIONS"])
    elif info.dataset["type"] == "STRUCTURED_GRID":
        info.c, info.ct = _generate_cells(dim=info.dataset["DIMENSIONS"])


def _generate_cells(dim):
    ele_dim = [d - 1 for d in dim if d > 1]
    ele_no = numpy.prod(ele_dim, dtype=int)
    spatial_dim = len(ele_dim)

    if spatial_dim == 1:
        # cells are lines in 1D
        cells = numpy.empty((ele_no, 3), dtype=int)
        cells[:, 0] = 2
        cells[:, 1] = numpy.arange(ele_no, dtype=int)
        cells[:, 2] = cells[:, 1] + 1
        cell_types = numpy.full(ele_no, 3, dtype=int)
    elif spatial_dim == 2:
        # cells are quad in 2D
        cells = numpy.empty((ele_no, 5), dtype=int)
        cells[:, 0] = 4
        cells[:, 1] = numpy.arange(0, ele_no, dtype=int)
        cells[:, 1] += numpy.arange(0, ele_no, dtype=int) // ele_dim[0]
        cells[:, 2] = cells[:, 1] + 1
        cells[:, 3] = cells[:, 1] + 2 + ele_dim[0]
        cells[:, 4] = cells[:, 3] - 1
        cell_types = numpy.full(ele_no, 9, dtype=int)
    else:
        # cells are hex in 3D
        cells = numpy.empty((ele_no, 9), dtype=int)
        cells[:, 0] = 8
        cells[:, 1] = numpy.arange(ele_no)
        cells[:, 1] += (ele_dim[0] + ele_dim[1] + 1) * (
            numpy.arange(ele_no) // (ele_dim[0] * ele_dim[1])
        )
        cells[:, 1] += (numpy.arange(ele_no) % (ele_dim[0] * ele_dim[1])) // ele_dim[0]
        cells[:, 2] = cells[:, 1] + 1
        cells[:, 3] = cells[:, 1] + 2 + ele_dim[0]
        cells[:, 4] = cells[:, 3] - 1
        cells[:, 5] = cells[:, 1] + (1 + ele_dim[0]) * (1 + ele_dim[1])
        cells[:, 6] = cells[:, 5] + 1
        cells[:, 7] = cells[:, 5] + 2 + ele_dim[0]
        cells[:, 8] = cells[:, 7] - 1
        cell_types = numpy.full(ele_no, 12, dtype=int)

    return cells.reshape(-1), cell_types


def _generate_points(axis):
    x_dim = len(axis[0])
    y_dim = len(axis[1])
    z_dim = len(axis[2])
    pnt_no = x_dim * y_dim * z_dim
    x_id, y_id, z_id = numpy.mgrid[0:x_dim, 0:y_dim, 0:z_dim]
    points = numpy.empty((pnt_no, 3), dtype=axis[0].dtype)
    # VTK sorts points and cells in Fortran order
    points[:, 0] = axis[0][x_id.reshape(-1, order="F")]
    points[:, 1] = axis[1][y_id.reshape(-1, order="F")]
    points[:, 2] = axis[2][z_id.reshape(-1, order="F")]
    return points


def _read_coords(f, data_type, is_ascii, num_points):
    dtype = numpy.dtype(vtk_to_numpy_dtype_name[data_type])
    if is_ascii:
        coords = numpy.fromfile(f, count=num_points, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        coords = numpy.fromfile(f, count=num_points, dtype=dtype)
        line = f.readline().decode("utf-8")
        assert line == "\n"

    return coords


def _read_points(f, data_type, is_ascii, num_points):
    dtype = numpy.dtype(vtk_to_numpy_dtype_name[data_type])
    if is_ascii:
        points = numpy.fromfile(f, count=num_points * 3, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        points = numpy.fromfile(f, count=num_points * 3, dtype=dtype)
        line = f.readline().decode("utf-8")
        assert line == "\n"

    return points.reshape((num_points, 3))


def _read_cells(f, is_ascii, num_items):
    if is_ascii:
        c = numpy.fromfile(f, count=num_items, sep=" ", dtype=int)
    else:
        c = numpy.fromfile(f, count=num_items, dtype=">i4")
        line = f.readline().decode("utf-8")
        assert line == "\n"

    return c


def _read_cell_types(f, is_ascii, num_items):
    if is_ascii:
        ct = numpy.fromfile(f, count=int(num_items), sep=" ", dtype=int)
    else:
        # binary
        ct = numpy.fromfile(f, count=int(num_items), dtype=">i4")
        line = f.readline().decode("utf-8")
        # Sometimes, there's no newline at the end
        assert line.strip() == ""
    return ct


def _read_scalar_field(f, num_data, split):
    data_name = split[1]
    data_type = split[2].lower()
    try:
        num_comp = int(split[3])
    except IndexError:
        num_comp = 1

    # The standard says:
    # > The parameter numComp must range between (1,4) inclusive; [...]
    assert 0 < num_comp < 5

    dtype = numpy.dtype(vtk_to_numpy_dtype_name[data_type])
    lt, _ = f.readline().decode("utf-8").split()
    assert lt.upper() == "LOOKUP_TABLE"
    data = numpy.fromfile(f, count=num_data, sep=" ", dtype=dtype)

    return {data_name: data}


def _read_field(f, num_data, split, shape):
    data_name = split[1]
    data_type = split[2].lower()

    dtype = numpy.dtype(vtk_to_numpy_dtype_name[data_type])
    # <https://stackoverflow.com/q/2104782/353337>
    k = reduce((lambda x, y: x * y), shape)
    data = numpy.fromfile(f, count=k * num_data, sep=" ", dtype=dtype).reshape(
        -1, *shape
    )

    return {data_name: data}


def _read_fields(f, num_fields, is_ascii):
    data = {}
    for _ in range(num_fields):
        line = f.readline().decode("utf-8").split()
        if line[0] == "METADATA":
            _skip_meta(f)
            name, shape0, shape1, data_type = f.readline().decode("utf-8").split()
        else:
            name, shape0, shape1, data_type = line
        shape0 = int(shape0)
        shape1 = int(shape1)
        dtype = numpy.dtype(vtk_to_numpy_dtype_name[data_type.lower()])

        if is_ascii:
            dat = numpy.fromfile(f, count=shape0 * shape1, sep=" ", dtype=dtype)
        else:
            # Binary data is big endian, see
            # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
            dtype = dtype.newbyteorder(">")
            dat = numpy.fromfile(f, count=shape0 * shape1, dtype=dtype)
            line = f.readline().decode("utf-8")
            assert line == "\n"

        if shape0 != 1:
            dat = dat.reshape((shape1, shape0))

        data[name] = dat

    return data


def _skip_meta(f):
    # skip possible metadata
    # https://vtk.org/doc/nightly/html/IOLegacyInformationFormat.html
    while True:
        line = f.readline().decode("utf-8").strip()
        if not line:
            # end of metadata is a blank line
            break


def translate_cells(data, types, cell_data_raw):
    # https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    bins = {u: numpy.where(types == u)[0] for u in numpy.unique(types)}
    has_polygon = meshio_to_vtk_type["polygon"] in bins

    # Deduct offsets from the cell types. This is much faster than manually
    # going through the data array. Slight disadvantage: This doesn't work for
    # cells with a custom number of points.
    numnodes = numpy.empty(len(types), dtype=int)
    if has_polygon:
        # If some polygons are in the VTK file, loop over the cells
        nbcells = len(types)
        offsets = numpy.empty(len(types), dtype=int)
        offsets[0] = 0
        for idx in range(nbcells - 1):
            numnodes[idx] = data[offsets[idx]]
            offsets[idx + 1] = offsets[idx] + numnodes[idx] + 1
        idx = nbcells - 1
        numnodes[idx] = data[offsets[idx]]
    else:
        for tpe, idx in bins.items():
            numnodes[idx] = vtk_type_to_numnodes[tpe]
        offsets = numpy.cumsum(numnodes + 1) - (numnodes + 1)

    assert numpy.all(numnodes == data[offsets])

    cells = {}
    cell_data = {}
    if has_polygon:
        # TODO: cell_data
        for idx in range(nbcells):
            nbedges = data[offsets[idx]]
            start = offsets[idx] + 1
            end = start + numnodes[idx]
            cell = data[start:end]
            if nbedges == vtk_type_to_numnodes[meshio_to_vtk_type["triangle"]]:
                key = "triangle"
            elif nbedges == vtk_type_to_numnodes[meshio_to_vtk_type["quad"]]:
                key = "quad"
            else:
                key = "polygon" + str(nbedges)
            if key in cells:
                cells[key] = numpy.vstack([cells[key], cell])
            else:
                cells[key] = numpy.reshape(cell, (1, -1))
    else:
        for tpe, b in bins.items():
            meshio_type = vtk_to_meshio_type[tpe]
            n = data[offsets[b[0]]]
            assert (data[offsets[b]] == n).all()
            indices = numpy.add.outer(offsets[b], numpy.arange(1, n + 1))
            cells[meshio_type] = data[indices]
            cell_data[meshio_type] = {
                key: value[b] for key, value in cell_data_raw.items()
            }

    return cells, cell_data


def write(filename, mesh, write_binary=True):
    def pad(array):
        return numpy.pad(array, ((0, 0), (0, 1)), "constant")

    if mesh.points.shape[1] == 2:
        logging.warning(
            "VTK requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = pad(mesh.points)

    if mesh.point_data:
        for name, values in mesh.point_data.items():
            if len(values.shape) == 2 and values.shape[1] == 2:
                logging.warning(
                    "VTK requires 3D vectors, but 2D vectors given. "
                    "Appending 0 third component to {}.".format(name)
                )
                mesh.point_data[name] = pad(values)

    if mesh.cell_data:
        for t, data in mesh.cell_data.items():
            for name, values in data.items():
                if len(values.shape) == 2 and values.shape[1] == 2:
                    logging.warning(
                        "VTK requires 3D vectors, but 2D vectors given. "
                        "Appending 0 third component to {}.".format(name)
                    )
                    mesh.cell_data[t][name] = pad(mesh.cell_data[t][name])

    if not write_binary:
        logging.warning("VTK ASCII files are only meant for debugging.")

    with open(filename, "wb") as f:
        f.write("# vtk DataFile Version 4.2\n".encode("utf-8"))
        f.write("written by meshio v{}\n".format(__version__).encode("utf-8"))
        f.write(("BINARY\n" if write_binary else "ASCII\n").encode("utf-8"))
        f.write("DATASET UNSTRUCTURED_GRID\n".encode("utf-8"))

        # write points and cells
        _write_points(f, mesh.points, write_binary)
        _write_cells(f, mesh.cells, write_binary)

        # write point data
        if mesh.point_data:
            num_points = mesh.points.shape[0]
            f.write("POINT_DATA {}\n".format(num_points).encode("utf-8"))
            _write_field_data(f, mesh.point_data, write_binary)

        # write cell data
        if mesh.cell_data:
            total_num_cells = sum([len(c) for c in mesh.cells.values()])
            cell_data_raw = raw_from_cell_data(mesh.cell_data)
            f.write("CELL_DATA {}\n".format(total_num_cells).encode("utf-8"))
            _write_field_data(f, cell_data_raw, write_binary)

    return


def _write_points(f, points, write_binary):
    f.write(
        "POINTS {} {}\n".format(
            len(points), numpy_to_vtk_dtype[points.dtype.name]
        ).encode("utf-8")
    )

    if write_binary:
        # Binary data must be big endian, see
        # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        points.astype(points.dtype.newbyteorder(">")).tofile(f, sep="")
    else:
        # ascii
        points.tofile(f, sep=" ")
    f.write("\n".encode("utf-8"))
    return


def _write_cells(f, cells, write_binary):
    total_num_cells = sum([len(c) for c in cells.values()])
    total_num_idx = sum([numpy.prod(c.shape) for c in cells.values()])
    # For each cell, the number of nodes is stored
    total_num_idx += total_num_cells
    f.write("CELLS {} {}\n".format(total_num_cells, total_num_idx).encode("utf-8"))
    if write_binary:
        for key in cells:
            n = cells[key].shape[1]
            d = numpy.column_stack([numpy.full(len(cells[key]), n), cells[key]]).astype(
                numpy.dtype(">i4")
            )
            f.write(d.tostring())
        if write_binary:
            f.write("\n".encode("utf-8"))
    else:
        # ascii
        for key in cells:
            n = cells[key].shape[1]
            for cell in cells[key]:
                f.write(
                    (
                        " ".join(
                            ["{}".format(idx) for idx in numpy.concatenate([[n], cell])]
                        )
                        + "\n"
                    ).encode("utf-8")
                )

    # write cell types
    f.write("CELL_TYPES {}\n".format(total_num_cells).encode("utf-8"))
    if write_binary:
        for key in cells:
            if key[:7] == "polygon":
                d = numpy.full(len(cells[key]), meshio_to_vtk_type[key[:7]]).astype(
                    numpy.dtype(">i4")
                )
            else:
                d = numpy.full(len(cells[key]), meshio_to_vtk_type[key]).astype(
                    numpy.dtype(">i4")
                )
            f.write(d.tostring())
        f.write("\n".encode("utf-8"))
    else:
        # ascii
        for key in cells:
            if key[:7] == "polygon":
                for _ in range(len(cells[key])):
                    f.write("{}\n".format(meshio_to_vtk_type[key[:7]]).encode("utf-8"))
            else:
                for _ in range(len(cells[key])):
                    f.write("{}\n".format(meshio_to_vtk_type[key]).encode("utf-8"))
    return


def _write_field_data(f, data, write_binary):
    f.write(("FIELD FieldData {}\n".format(len(data))).encode("utf-8"))
    for name, values in data.items():
        if len(values.shape) == 1:
            num_tuples = values.shape[0]
            num_components = 1
        else:
            assert (
                len(values.shape) == 2
            ), "Only one and two-dimensional field data supported."
            num_tuples = values.shape[0]
            num_components = values.shape[1]

        assert (
            " " not in name
        ), 'VTK doesn\'t support spaces in field names ("{}").'.format(name)

        f.write(
            (
                "{} {} {} {}\n".format(
                    name,
                    num_components,
                    num_tuples,
                    numpy_to_vtk_dtype[values.dtype.name],
                )
            ).encode("utf-8")
        )
        if write_binary:
            values.astype(values.dtype.newbyteorder(">")).tofile(f, sep="")
        else:
            # ascii
            values.tofile(f, sep=" ")
            # numpy.savetxt(f, points)
        f.write("\n".encode("utf-8"))
    return
