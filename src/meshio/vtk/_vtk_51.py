from functools import reduce

import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._mesh import Mesh
from .._vtk_common import (
    Info,
    meshio_to_vtk_order,
    meshio_to_vtk_type,
    vtk_cells_from_data,
)

# VTK 5.1 data types
vtk_to_numpy_dtype_name = {
    "float": "float32",
    "double": "float64",
    "int": "int",
    "vtktypeint8": "int8",
    "vtktypeint16": "int16",
    "vtktypeint32": "int32",
    "vtktypeint64": "int64",
    "vtktypeuint8": "uint8",
    "vtktypeuint16": "uint16",
    "vtktypeuint32": "uint32",
    "vtktypeuint64": "uint64",
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
    "COLOR_SCALARS",
]


def read(filename):
    with open_file(filename, "rb") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # initialize output data
    info = Info()

    # skip title comment
    f.readline()

    data_type = f.readline().decode().strip().upper()
    if data_type not in ["ASCII", "BINARY"]:
        raise ReadError(f"Unknown VTK data type '{data_type}'.")
    info.is_ascii = data_type == "ASCII"

    while True:
        line = f.readline().decode()
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
            _read_subsection(f, info)

    _check_mesh(info)

    cells, cell_data = vtk_cells_from_data(
        info.connectivity, info.offsets, info.types, info.cell_data_raw
    )

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
        if info.dataset["type"] not in vtk_dataset_types:
            legal = ", ".join(vtk_dataset_types)
            raise ReadError(
                f"Only VTK '{legal}' supported (not {info.dataset['type']})."
            )

    elif info.section == "POINTS":
        info.active = "POINTS"
        info.num_points = int(info.split[1])
        data_type = info.split[2].lower()
        info.points = _read_points(f, data_type, info.is_ascii, info.num_points)

    elif info.section == "CELLS":
        info.active = "CELLS"
        try:
            line = f.readline().decode()
        except UnicodeDecodeError:
            line = ""

        assert line.startswith("OFFSETS")
        # vtk DataFile Version 5.1 - appearing in Paraview 5.8.1 outputs
        # No specification found for this file format.
        # See the question on ParaView Discourse Forum:
        # https://discourse.paraview.org/t/specification-of-vtk-datafile-version-5-1/5127
        info.num_offsets = int(info.split[1])
        info.num_items = int(info.split[2])
        dtype = np.dtype(vtk_to_numpy_dtype_name[line.split()[1]])
        offsets = _read_int_data(f, info.is_ascii, info.num_offsets, dtype)

        line = f.readline().decode()
        assert line.startswith("CONNECTIVITY")
        dtype = np.dtype(vtk_to_numpy_dtype_name[line.split()[1]])
        connectivity = _read_int_data(f, info.is_ascii, info.num_items, dtype)
        info.connectivity = connectivity
        assert offsets[0] == 0
        assert offsets[-1] == len(connectivity)
        info.offsets = offsets[1:]

    elif info.section == "CELL_TYPES":
        info.active = "CELL_TYPES"
        info.num_items = int(info.split[1])
        info.types = _read_cell_types(f, info.is_ascii, info.num_items)

    elif info.section == "POINT_DATA":
        info.active = "POINT_DATA"
        info.num_items = int(info.split[1])

    elif info.section == "CELL_DATA":
        info.active = "CELL_DATA"
        info.num_items = int(info.split[1])

    elif info.section == "LOOKUP_TABLE":
        info.num_items = int(info.split[2])
        np.fromfile(f, count=info.num_items * 4, sep=" ", dtype=float)
        # rgba = data.reshape((info.num_items, 4))

    elif info.section == "COLOR_SCALARS":
        nValues = int(info.split[2])
        # re-use num_items from active POINT/CELL_DATA
        num_items = info.num_items
        dtype = np.ubyte
        if info.is_ascii:
            dtype = float
        np.fromfile(f, count=num_items * nValues, dtype=dtype)


def _read_subsection(f, info):
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
            if len(d[info.section]) != 3:
                raise ReadError(
                    f"Wrong number of info in section '{info.section}'. "
                    f"Need 3, got {len(d[info.section])}."
                )
    elif info.section == "SCALARS":
        d.update(_read_scalar_field(f, info.num_items, info.split, info.is_ascii))
    elif info.section == "VECTORS":
        d.update(_read_field(f, info.num_items, info.split, [3], info.is_ascii))
    elif info.section == "TENSORS":
        d.update(_read_field(f, info.num_items, info.split, [3, 3], info.is_ascii))
    elif info.section == "FIELD":
        d.update(_read_fields(f, int(info.split[2]), info.is_ascii))
    else:
        raise ReadError(f"Unknown section '{info.section}'.")


def _check_mesh(info):
    if info.dataset["type"] == "UNSTRUCTURED_GRID":
        if info.connectivity is None:
            raise ReadError("Required section CELLS not found.")
        if info.types is None:
            raise ReadError("Required section CELL_TYPES not found.")
    elif info.dataset["type"] == "STRUCTURED_POINTS":
        dim = info.dataset["DIMENSIONS"]
        ori = info.dataset["ORIGIN"]
        spa = (
            info.dataset["SPACING"]
            if "SPACING" in info.dataset
            else info.dataset["ASPECT_RATIO"]
        )
        axis = [
            np.linspace(ori[i], ori[i] + (dim[i] - 1.0) * spa[i], dim[i])
            for i in range(3)
        ]
        info.points = _generate_points(axis)
        info.connectivity, info.types = _generate_cells(dim=info.dataset["DIMENSIONS"])
    elif info.dataset["type"] == "RECTILINEAR_GRID":
        axis = [
            info.dataset["X_COORDINATES"],
            info.dataset["Y_COORDINATES"],
            info.dataset["Z_COORDINATES"],
        ]
        info.points = _generate_points(axis)
        info.connectivity, info.types = _generate_cells(dim=info.dataset["DIMENSIONS"])
    elif info.dataset["type"] == "STRUCTURED_GRID":
        info.connectivity, info.types = _generate_cells(dim=info.dataset["DIMENSIONS"])


def _generate_cells(dim):
    ele_dim = [d - 1 for d in dim if d > 1]
    # TODO use math.prod when requiring Python 3.8+? this would save the int conversion
    # <https://github.com/microsoft/pyright/issues/1226>
    ele_no = int(np.prod(ele_dim))
    spatial_dim = len(ele_dim)

    if spatial_dim == 1:
        # cells are lines in 1D
        cells = np.empty((ele_no, 3), dtype=int)
        cells[:, 0] = 2
        cells[:, 1] = np.arange(ele_no, dtype=int)
        cells[:, 2] = cells[:, 1] + 1
        cell_types = np.full(ele_no, 3, dtype=int)
    elif spatial_dim == 2:
        # cells are quad in 2D
        cells = np.empty((ele_no, 5), dtype=int)
        cells[:, 0] = 4
        cells[:, 1] = np.arange(0, ele_no, dtype=int)
        cells[:, 1] += np.arange(0, ele_no, dtype=int) // ele_dim[0]
        cells[:, 2] = cells[:, 1] + 1
        cells[:, 3] = cells[:, 1] + 2 + ele_dim[0]
        cells[:, 4] = cells[:, 3] - 1
        cell_types = np.full(ele_no, 9, dtype=int)
    else:
        # cells are hex in 3D
        cells = np.empty((ele_no, 9), dtype=int)
        cells[:, 0] = 8
        cells[:, 1] = np.arange(ele_no)
        cells[:, 1] += (ele_dim[0] + ele_dim[1] + 1) * (
            np.arange(ele_no) // (ele_dim[0] * ele_dim[1])
        )
        cells[:, 1] += (np.arange(ele_no) % (ele_dim[0] * ele_dim[1])) // ele_dim[0]
        cells[:, 2] = cells[:, 1] + 1
        cells[:, 3] = cells[:, 1] + 2 + ele_dim[0]
        cells[:, 4] = cells[:, 3] - 1
        cells[:, 5] = cells[:, 1] + (1 + ele_dim[0]) * (1 + ele_dim[1])
        cells[:, 6] = cells[:, 5] + 1
        cells[:, 7] = cells[:, 5] + 2 + ele_dim[0]
        cells[:, 8] = cells[:, 7] - 1
        cell_types = np.full(ele_no, 12, dtype=int)

    return cells.reshape(-1), cell_types


def _generate_points(axis):
    x_dim = len(axis[0])
    y_dim = len(axis[1])
    z_dim = len(axis[2])
    pnt_no = x_dim * y_dim * z_dim
    x_id, y_id, z_id = np.mgrid[0:x_dim, 0:y_dim, 0:z_dim]
    points = np.empty((pnt_no, 3), dtype=axis[0].dtype)
    # VTK sorts points and cells in Fortran order
    points[:, 0] = axis[0][x_id.reshape(-1, order="F")]
    points[:, 1] = axis[1][y_id.reshape(-1, order="F")]
    points[:, 2] = axis[2][z_id.reshape(-1, order="F")]
    return points


def _read_coords(f, data_type, is_ascii, num_points):
    dtype = np.dtype(vtk_to_numpy_dtype_name[data_type])
    if is_ascii:
        coords = np.fromfile(f, count=num_points, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        coords = np.fromfile(f, count=num_points, dtype=dtype)
        line = f.readline().decode()
        if line != "\n":
            raise ReadError()
    return coords


def _read_points(f, data_type, is_ascii, num_points):
    dtype = np.dtype(vtk_to_numpy_dtype_name[data_type])
    if is_ascii:
        points = np.fromfile(f, count=num_points * 3, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        points = np.fromfile(f, count=num_points * 3, dtype=dtype)
        line = f.readline().decode()
        if line != "\n":
            raise ReadError()
    return points.reshape((num_points, 3))


def _read_int_data(f, is_ascii, num_items, dtype):
    if is_ascii:
        c = np.fromfile(f, count=num_items, sep=" ", dtype=dtype)
    else:
        dtype = dtype.newbyteorder(">")
        c = np.fromfile(f, count=num_items, dtype=dtype)
        line = f.readline().decode()
        if line != "\n":
            raise ReadError("Expected newline")
    return c


def _read_cell_types(f, is_ascii, num_items):
    if is_ascii:
        ct = np.fromfile(f, count=int(num_items), sep=" ", dtype=int)
    else:
        # binary
        ct = np.fromfile(f, count=int(num_items), dtype=">i4")
        line = f.readline().decode()
        # Sometimes, there's no newline at the end
        if line.strip() != "":
            raise ReadError()
    return ct


def _read_scalar_field(f, num_data, split, is_ascii):
    data_name = split[1]
    data_type = split[2].lower()
    try:
        num_comp = int(split[3])
    except IndexError:
        num_comp = 1

    # The standard says:
    # > The parameter numComp must range between (1,4) inclusive; [...]
    if not (0 < num_comp < 5):
        raise ReadError("The parameter numComp must range between (1,4) inclusive")

    dtype = np.dtype(vtk_to_numpy_dtype_name[data_type])
    lt, _ = f.readline().decode().split()
    if lt.upper() != "LOOKUP_TABLE":
        raise ReadError()

    if is_ascii:
        data = np.fromfile(f, count=num_data * num_comp, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        data = np.fromfile(f, count=num_data * num_comp, dtype=dtype)
        line = f.readline().decode()
        if line != "\n":
            raise ReadError()

    data = data.reshape(-1, num_comp)
    return {data_name: data}


def _read_field(f, num_data, split, shape, is_ascii):
    data_name = split[1]
    data_type = split[2].lower()

    dtype = np.dtype(vtk_to_numpy_dtype_name[data_type])
    # prod()
    # <https://stackoverflow.com/q/2104782/353337>
    k = reduce((lambda x, y: x * y), shape)

    if is_ascii:
        data = np.fromfile(f, count=k * num_data, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        data = np.fromfile(f, count=k * num_data, dtype=dtype)
        line = f.readline().decode()
        if line != "\n":
            raise ReadError()

    data = data.reshape(-1, *shape)
    return {data_name: data}


def _read_fields(f, num_fields, is_ascii):
    data = {}
    for _ in range(num_fields):
        line = f.readline().decode().split()
        if line[0] == "METADATA":
            _skip_meta(f)
            name, shape0, shape1, data_type = f.readline().decode().split()
        else:
            name, shape0, shape1, data_type = line

        shape0 = int(shape0)
        shape1 = int(shape1)
        dtype = np.dtype(vtk_to_numpy_dtype_name[data_type.lower()])

        if is_ascii:
            dat = np.fromfile(f, count=shape0 * shape1, sep=" ", dtype=dtype)
        else:
            # Binary data is big endian, see
            # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
            dtype = dtype.newbyteorder(">")
            dat = np.fromfile(f, count=shape0 * shape1, dtype=dtype)
            line = f.readline().decode()
            if line != "\n":
                raise ReadError()

        if shape0 != 1:
            dat = dat.reshape((shape1, shape0))

        data[name] = dat

    return data


def _skip_meta(f):
    # skip possible metadata
    # https://vtk.org/doc/nightly/html/IOLegacyInformationFormat.html
    while True:
        line = f.readline().decode().strip()
        if not line:
            # end of metadata is a blank line
            break


def _pad(array):
    return np.pad(array, ((0, 0), (0, 1)), "constant")


def write(filename, mesh, binary=True):
    if mesh.points.shape[1] == 2:
        warn(
            "VTK requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = _pad(mesh.points)
    else:
        points = mesh.points

    if mesh.point_data:
        for name, values in mesh.point_data.items():
            if len(values.shape) == 2 and values.shape[1] == 2:
                warn(
                    "VTK requires 3D vectors, but 2D vectors given. "
                    f"Appending 0 third component to {name}."
                )
                mesh.point_data[name] = _pad(values)

    for name, data in mesh.cell_data.items():
        for k, values in enumerate(data):
            if len(values.shape) == 2 and values.shape[1] == 2:
                warn(
                    "VTK requires 3D vectors, but 2D vectors given. "
                    f"Appending 0 third component to {name}."
                )
                data[k] = _pad(data[k])

    if not binary:
        warn("VTK ASCII files are only meant for debugging.")

    if mesh.point_sets or mesh.cell_sets:
        warn(
            "VTK format cannot write sets. "
            + "Consider converting them to \\[point,cell]_data.",
            highlight=False,
        )

    with open_file(filename, "wb") as f:
        f.write(b"# vtk DataFile Version 5.1\n")
        f.write(f"written by meshio v{__version__}\n".encode())
        f.write(("BINARY\n" if binary else "ASCII\n").encode())
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # write points and cells
        _write_points(f, points, binary)
        _write_cells(f, mesh.cells, binary)

        # write point data
        if mesh.point_data:
            num_points = mesh.points.shape[0]
            f.write(f"POINT_DATA {num_points}\n".encode())
            _write_field_data(f, mesh.point_data, binary)

        # write cell data
        if mesh.cell_data:
            total_num_cells = sum(len(c.data) for c in mesh.cells)
            f.write(f"CELL_DATA {total_num_cells}\n".encode())
            _write_field_data(f, mesh.cell_data, binary)


def _write_points(f, points, binary):
    dtype = numpy_to_vtk_dtype[points.dtype.name]
    f.write(f"POINTS {len(points)} {dtype}\n".encode())

    if binary:
        # Binary data must be big endian, see
        # <https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        # if points.dtype.byteorder == "<" or (
        #     points.dtype.byteorder == "=" and sys.byteorder == "little"
        # ):
        #     logging.warn("Converting to new byte order")
        points.astype(points.dtype.newbyteorder(">")).tofile(f, sep="")
    else:
        # ascii
        points.tofile(f, sep=" ")
    f.write(b"\n")


def _write_cells(f, cells, binary):
    total_num_cells = sum(len(c.data) for c in cells)
    total_num_idx = sum(c.data.size for c in cells)
    f.write(f"CELLS {total_num_cells + 1} {total_num_idx}\n".encode())

    # offsets
    offsets = [[0]]
    k = 0
    for cell_block in cells:
        m, n = cell_block.data.shape
        offsets.append(np.arange(k + n, k + (m + 1) * n, n))
        k = offsets[-1][-1]
    offsets = np.concatenate(offsets)

    if binary:
        f.write(b"OFFSETS vtktypeint64\n")
        # force big-endian and int64
        offsets.astype(">i8").tofile(f, sep="")
        f.write(b"\n")

        f.write(b"CONNECTIVITY vtktypeint64\n")
        for cell_block in cells:
            d = cell_block.data
            cell_idx = meshio_to_vtk_order(cell_block.type)
            if cell_idx is not None:
                d = d[:, cell_idx]
            # force big-endian and int64
            d.astype(">i8").tofile(f, sep="")
        f.write(b"\n")
    else:
        # ascii
        f.write(b"OFFSETS vtktypeint64\n")
        offsets.tofile(f, sep="\n")
        f.write(b"\n")

        f.write(b"CONNECTIVITY vtktypeint64\n")
        for cell_block in cells:
            d = cell_block.data
            cell_idx = meshio_to_vtk_order(cell_block.type)
            if cell_idx is not None:
                d = d[:, cell_idx]
            d.tofile(f, sep="\n")
            f.write(b"\n")

    # write cell types
    f.write(f"CELL_TYPES {total_num_cells}\n".encode())
    if binary:
        for c in cells:
            vtk_type = meshio_to_vtk_type[c.type]
            np.full(len(c.data), vtk_type, dtype=np.dtype(">i4")).tofile(f, sep="")
        f.write(b"\n")
    else:
        # ascii
        for c in cells:
            vtk_type = meshio_to_vtk_type[c.type]
            np.full(len(c.data), vtk_type).tofile(f, sep="\n")
            f.write(b"\n")


def _write_field_data(f, data, binary):
    f.write((f"FIELD FieldData {len(data)}\n").encode())
    for name, values in data.items():
        if isinstance(values, list):
            values = np.concatenate(values)
        if len(values.shape) == 1:
            num_tuples = values.shape[0]
            num_components = 1
        else:
            num_tuples = values.shape[0]
            num_components = values.shape[1]

        if " " in name:
            raise WriteError(f"VTK doesn't support spaces in field names ('{name}').")

        vtk_dtype = numpy_to_vtk_dtype[values.dtype.name]
        f.write(f"{name} {num_components} {num_tuples} {vtk_dtype}\n".encode())
        if binary:
            values.astype(values.dtype.newbyteorder(">")).tofile(f, sep="")
        else:
            # ascii
            values.tofile(f, sep=" ")
            # np.savetxt(f, points)
        f.write(b"\n")
