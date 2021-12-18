"""
I/O for AFLR's UGRID format
[1] <https://www.simcenter.msstate.edu/software/documentation/ug_io/3d_grid_file_type_ugrid.html>
Check out
[2] <https://www.simcenter.msstate.edu/software/public/ug_io/index_simsys_web.php>
for UG_IO C code able to read and convert UGRID files
Node ordering described in
[3] <https://www.simcenter.msstate.edu/software/documentation/ug_io/3d_input_output_grids.html>
"""
import numpy as np

from .._common import _pick_first_int_data, warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

# Float size and endianness are recorded by these suffixes
# binary files come in C-type or FORTRAN type
# https://www.simcenter.msstate.edu/software/documentation/ug_io/ugc_file_formats.html
#
# 64-bit versions described here
# https://www.simcenter.msstate.edu/software/documentation/ug_io/ugc_l_file_formats.html
file_types = {
    "ascii": {"type": "ascii", "float_type": "f", "int_type": "i"},
    "b8l": {"type": "C", "float_type": ">f8", "int_type": ">i8"},
    "b8": {"type": "C", "float_type": ">f8", "int_type": ">i4"},
    "b4": {"type": "C", "float_type": ">f4", "int_type": ">i4"},
    "lb8l": {"type": "C", "float_type": "<f8", "int_type": "<i8"},
    "lb8": {"type": "C", "float_type": "<f8", "int_type": "<i4"},
    "lb4": {"type": "C", "float_type": "<f4", "int_type": "<i4"},
    "r8": {"type": "F", "float_type": ">f8", "int_type": ">i4"},
    "r4": {"type": "F", "float_type": ">f4", "int_type": ">i4"},
    "lr8": {"type": "F", "float_type": "<f8", "int_type": "<i4"},
    "lr4": {"type": "F", "float_type": "<f4", "int_type": "<i4"},
}


def determine_file_type(filename):
    file_type = file_types["ascii"]
    filename_parts = str(filename).split(".")
    if len(filename_parts) > 1:
        type_suffix = filename_parts[-2]
        if type_suffix in file_types.keys():
            file_type = file_types[type_suffix]
    return file_type


def read(filename):
    file_type = determine_file_type(filename)
    with open_file(filename) as f:
        mesh = read_buffer(f, file_type)
    return mesh


def _read_section(f, file_type, count, dtype):
    if file_type["type"] == "ascii":
        return np.fromfile(f, count=count, dtype=dtype, sep=" ")
    return np.fromfile(f, count=count, dtype=dtype)


def read_buffer(f, file_type):
    cells = []
    cell_data = []

    itype = file_type["int_type"]
    ftype = file_type["float_type"]

    # Fortran type includes a number of bytes before and after each record, according to
    # documentation [1] there are two records in the file; see also UG_IO freely
    # available code at [2].
    if file_type["type"] == "F":
        _read_section(f, file_type, count=1, dtype=itype)

    nitems = _read_section(f, file_type, count=7, dtype=itype)

    if file_type["type"] == "F":
        _read_section(f, file_type, count=1, dtype=itype)

    if not nitems.size == 7:
        raise ReadError("Header of ugrid file is ill-formed")

    ugrid_counts = {
        "points": (nitems[0], 3),
        "triangle": (nitems[1], 3),
        "quad": (nitems[2], 4),
        "tetra": (nitems[3], 4),
        "pyramid": (nitems[4], 5),
        "wedge": (nitems[5], 6),
        "hexahedron": (nitems[6], 8),
    }

    if file_type["type"] == "F":
        _read_section(f, file_type, count=1, dtype=itype)

    nnodes = ugrid_counts["points"][0]
    points = _read_section(f, file_type, count=nnodes * 3, dtype=ftype).reshape(
        nnodes, 3
    )

    for key in ["triangle", "quad"]:
        nitems = ugrid_counts[key][0]
        nvertices = ugrid_counts[key][1]
        if nitems == 0:
            continue
        out = _read_section(
            f, file_type, count=nitems * nvertices, dtype=itype
        ).reshape(nitems, nvertices)
        # UGRID is one-based
        cells.append(CellBlock(key, out - 1))

    cell_data = {"ugrid:ref": []}
    for key in ["triangle", "quad"]:
        nitems = ugrid_counts[key][0]
        if nitems == 0:
            continue
        out = _read_section(f, file_type, count=nitems, dtype=itype)
        cell_data["ugrid:ref"].append(out)

    for key in ["tetra", "pyramid", "wedge", "hexahedron"]:
        nitems = ugrid_counts[key][0]
        nvertices = ugrid_counts[key][1]
        if nitems == 0:
            continue
        out = _read_section(
            f, file_type, count=nitems * nvertices, dtype=itype
        ).reshape(nitems, nvertices)

        if key == "pyramid":
            out = out[:, [1, 0, 3, 4, 2]]

        # UGRID is one-based
        cells.append(CellBlock(key, out - 1))

        # fill volume element attributes with zero
        cell_data["ugrid:ref"].append(np.zeros(nitems, dtype=int))

    if file_type["type"] == "F":
        _read_section(f, file_type, count=1, dtype=itype)

    return Mesh(points, cells, cell_data=cell_data)


def _write_section(f, file_type, array, dtype):
    if file_type["type"] == "ascii":
        ncols = array.shape[1]
        fmt = " ".join(["%r"] * ncols)
        np.savetxt(f, array, fmt=fmt)
    else:
        array.astype(dtype).tofile(f)


def write(filename, mesh):
    file_type = determine_file_type(filename)

    with open_file(filename, "w") as f:
        _write_buffer(f, file_type, mesh)


def _write_buffer(f, file_type, mesh):
    itype = file_type["int_type"]
    ftype = file_type["float_type"]

    ugrid_counts = {
        "points": 0,
        "triangle": 0,
        "quad": 0,
        "tetra": 0,
        "pyramid": 0,
        "wedge": 0,
        "hexahedron": 0,
    }
    # ugrid type to cell array id
    ugrid_meshio_id = {
        "points": -1,
        "triangle": -1,
        "quad": -1,
        "tetra": -1,
        "pyramid": -1,
        "wedge": -1,
        "hexahedron": -1,
    }

    ugrid_counts["points"] = mesh.points.shape[0]

    for i, cell_block in enumerate(mesh.cells):
        key = cell_block.type
        data = cell_block.data
        if key in ugrid_counts:
            if ugrid_counts[key] > 0:
                raise ValueError("Ugrid can only handle one cell block of a type.")
            ugrid_counts[key] = data.shape[0]
            ugrid_meshio_id[key] = i
        else:
            msg = f"UGRID mesh format doesn't know {key} cells. Skipping."
            warn(msg)
            continue

    nitems = np.array([list(ugrid_counts.values())])
    # header

    # fortran_header corresponds to the number of bytes in each record
    # it has to be before and after each record
    fortran_header = None
    if file_type["type"] == "F":
        fortran_header = np.array([nitems.nbytes])
        _write_section(f, file_type, fortran_header, itype)

    _write_section(f, file_type, nitems, itype)

    if file_type["type"] == "F":
        # finish current record
        _write_section(f, file_type, fortran_header, itype)

        # start next record
        fortran_header = mesh.points.nbytes
        for cell_block in mesh.cells:
            fortran_header += cell_block.data.nbytes
        # boundary tags
        if ugrid_counts["triangle"] > 0:
            fortran_header += ugrid_counts["triangle"] * np.dtype(itype).itemsize
        if ugrid_counts["quad"] > 0:
            fortran_header += ugrid_counts["quad"] * np.dtype(itype).itemsize
        fortran_header = np.array([fortran_header])
        _write_section(f, file_type, fortran_header, itype)

    _write_section(f, file_type, mesh.points, ftype)

    for key in ["triangle", "quad"]:
        if ugrid_counts[key] == 0:
            continue
        c = mesh.cells[ugrid_meshio_id[key]]
        # UGRID is one-based
        _write_section(f, file_type, c.data + 1, itype)

    # write boundary tags
    for key in ["triangle", "quad"]:
        if ugrid_counts[key] == 0:
            continue

        # pick out cell data
        # for data in mesh.cell_data.values():
        #     if data.dtype in [np.int8, np.int16, np.int32, np.int64]:
        #         labels = data
        #         break

        # pick out cell_data
        labels_key, other = _pick_first_int_data(mesh.cell_data)
        if labels_key and other:
            warn(
                "UGRID can only write one cell data array. "
                f'Picking {labels_key}, skipping {", ".join(other)}.'
            )
        labels = (
            mesh.cell_data[labels_key]
            if labels_key
            else np.ones(ugrid_counts[key], dtype=int)
        )

        labels = labels.reshape(ugrid_counts[key], 1)
        _write_section(f, file_type, labels, itype)

    # write volume elements
    for key in ["tetra", "pyramid", "wedge", "hexahedron"]:
        if ugrid_counts[key] == 0:
            continue
        c = mesh.cells[ugrid_meshio_id[key]]
        # UGRID is one-based
        out = c.data + 1
        if c.type == "pyramid":
            out = out[:, [1, 0, 4, 2, 3]]
        _write_section(f, file_type, out, itype)

    if file_type["type"] == "F":
        _write_section(f, file_type, fortran_header, itype)


register_format("ugrid", [".ugrid"], read, {"ugrid": write})
