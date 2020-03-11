"""
I/O for the PLY format, cf.
<https://en.wikipedia.org/wiki/PLY_(file_format)>.
<https://web.archive.org/web/20161221115231/http://www.cs.virginia.edu/~gfx/Courses/2001/Advanced.spring.01/plylib/Ply.txt>.
"""
import re
import sys
import warnings

import numpy

from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import CellBlock, Mesh

# Reference dtypes
ply_to_numpy_dtype = {
    "short": numpy.int16,
    "ushort": numpy.uint16,
    "int": numpy.int32,
    "int8": numpy.int8,
    "int32": numpy.int32,
    "int64": numpy.int64,
    "uint": numpy.uint32,
    "uint8": numpy.uint8,
    "uint16": numpy.uint16,
    "uint32": numpy.uint32,
    "uint64": numpy.uint64,
    "float": numpy.float32,
    "float32": numpy.float32,
    "float64": numpy.float64,
    "double": numpy.float64,
}
numpy_to_ply_dtype = {numpy.dtype(v): k for k, v in ply_to_numpy_dtype.items()}


def read(filename):
    with open_file(filename, "rb") as f:
        mesh = read_buffer(f)
    return mesh


def _fast_forward(f):
    # fast forward to the next significant line
    while True:
        line = f.readline().decode("utf-8").strip()
        if line and line[:7] != "comment":
            break
    return line


def read_buffer(f):
    # assert that the first line reads `ply`
    line = f.readline().decode("utf-8").strip()
    if line != "ply":
        raise ReadError()

    line = _fast_forward(f)
    if line == "format ascii 1.0":
        is_binary = False
    elif line == "format binary_big_endian 1.0":
        is_binary = True
        endianness = ">"
    else:
        if line != "format binary_little_endian 1.0":
            raise ReadError()
        is_binary = True
        endianness = "<"

    line = _fast_forward(f)
    m = re.match("element vertex (\\d+)", line)
    num_verts = int(m.groups()[0])

    # read point data
    point_data_formats = []
    point_data_names = []
    line = _fast_forward(f)
    while line[:8] == "property":
        m = re.match("property (.+) (.+)", line)
        point_data_formats.append(m.groups()[0])
        point_data_names.append(m.groups()[1])
        line = _fast_forward(f)

    m = re.match("element face (\\d+)", line)
    num_cells = int(m.groups()[0])

    assert num_cells > 0

    # read property lists
    line = _fast_forward(f)
    cell_data_names = []
    cell_data_dtypes = []
    # read cell data
    while line[:8] == "property":
        if line[:13] == "property list":
            m = re.match("property list (.+) (.+) (.+)", line)
            cell_data_dtypes.append(tuple(m.groups()[:-1]))
        else:
            m = re.match("property (.+) (.+)", line)
            cell_data_dtypes.append(m.groups()[0])
        cell_data_names.append(m.groups()[-1])
        line = _fast_forward(f)

    if line != "end_header":
        raise ReadError()

    if is_binary:
        mesh = _read_binary(
            f,
            endianness,
            point_data_names,
            point_data_formats,
            num_verts,
            num_cells,
            cell_data_names,
            cell_data_dtypes,
        )
    else:
        mesh = _read_ascii(
            f,
            point_data_names,
            point_data_formats,
            num_verts,
            num_cells,
            cell_data_names,
            cell_data_dtypes,
        )

    return mesh


def _read_ascii(
    f,
    point_data_names,
    point_data_formats,
    num_verts,
    num_cells,
    cell_data_names,
    cell_dtypes,
):
    # assert that all formats are the same
    # Now read the data
    dtype = numpy.dtype(
        [
            (name, ply_to_numpy_dtype[fmt])
            for name, fmt in zip(point_data_names, point_data_formats)
        ]
    )
    pd = numpy.genfromtxt(f, max_rows=num_verts, dtype=dtype)

    # split off coordinate data and additional point data
    verts = []
    k = 0
    if point_data_names[0] == "x":
        verts.append(pd["x"])
        k += 1
    if point_data_names[1] == "y":
        verts.append(pd["y"])
        k += 1
    if point_data_names[2] == "z":
        verts.append(pd["z"])
        k += 1
    verts = numpy.column_stack(verts)

    point_data = {
        point_data_names[i]: pd[point_data_names[i]]
        for i in range(k, len(point_data_names))
    }

    # the faces must be read line-by-line
    triangles = []
    quads = []
    for k in range(num_cells):
        line = f.readline().decode("utf-8").strip()
        data = line.split()
        if k == 0:
            # initialize the cell data arrays
            n = []
            i = 0
            cell_data = {}
            for name, dtype in zip(cell_data_names, cell_dtypes):
                n = int(data[i])
                if name != "vertex_indices":
                    cell_data[name] = []
                i += n + 1

        i = 0
        for name, dtype in zip(cell_data_names, cell_dtypes):
            n = int(data[i])
            dtype = ply_to_numpy_dtype[dtype[1]]
            data = [dtype(data[j]) for j in range(i + 1, i + n + 1)]
            if name == "vertex_indices":
                if n == 3:
                    triangles.append(data)
                else:
                    if n != 4:
                        raise ReadError()
                    quads.append(data)
            else:
                cell_data[name].append(data)
            i += n + 1

    cells = []
    if len(triangles) > 0:
        cells.append(CellBlock("triangle", numpy.array(triangles)))
    if len(quads) > 0:
        cells.append(CellBlock("quad", numpy.array(quads)))

    return Mesh(verts, cells, point_data=point_data, cell_data=cell_data)


def _read_binary(
    f,
    endianness,
    point_data_names,
    formats,
    num_verts,
    num_cells,
    cell_data_names,
    cell_data_dtypes,
):
    ply_to_numpy_dtype_string = {
        "uchar": "i1",
        "uint": "u4",
        "uint8": "u1",
        "uint16": "u2",
        "uint32": "u4",
        "uint64": "u8",
        "int": "i4",
        "int8": "i1",
        "int32": "i4",
        "int64": "i8",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
    }

    # read point data
    dtype = [
        (name, endianness + ply_to_numpy_dtype_string[fmt])
        for name, fmt in zip(point_data_names, formats)
    ]
    point_data = numpy.fromfile(f, count=num_verts, dtype=dtype)
    verts = numpy.column_stack([point_data["x"], point_data["y"], point_data["z"]])
    point_data = {
        name: point_data[name]
        for name in point_data_names
        if name not in ["x", "y", "z"]
    }

    # Convert strings to proper numpy dtypes
    dts = [
        (
            endianness + ply_to_numpy_dtype_string[dtype[0]],
            endianness + ply_to_numpy_dtype_string[dtype[1]],
        )
        if isinstance(dtype, tuple)
        else endianness + ply_to_numpy_dtype_string[dtype]
        for dtype in cell_data_dtypes
    ]

    # read cell data -- this part is really slow
    cells = []
    last_cell_type = None
    cell_data = {name: [] for name in cell_data_names if name != "vertex_indices"}
    for _ in range(num_cells):
        for name, dt in zip(cell_data_names, dts):
            if name == "vertex_indices":
                assert isinstance(dt, tuple)
                count = numpy.fromfile(f, count=1, dtype=dt[0])[0]
                data = numpy.fromfile(f, count=count, dtype=dt[1])
                if count not in [3, 4]:
                    raise ReadError("Expected count 3 or 4, got {}.".format(count))
                cell_type = "triangle" if count == 3 else "quad"
                is_new_block = last_cell_type != cell_type
                if is_new_block:
                    cells.append(CellBlock(cell_type, []))
                    last_cell_type = cell_type
                cells[-1].data.append(data)
            else:
                if isinstance(dt, tuple):
                    count = numpy.fromfile(f, count=1, dtype=dt[0])[0]
                    data = numpy.fromfile(f, count=count, dtype=dt[1])
                else:
                    data = numpy.fromfile(f, count=1, dtype=dt)[0]
                if is_new_block:
                    cell_data[name].append([])
                cell_data[name][-1].append(data)

    # convert to numpy arrays
    cells = [CellBlock(block.type, numpy.array(block.data)) for block in cells]
    for key, values in cell_data.items():
        cell_data[key] = [numpy.array(val) for val in values]

    return Mesh(verts, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh, binary=True):  # noqa: C901
    for key in mesh.cells:
        if not any(c.type in ["triangle", "quad"] for c in mesh.cells):
            raise WriteError("Can only deal with triangular and quadrilateral faces")

    with open_file(filename, "wb") as fh:
        fh.write(b"ply\n")
        fh.write(b"comment Created by meshio\n")

        if binary:
            fh.write(
                "format binary_{}_endian 1.0\n".format(sys.byteorder).encode("utf-8")
            )
        else:
            fh.write(b"format ascii 1.0\n")

        # counts
        fh.write("element vertex {:d}\n".format(mesh.points.shape[0]).encode("utf-8"))
        #
        dim_names = ["x", "y", "z"]
        # From <https://en.wikipedia.org/wiki/PLY_(file_format)>:
        #
        # > The type can be specified with one of char uchar short ushort int uint float
        # > double, or one of int8 uint8 int16 uint16 int32 uint32 float32 float64.
        #
        # We're adding [u]int64 here.
        type_name_table = {
            numpy.dtype(numpy.int8): "int8",
            numpy.dtype(numpy.int16): "int16",
            numpy.dtype(numpy.int32): "int32",
            numpy.dtype(numpy.int64): "int64",
            numpy.dtype(numpy.uint8): "uint8",
            numpy.dtype(numpy.uint16): "uint16",
            numpy.dtype(numpy.uint32): "uint32",
            numpy.dtype(numpy.uint64): "uint64",
            numpy.dtype(numpy.float32): "float",
            numpy.dtype(numpy.float64): "double",
        }
        for k in range(mesh.points.shape[1]):
            type_name = type_name_table[mesh.points.dtype]
            fh.write("property {} {}\n".format(type_name, dim_names[k]).encode("utf-8"))
        for key, value in mesh.point_data.items():
            type_name = type_name_table[value.dtype]
            fh.write("property {} {}\n".format(type_name, key).encode("utf-8"))

        num_cells = 0
        for cell_type, c in mesh.cells:
            if cell_type in ["triangle", "quad"]:
                num_cells += c.data.shape[0]
        fh.write("element face {:d}\n".format(num_cells).encode("utf-8"))

        # possibly cast down to int32
        cells = mesh.cells
        has_cast = False
        for k, (cell_type, data) in enumerate(mesh.cells):
            if data.dtype == numpy.int64:
                has_cast = True
                mesh.cells[k] = CellBlock(cell_type, data.astype(numpy.int32))

        if has_cast:
            warnings.warn(
                "PLY doesn't support 64-bit integers. Casting down to 32-bit."
            )

        # TODO use uint8 for cell count

        # assert that all cell dtypes are equal
        cell_dtype = None
        for _, cell in cells:
            if cell_dtype is None:
                cell_dtype = cell.dtype
            if cell.dtype != cell_dtype:
                raise WriteError()

        ply_type = numpy_to_ply_dtype[cell_dtype]
        fh.write(
            "property list {} {} vertex_indices\n".format(ply_type, ply_type).encode(
                "utf-8"
            )
        )
        # TODO other cell data
        fh.write(b"end_header\n")

        if binary:
            # points and point_data
            out = numpy.rec.fromarrays(
                [coord for coord in mesh.points.T] + list(mesh.point_data.values())
            )
            out.tofile(fh)

            # cells
            for cell_type, data in cells:
                if cell_type not in ["triangle", "quad"]:
                    continue
                # prepend with count
                count = numpy.full(data.shape[0], data.shape[1], dtype=data.dtype)
                out = numpy.column_stack([count, data])
                out.tofile(fh)
        else:
            # vertices
            # numpy.savetxt(fh, mesh.points, "%r")  # slower
            # out = numpy.column_stack([mesh.points] + list(mesh.point_data.values()))
            out = numpy.rec.fromarrays(
                [coord for coord in mesh.points.T] + list(mesh.point_data.values())
            )
            fmt = " ".join(["{}"] * len(out[0]))
            out = "\n".join([fmt.format(*row) for row in out]) + "\n"
            fh.write(out.encode("utf-8"))

            # cells
            for cell_type, data in cells:
                if cell_type not in ["triangle", "quad"]:
                    continue
                out = numpy.column_stack(
                    [numpy.full(data.shape[0], data.shape[1], dtype=data.dtype), data]
                )
                # savetxt is slower
                # numpy.savetxt(fh, out, "%d  %d %d %d")
                fmt = " ".join(["{}"] * out.shape[1])
                out = "\n".join([fmt.format(*row) for row in out]) + "\n"
                fh.write(out.encode("utf-8"))


register("ply", [".ply"], read, {"ply": write})
