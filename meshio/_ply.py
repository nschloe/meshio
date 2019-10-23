"""
I/O for the PLY format, cf.
<https://en.wikipedia.org/wiki/PLY_(file_format)>.
<https://web.archive.org/web/20161221115231/http://www.cs.virginia.edu/~gfx/Courses/2001/Advanced.spring.01/plylib/Ply.txt>.
"""
import re
import sys
import warnings

import numpy

from ._mesh import Mesh

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
    "float": numpy.float32,
    "float32": numpy.float32,
    "double": numpy.float64,
}
numpy_to_ply_dtype = {numpy.dtype(v): k for k, v in ply_to_numpy_dtype.items()}


def read(filename):
    with open(filename, "rb") as f:
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
    assert line == "ply"

    line = _fast_forward(f)
    if line == "format ascii 1.0":
        is_binary = False
    elif line == "format binary_big_endian 1.0":
        is_binary = True
        endianness = ">"
    else:
        assert line == "format binary_little_endian 1.0"
        is_binary = True
        endianness = "<"

    line = _fast_forward(f)
    m = re.match("element vertex (\\d+)", line)
    num_verts = int(m.groups()[0])

    # fast forward to the next significant line
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

    # read property lists
    line = _fast_forward(f)
    cell_data_names = []
    cell_dtypes = []
    while line != "end_header":
        m = re.match("property list (.+) (.+) (.+)", line)
        types = m.groups()[:2]
        name = m.groups()[2]
        cell_data_names.append(name)
        cell_dtypes.append(tuple(types))
        line = _fast_forward(f)

    assert line == "end_header"

    if is_binary:
        mesh = _read_binary(
            f,
            endianness,
            point_data_names,
            point_data_formats,
            num_verts,
            num_cells,
            cell_data_names,
            cell_dtypes,
        )
    else:
        mesh = _read_ascii(
            f,
            point_data_names,
            point_data_formats,
            num_verts,
            num_cells,
            cell_data_names,
            cell_dtypes,
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
                    assert n == 4
                    quads.append(data)
            else:
                cell_data[name].append(data)
            i += n + 1

    cells = {}
    if len(triangles) > 0:
        cells["triangle"] = numpy.array(triangles)
        # cell_data = {"triangle": cell_data}

    if len(quads) > 0:
        cells["quad"] = numpy.array(quads)
        # cell_data = {"quad": cell_data}

    return Mesh(verts, cells, point_data=point_data, cell_data=cell_data)


def _read_binary(
    f,
    endianness,
    point_data_names,
    formats,
    num_verts,
    num_cells,
    cell_data_names,
    cell_dtypes,
):
    ply_to_numpy_dtype_string = {
        "uchar": "i1",
        "uint8": "u1",
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
    dt = [
        (
            endianness + ply_to_numpy_dtype_string[dtypes[0]],
            endianness + ply_to_numpy_dtype_string[dtypes[1]],
        )
        for dtypes in cell_dtypes
    ]

    # read cell data -- this part is really slow
    triangles = []
    quads = []
    for _ in range(num_cells):
        count = numpy.fromfile(f, count=1, dtype=dt[0][0])[0]
        data = numpy.fromfile(f, count=count, dtype=dt[0][1])
        if count == 3:
            triangles.append(data)
        else:
            assert count == 4
            quads.append(data)
        # cell data
        for dtypes in dt[1:]:
            pass

    cells = {}
    if len(triangles) > 0:
        cells["triangle"] = numpy.array(triangles)
    if len(quads) > 0:
        cells["quad"] = numpy.array(quads)

    return Mesh(verts, cells, point_data=point_data, cell_data={})


def write(filename, mesh, binary=True):
    for key in mesh.cells:
        assert key in [
            "triangle",
            "quad",
        ], "Can only deal with triangular and quadrilateral faces"

    with open(filename, "wb") as fh:
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
        type_name_table = {
            numpy.dtype(numpy.int8): "int8",
            numpy.dtype(numpy.int64): "int64",
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
        if "triangle" in mesh.cells:
            num_cells += mesh.cells["triangle"].shape[0]
        if "quad" in mesh.cells:
            num_cells += mesh.cells["quad"].shape[0]
        fh.write("element face {:d}\n".format(num_cells).encode("utf-8"))

        # possibly cast down to int32
        cells = mesh.cells
        has_cast = False
        for key in mesh.cells:
            if mesh.cells[key].dtype == numpy.int64:
                has_cast = True
                cells[key] = mesh.cells[key].astype(numpy.int32)

        if has_cast:
            warnings.warn(
                "PLY doesn't support 64-bit integers. Casting down to 32-bit."
            )

        # TODO use uint8 for cell count

        # assert that all cell dtypes are equal
        cell_dtype = None
        for cell in cells.values():
            if cell_dtype is None:
                cell_dtype = cell.dtype
            assert cell.dtype == cell_dtype

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
            fh.write(out.tostring())

            # cells
            for key in ["triangle", "quad"]:
                if key not in cells:
                    continue
                dat = cells[key]
                # prepend with count
                count = numpy.full(dat.shape[0], dat.shape[1], dtype=dat.dtype)
                out = numpy.column_stack([count, dat])
                fh.write(out.tostring())
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
            for key in ["triangle", "quad"]:
                if key not in cells:
                    continue
                dat = cells[key]
                out = numpy.column_stack([numpy.full(dat.shape[0], dat.shape[1]), dat])
                # savetxt is slower
                # numpy.savetxt(fh, out, "%d  %d %d %d")
                fmt = " ".join(["{}"] * out.shape[1])
                out = "\n".join([fmt.format(*row) for row in out]) + "\n"
                fh.write(out.encode("utf-8"))

    return
