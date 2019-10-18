"""
I/O for the PLY format, cf.
<https://en.wikipedia.org/wiki/PLY_(file_format)>.
<https://web.archive.org/web/20161221115231/http://www.cs.virginia.edu/~gfx/Courses/2001/Advanced.spring.01/plylib/Ply.txt>.
"""
import re
import warnings

import numpy

from ._mesh import Mesh

# Reference dtypes
ply_to_numpy_dtype = {
    "short": numpy.int16,
    "ushort": numpy.uint16,
    "int": numpy.int32,
    "int32": numpy.int32,
    "uint": numpy.uint32,
    "uint8": numpy.uint8,
    "float": numpy.float32,
    "float32": numpy.float32,
    "double": numpy.float64,
}
numpy_to_ply_dtype = {numpy.dtype(v): k for k, v in ply_to_numpy_dtype.items()}


def read(filename):
    with open(filename) as f:
        mesh = read_buffer(f)
    return mesh


def _fast_forward(f):
    # fast forward to the next significant line
    while True:
        line = f.readline().strip()
        if line and line[:7] != "comment":
            break
    return line


def read_buffer(f):
    # assert that the first line reads `ply`
    line = f.readline()
    assert line.strip() == "ply"

    line = _fast_forward(f)
    assert line == "format ascii 1.0"

    line = _fast_forward(f)
    m = re.match("element vertex (\\d+)", line)
    num_verts = int(m.groups()[0])

    # fast forward to the next significant line
    formats = []
    point_data_names = []
    line = _fast_forward(f)
    while line[:8] == "property":
        m = re.match("property (.+) (.+)", line)
        formats.append(m.groups()[0])
        point_data_names.append(m.groups()[1])
        line = _fast_forward(f)
    # assert that all formats are the same
    assert all(formats[0] == fmt for fmt in formats)
    dtype_verts = ply_to_numpy_dtype[formats[0]]

    m = re.match("element face (\\d+)", line)
    num_faces = int(m.groups()[0])

    # read property lists
    line = _fast_forward(f)
    cell_data_names = []
    dtypes = []
    while line != "end_header":
        m = re.match("property list (.+) (.+) (.+)", line)
        types = m.groups()[:2]
        name = m.groups()[2]
        cell_data_names.append(name)
        dtypes.append(ply_to_numpy_dtype[types[1]])
        line = _fast_forward(f)

    assert line == "end_header"

    # Now read the data
    point_data = numpy.fromfile(f, dtype=dtype_verts, count=len(point_data_names) * num_verts, sep=" ").reshape(
        num_verts, len(point_data_names)
    )
    assert point_data_names[0] == "x"
    assert point_data_names[1] == "y"
    k = 3 if point_data_names[2] == "z" else 2
    verts = point_data[:, :k]
    point_data = {
        name: data
        for name, data in zip(point_data_names[k:], point_data.T[k:])
    }

    # the faces must be read line-by-line
    for k in range(num_faces):
        line = f.readline().strip()
        data = line.split()
        if k == 0:
            # initialize the cell data arrays
            n = []
            i = 0
            cell_data = {}
            for name, dtype in zip(cell_data_names, dtypes):
                n = int(data[i])
                cell_data[name] = numpy.empty((num_faces, n), dtype=dtype)
                i += n + 1

        i = 0
        for name, dtype in zip(cell_data_names, dtypes):
            n = int(data[i])
            cell_data[name][k] = [dtype(data[j]) for j in range(i + 1, i + n + 1)]
            i += n + 1

    assert cell_data["vertex_indices"].shape[1] == 3
    cells = {"triangle": cell_data["vertex_indices"]}
    cell_data.pop("vertex_indices", None)

    cell_data = {"triangle": cell_data}

    return Mesh(verts, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh):
    for key in mesh.cells:
        assert key in ["triangle"], "Can only deal with triangular faces"

    tri = mesh.cells["triangle"]

    with open(filename, "wb") as fh:
        fh.write(b"ply\n")
        fh.write(b"comment Created by meshio\n")
        fh.write(b"format ascii 1.0\n")

        # counts
        fh.write("element vertex {:d}\n".format(mesh.points.shape[0]).encode("utf-8"))
        #
        dim_names = ["x", "y", "z"]
        type_name = {numpy.dtype(numpy.float64): "double"}[mesh.points.dtype]
        for k in range(mesh.points.shape[1]):
            fh.write("property {} {}\n".format(type_name, dim_names[k]).encode("utf-8"))

        fh.write(
            "element face {:d}\n".format(mesh.cells["triangle"].shape[0]).encode(
                "utf-8"
            )
        )

        if mesh.cells["triangle"].dtype == numpy.int64:
            warnings.warn(
                "PLY doesn't support 64-bit integers. Casting down to 32-bit."
            )
            mesh.cells["triangle"] = mesh.cells["triangle"].astype(numpy.int32)

        ply_type = numpy_to_ply_dtype[mesh.cells["triangle"].dtype]
        fh.write(
            "property list uint8 {} vertex_indices\n".format(ply_type).encode("utf-8")
        )
        # TODO other cell data
        fh.write(b"end_header\n")

        # vertices
        # numpy.savetxt(fh, mesh.points, "%r")  # slower
        out = mesh.points
        fmt = " ".join(["{}"] * out.shape[1])
        out = "\n".join([fmt.format(*row) for row in out]) + "\n"
        fh.write(out.encode("utf-8"))

        # triangles
        out = numpy.column_stack([numpy.full(tri.shape[0], tri.shape[1]), tri])
        # savetxt is slower
        # numpy.savetxt(fh, out, "%d  %d %d %d")
        fmt = " ".join(["{}"] * out.shape[1])
        out = "\n".join([fmt.format(*row) for row in out]) + "\n"
        fh.write(out.encode("utf-8"))

    return
