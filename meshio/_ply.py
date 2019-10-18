"""
I/O for the PLY format, cf.
<https://en.wikipedia.org/wiki/PLY_(file_format)>.
<https://web.archive.org/web/20161221115231/http://www.cs.virginia.edu/~gfx/Courses/2001/Advanced.spring.01/plylib/Ply.txt>.
"""
import logging
import re

import numpy

from ._mesh import Mesh


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

    # Reference dtypes
    dtype_table = {
        "short": numpy.int16,
        "ushort": numpy.uint16,
        "int": numpy.int32,
        "uint": numpy.uint32,
        "float": numpy.float32,
        "double": numpy.float64,
    }

    # fast forward to the next significant line
    line = _fast_forward(f)
    m = re.match("property (.+) x", line)
    format_x = m.groups()[0]
    #
    line = _fast_forward(f)
    m = re.match("property (.+) y", line)
    format_y = m.groups()[0]
    #
    line = _fast_forward(f)
    m = re.match("property (.+) z", line)
    format_z = m.groups()[0]
    assert format_x == format_y == format_z
    dtype_verts = dtype_table[format_x]

    line = _fast_forward(f)
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
        dtypes.append(dtype_table[types[1]])
        line = _fast_forward(f)

    assert line == "end_header"

    # Now read the data
    verts = numpy.fromfile(f, dtype=dtype_verts, count=3 * num_verts, sep=" ").reshape(
        num_verts, 3
    )

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

    return Mesh(verts, cells, cell_data=cell_data)


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        logging.warning(
            "OFF requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    for key in mesh.cells:
        assert key in ["triangle"], "Can only deal with triangular faces"

    tri = mesh.cells["triangle"]

    with open(filename, "wb") as fh:
        fh.write(b"OFF\n")
        fh.write(b"# Created by meshio\n\n")

        # counts
        c = "{} {} {}\n\n".format(mesh.points.shape[0], tri.shape[0], 0)
        fh.write(c.encode("utf-8"))

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
