"""
I/O for the OFF surface format, cf.
<https://en.wikipedia.org/wiki/OFF_(file_format)>,
<http://www.geomview.org/docs/html/OFF.html>.
"""
import logging

import numpy

from ._mesh import Mesh


def read(filename):
    with open(filename) as f:
        points, cells = read_buffer(f)
    return Mesh(points, cells)


def read_buffer(f):
    # assert that the first line reads `OFF`
    line = f.readline()
    assert line.strip() == "OFF"

    # fast forward to the next significant line
    while True:
        line = f.readline().strip()
        if line and line[0] != "#":
            break

    # This next line contains:
    # <number of vertices> <number of faces> <number of edges>
    num_verts, num_faces, _ = line.split(" ")
    num_verts = int(num_verts)
    num_faces = int(num_faces)

    verts = numpy.fromfile(f, dtype=float, count=3 * num_verts, sep=" ").reshape(
        num_verts, 3
    )

    data = numpy.fromfile(f, dtype=int, count=4 * num_faces, sep=" ").reshape(
        num_faces, 4
    )
    assert numpy.all(data[:, 0] == 3), "Can only read triangular faces"
    cells = {"triangle": data[:, 1:]}

    return verts, cells


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
