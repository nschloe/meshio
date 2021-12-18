"""
I/O for the OFF surface format, cf.
<https://en.wikipedia.org/wiki/OFF_(file_format)>,
<http://www.geomview.org/docs/html/OFF.html>.
"""
import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh


def read(filename):
    with open_file(filename) as f:
        points, cells = read_buffer(f)
    return Mesh(points, cells)


def read_buffer(f):
    # assert that the first line reads `OFF`
    line = f.readline()

    if isinstance(line, (bytes, bytearray)):
        raise ReadError("Expected text buffer, not bytes.")

    if line.strip() != "OFF":
        raise ReadError("Expected the first line to be `OFF`.")

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

    verts = np.fromfile(f, dtype=float, count=3 * num_verts, sep=" ").reshape(
        num_verts, 3
    )
    data = np.fromfile(f, dtype=int, count=4 * num_faces, sep=" ").reshape(num_faces, 4)
    if not np.all(data[:, 0] == 3):
        raise ReadError("Can only read triangular faces")
    cells = [CellBlock("triangle", data[:, 1:])]

    return verts, cells


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        warn(
            "OFF requires 3D points, but 2D points given. "
            "Appending 0 as third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    skip = [c for c in mesh.cells if c.type != "triangle"]
    if skip:
        string = ", ".join(item.type for item in skip)
        warn(f"OFF only supports triangle cells. Skipping {string}.")

    tri = mesh.get_cells_type("triangle")

    with open(filename, "wb") as fh:
        fh.write(b"OFF\n")
        fh.write(b"# Created by meshio\n\n")

        # counts
        c = f"{mesh.points.shape[0]} {tri.shape[0]} {0}\n\n"
        fh.write(c.encode())

        # vertices
        # np.savetxt(fh, mesh.points, "%r")  # slower
        fmt = " ".join(["{}"] * points.shape[1])
        out = "\n".join([fmt.format(*row) for row in points]) + "\n"
        fh.write(out.encode())

        # triangles
        out = np.column_stack([np.full(tri.shape[0], 3, dtype=tri.dtype), tri])
        # savetxt is slower
        # np.savetxt(fh, out, "%d  %d %d %d")
        fmt = " ".join(["{}"] * out.shape[1])
        out = "\n".join([fmt.format(*row) for row in out]) + "\n"
        fh.write(out.encode())


register_format("off", [".off"], read, {"off": write})
