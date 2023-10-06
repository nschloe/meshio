"""
Neuroglancer format, used in large-scale neuropil segmentation data.

Adapted from https://github.com/HumanBrainProject/neuroglancer-scripts/blob/1fcabb613a715ba17c65d52596dec3d687ca3318/src/neuroglancer_scripts/mesh.py (MIT license)
"""
import struct

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh


def write(filename, mesh):
    with open_file(filename, "wb") as f:
        write_buffer(f, mesh)


def write_buffer(f, mesh):
    """Store a mesh in Neuroglancer pre-computed format.
    :param file: a file-like object opened in binary mode (its ``write`` method
        will be called with :class:`bytes` objects).
    :param meshio.Mesh mesh: Mesh object to write
    """
    vertices = np.asarray(mesh.points, "<f")
    skip = [c for c in mesh.cells if c.type != "triangle"]
    if skip:
        warn('Neuroglancer only supports triangle cells. Skipping {", ".join(skip)}.')

    f.write(struct.pack("<I", vertices.shape[0]))
    f.write(vertices.tobytes(order="C"))
    for c in filter(lambda c: c.type == "triangle", mesh.cells):
        f.write(np.asarray(c.data, "<I").tobytes(order="C"))


def read(filename):
    with open_file(filename, "rb") as f:
        return read_buffer(f)


def read_buffer(file):
    """Load a mesh in Neuroglancer pre-computed format.
    :param file: a file-like object opened in binary mode (its ``read`` method
        is expected to return :class:`bytes` objects).
    :returns meshio.Mesh:
    """
    num_vertices = struct.unpack("<I", file.read(4))[0]
    # TODO handle format errors
    #
    # Use frombuffer instead of np.fromfile, because the latter expects a
    # real file and performs direct I/O on file.fileno(), which can fail or
    # read garbage e.g. if the file is an instance of gzip.GzipFile.
    buf = file.read(4 * 3 * num_vertices)
    if len(buf) != 4 * 3 * num_vertices:
        raise ReadError("The precomputed mesh data is too short")
    flat_vertices = np.frombuffer(buf, "<f")
    vertices = np.reshape(flat_vertices, (num_vertices, 3),).copy(
        order="C"
    )  # TODO remove copy
    # BUG: this could easily exhaust memory if reading a large file that is not
    # in precomputed format.
    buf = file.read()
    if len(buf) % (3 * 4) != 0:
        raise ReadError("The size of the precomputed mesh data is not adequate")
    flat_triangles = np.frombuffer(buf, "<I")
    triangles = np.reshape(flat_triangles, (-1, 3)).copy(order="C")  # TODO remove copy
    if np.any(triangles > num_vertices):
        raise ReadError("The mesh references nonexistent vertices")

    return Mesh(vertices, [CellBlock("triangle", triangles)])


register_format("neuroglancer", [], read, {"neuroglancer": write})
