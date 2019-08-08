"""
I/O for the STL format, cf.
<https://en.wikipedia.org/wiki/STL_(file_format)>.
"""
import logging

import numpy

from ._mesh import Mesh


def read(filename):
    """Reads a Gmsh msh file.
    """
    with open(filename, "rb") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    data = numpy.frombuffer(f.read(5), dtype=numpy.uint8)
    if "".join([chr(item) for item in data]) == "solid":
        # read until the end of the line
        f.readline()
        return _read_ascii(f)

    # binary: read and discard 75 more bytes
    f.read(75)
    return _read_binary(f)


# numpy.loadtxt is super slow
# Code adapted from <https://stackoverflow.com/a/8964779/353337>.
def iter_loadtxt(
    infile, delimiter=" ", skiprows=0, comments=["#"], dtype=float, usecols=None
):
    def iter_func():
        for _ in range(skiprows):
            next(infile)
        for line in infile:
            line = line.decode("utf-8").strip()
            if line.startswith(comments):
                continue
            items = line.split(delimiter)
            usecols_ = range(len(items)) if usecols is None else usecols
            for idx in usecols_:
                yield dtype(items[idx])
        iter_loadtxt.rowlength = len(line) if usecols is None else len(usecols)

    data = numpy.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def _read_ascii(f):
    # The file has the form
    # ```
    # solid foo
    #   facet normal 0.455194 -0.187301 -0.870469
    #    outer loop
    #     vertex 266.36 234.594 14.6145
    #     vertex 268.582 234.968 15.6956
    #     vertex 267.689 232.646 15.7283
    #    endloop
    #   endfacet
    #   # [...] more facets [...]
    # endsolid
    # ```
    # In the interest of speed, don't verify the format and instead just skip
    # the text.

    # TODO Pandas is MUCH faster than numpy for i/o, see
    # <https://stackoverflow.com/a/18260092/353337>.
    # import pandas
    # data = pandas.read_csv(
    #     f,
    #     skiprows=lambda row: row == 0 or (row - 1) % 7 in [0, 1, 5, 6],
    #     skipfooter=1,
    #     usecols=(1, 2, 3),
    # )

    # numpy.loadtxt is super slow
    # data = numpy.loadtxt(
    #     f,
    #     comments=["solid", "facet", "outer loop", "endloop", "endfacet", "endsolid"],
    #     usecols=(1, 2, 3),
    # )
    data = iter_loadtxt(
        f,
        comments=("solid", "facet", "outer loop", "endloop", "endfacet", "endsolid"),
        usecols=(1, 2, 3),
    )

    assert data.shape[0] % 3 == 0

    facets = numpy.split(data, data.shape[0] // 3)
    points, cells = data_from_facets(facets)
    return Mesh(points, cells)


def data_from_facets(facets):
    # Now, all facets contain the point coordinate. Try to identify individual
    # points and build the data arrays.
    pts = numpy.concatenate(facets)

    # TODO equip `unique()` with a tolerance
    # Use return_index so we can use sort on `idx` such that the order is
    # preserved; see <https://stackoverflow.com/a/15637512/353337>.
    _, idx, inv = numpy.unique(pts, axis=0, return_index=True, return_inverse=True)
    k = numpy.argsort(idx)
    points = pts[idx[k]]
    inv_k = numpy.argsort(k)
    cells = {"triangle": inv_k[inv].reshape(-1, 3)}
    return points, cells


def _read_binary(f):
    # read the first uint32 byte to get the number of triangles
    num_triangles = numpy.fromfile(f, count=1, dtype=numpy.uint32)[0]

    # for each triangle, one has 3 float32 (facet normal), 9 float32 (facet), and 1
    # int16 (attribute count)
    out = numpy.fromfile(
        f,
        count=num_triangles,
        dtype=numpy.dtype(
            [("normal", "f4", (3,)), ("facet", "f4", (3, 3)), ("attr count", "i2")]
        ),
    )
    # discard normals, attribute count
    facets = out["facet"]
    assert numpy.all(out["attr count"] == 0)

    points, cells = data_from_facets(facets)
    return Mesh(points, cells)


def write(filename, mesh, write_binary=False):
    assert (
        len(mesh.cells.keys()) == 1 and list(mesh.cells.keys())[0] == "triangle"
    ), "STL can only write triangle cells."

    if mesh.points.shape[1] == 2:
        logging.warning(
            "STL requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    if write_binary:
        _write_binary(filename, mesh.points, mesh.cells)
    else:
        _write_ascii(filename, mesh.points, mesh.cells)

    return


def _compute_normals(pts):
    normals = numpy.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    nrm = numpy.sqrt(numpy.einsum("ij,ij->i", normals, normals))
    normals = (normals.T / nrm).T
    return normals


def _write_ascii(filename, points, cells):
    pts = points[cells["triangle"]]
    normals = _compute_normals(pts)

    with open(filename, "wb") as fh:
        fh.write("solid\n".encode("utf-8"))

        for local_pts, normal in zip(pts, normals):
            # facet normal 0.455194 -0.187301 -0.870469
            #  outer loop
            #   vertex 266.36 234.594 14.6145
            #   vertex 268.582 234.968 15.6956
            #   vertex 267.689 232.646 15.7283
            #  endloop
            # endfacet
            out = ["facet normal {} {} {}".format(*normal), " outer loop"]
            for pt in local_pts:
                out += ["  vertex {} {} {}".format(*pt)]
            out += [" endloop", "endfacet"]
            fh.write(("\n".join(out) + "\n").encode("utf-8"))

        fh.write("endsolid\n".encode("utf-8"))

    return


def _write_binary(filename, points, cells):
    pts = points[cells["triangle"]]
    normals = _compute_normals(pts)

    with open(filename, "wb") as fh:
        # 80 character header data
        msg = "This file was generated by meshio."
        msg += (79 - len(msg)) * "X"
        msg += "\n"
        fh.write(msg.encode("utf-8"))
        fh.write(numpy.uint32(len(cells["triangle"])))
        for pt, normal in zip(pts, normals):
            fh.write(normal.astype(numpy.float32))
            fh.write(pt.astype(numpy.float32))
            fh.write(numpy.uint16(0))

    return
