"""
I/O for the STL format, cf.
<https://en.wikipedia.org/wiki/STL_(file_format)>.
"""
from __future__ import annotations

import os

import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh


def read(filename):
    with open_file(filename, "rb") as f:
        # Checking if the file is ASCII format is normally done by checking if the
        # first 5 characters of the header is "solid".
        # ```
        # header = f.read(80).decode()
        # ```
        # Unfortunately, there are mesh files out there which are binary and still put
        # "solid" there.
        # A suggested alternative is to pretend the file is binary, read the
        # num_triangles and see if it matches the file size
        # (https://stackoverflow.com/a/7394842/353337).
        filesize_bytes = os.path.getsize(filename)
        if filesize_bytes < 80:
            return _read_ascii(f)

        f.read(80)
        num_triangles = np.fromfile(f, count=1, dtype="<u4")[0]
        # for each triangle, one has 3 float32 (facet normal), 9 float32 (facet),
        # and 1 int16 (attribute count), 50 bytes in total
        if 84 + num_triangles * 50 == filesize_bytes:
            return _read_binary(f, num_triangles)

        # rewind and skip header
        f.seek(0)
        f.readline()
        return _read_ascii(f)


# np.loadtxt is super slow
# Code adapted from <https://stackoverflow.com/a/8964779/353337>.
def iter_loadtxt(
    infile,
    skiprows: int = 0,
    comments: str | tuple[str, ...] = "#",
    dtype=float,
    usecols: tuple[int] | None = None,
):
    def iter_func():
        items = None
        for _ in range(skiprows):
            try:
                next(infile)
            except StopIteration:
                raise ReadError("EOF Skipped too many rows")

        for line in infile:
            line = line.decode().strip()
            if line.startswith(comments):
                continue
            # remove all text
            items = line.split()[-3:]
            usecols_ = range(len(items)) if usecols is None else usecols
            for idx in usecols_:
                yield dtype(items[idx])

        if items is None:
            iter_loadtxt.rowlength = 3
            return

        iter_loadtxt.rowlength = len(items) if usecols is None else len(usecols)

    data = np.fromiter(iter_func(), dtype=dtype)
    return data.reshape((-1, iter_loadtxt.rowlength))


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
    # In the interest of speed, don't verify the format and instead just skip the text.

    # TODO Pandas is MUCH faster than numpy for i/o, see
    # <https://stackoverflow.com/a/18260092/353337>.
    # import pandas
    # data = pandas.read_csv(
    #     f,
    #     skiprows=lambda row: row == 0 or (row - 1) % 7 in [0, 1, 5, 6],
    #     skipfooter=1,
    #     usecols=(1, 2, 3),
    # )

    # np.loadtxt is super slow
    # data = np.loadtxt(
    #     f,
    #     comments=["solid", "facet", "outer loop", "endloop", "endfacet", "endsolid"],
    #     usecols=(1, 2, 3),
    # )
    data = iter_loadtxt(
        f,
        comments=("solid", "outer loop", "endloop", "endfacet", "endsolid"),
        # usecols=(1, 2, 3),
    )

    if data.shape[0] % 4 != 0:
        raise ReadError()

    # split off the facet normals
    facet_rows = np.zeros(len(data), dtype=bool)
    facet_rows[0::4] = True
    facet_normals = data[facet_rows]
    data = data[~facet_rows]

    if data.shape[0] == 0:
        points = []
        cells = {}
        cell_data = {}
    else:
        facets = np.split(data, data.shape[0] // 3)
        points, cells = data_from_facets(facets)
        cell_data = {"facet_normals": [facet_normals]}

    return Mesh(points, cells, cell_data=cell_data)


def data_from_facets(facets):
    # Now, all facets contain the point coordinate. Try to identify individual points
    # and build the data arrays.
    if len(facets) == 0:
        points = np.empty((0, 3), dtype=float)
        cells = []
    else:
        pts = np.concatenate(facets)
        # TODO equip `unique()` with a tolerance
        # Use return_index so we can use sort on `idx` such that the order is
        # preserved; see <https://stackoverflow.com/a/15637512/353337>.
        _, idx, inv = np.unique(pts, axis=0, return_index=True, return_inverse=True)
        k = np.argsort(idx)
        points = pts[idx[k]]
        inv_k = np.argsort(k)
        cells = [CellBlock("triangle", inv_k[inv].reshape(-1, 3))]
    return points, cells


def _read_binary(f, num_triangles: int):
    # for each triangle, one has 3 float32 (facet normal), 9 float32 (facet), and 1
    # int16 (attribute count)
    out = np.fromfile(
        f,
        count=num_triangles,
        dtype=np.dtype(
            [("normal", "<f4", (3,)), ("facet", "<f4", (3, 3)), ("attr count", "<i2")]
        ),
    )
    # discard normals, attribute count
    facets = out["facet"]
    # if not np.all(out["attr count"] == 0):
    #     print(out["attr count"])
    #     raise ReadError("Nonzero attr count")

    points, cells = data_from_facets(facets)
    return Mesh(points, cells)


def write(filename, mesh, binary=False):
    if "triangle" not in {block.type for block in mesh.cells}:
        warn("STL can only write triangle cells. No triangle cells found.")
    if len(mesh.cells) > 1:
        invalid = {block.type for block in mesh.cells if block.type != "triangle"}
        invalid = ", ".join(invalid)
        warn(f"STL can only write triangle cells. Discarding {invalid}.")

    if mesh.points.shape[1] == 2:
        warn(
            "STL requires 3D points, but 2D points given. Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    pts = points[mesh.get_cells_type("triangle")]
    if "facet_normals" in mesh.cell_data:
        normals = mesh.get_cell_data("facet_normals", "triangle")
    else:
        normals = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        nrm = np.sqrt(np.einsum("ij,ij->i", normals, normals))
        normals = (normals.T / nrm).T

    fun = _write_binary if binary else _write_ascii
    fun(filename, pts, normals)


def _write_ascii(filename, pts, normals):
    with open_file(filename, "w") as fh:
        fh.write("solid\n")
        for local_pts, normal in zip(pts, normals):
            out = (
                "\n".join(
                    [
                        "facet normal {} {} {}".format(*normal),
                        " outer loop",
                        "  vertex {} {} {}".format(*local_pts[0]),
                        "  vertex {} {} {}".format(*local_pts[1]),
                        "  vertex {} {} {}".format(*local_pts[2]),
                        " endloop",
                        "endfacet",
                    ]
                )
                + "\n"
            )
            fh.write(out)
        fh.write("endsolid\n")


def _write_binary(filename, pts, normals):
    with open_file(filename, "wb") as fh:
        # 80 character header data
        msg = f"This file was generated by meshio v{__version__}."
        msg += (79 - len(msg)) * "X"
        msg += "\n"
        fh.write(msg.encode())

        fh.write(np.array(len(pts)).astype("<u4"))

        dtype = np.dtype(
            [
                ("normal", ("<f4", 3)),
                ("points", ("<f4", (3, 3))),
                ("attr", "<u2"),
            ]
        )
        a = np.empty(len(pts), dtype=dtype)
        a["normal"] = normals
        a["points"] = pts
        a["attr"] = 0
        a.tofile(fh)


register_format("stl", [".stl"], read, {"stl": write})
