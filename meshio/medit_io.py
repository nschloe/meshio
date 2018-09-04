# -*- coding: utf-8 -*-
#
"""
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
<https://www.math.u-bordeaux.fr/~dobrzyns/logiciels/RT-422/node58.html>
for something like a specification.
"""
from ctypes import c_float, c_double
import re
import logging
import numpy

from .mesh import Mesh


def read(filename):
    with open(filename) as f:
        mesh = read_buffer(f)

    return mesh


class _ItemReader:
    def __init__(self, file, delimiter=r"\s+"):
        # Items can be separated by any whitespace, including new lines.
        self._re_delimiter = re.compile(delimiter, re.MULTILINE)
        self._file = file
        self._line = []
        self._line_ptr = 0

    def next_items(self, n):
        """Returns the next n items.

        Throws StopIteration when there is not enough data to return n items.
        """
        items = []
        while len(items) < n:
            if self._line_ptr >= len(self._line):
                # Load the next line.
                line = next(self._file).strip()
                # Skip all comment and empty lines.
                while not line or line[0] == "#":
                    line = next(self._file).strip()
                self._line = self._re_delimiter.split(line)
                self._line_ptr = 0
            n_read = min(n - len(items), len(self._line) - self._line_ptr)
            items.extend(self._line[self._line_ptr : self._line_ptr + n_read])
            self._line_ptr += n_read
        return items

    def next_item(self):
        return self.next_items(1)[0]


def read_buffer(file):
    dim = 0
    cells = {}
    point_data = {}
    cell_data = {}

    meshio_from_medit = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Hexahedra": ("hexahedron", 8),  # Frey
        "Hexaedra": ("hexahedron", 8),  # Dobrzynski
    }

    reader = _ItemReader(file)

    while True:
        try:
            keyword = reader.next_item()
        except StopIteration:
            break

        assert keyword.isalpha()

        if keyword == "MeshVersionFormatted":
            version = reader.next_item()
            dtype = {"1": c_float, "2": c_double}[version]
        elif keyword == "Dimension":
            dim = int(reader.next_item())
        elif keyword == "Vertices":
            assert dim > 0
            # The first value is the number of nodes
            num_verts = int(reader.next_item())
            points = numpy.empty((num_verts, dim), dtype=dtype)
            point_data["medit:ref"] = numpy.empty(num_verts, dtype=int)
            for k in range(num_verts):
                points[k] = numpy.array(reader.next_items(dim), dtype=dtype)
                point_data["medit:ref"][k] = reader.next_item()
        elif keyword in meshio_from_medit:
            meshio_name, num = meshio_from_medit[keyword]
            # The first value is the number of elements
            num_cells = int(reader.next_item())
            cell_data[meshio_name] = {"medit:ref": numpy.empty(num_cells, dtype=int)}
            cells1 = numpy.empty((num_cells, num), dtype=int)
            for k in range(num_cells):
                data = numpy.array(reader.next_items(num + 1), dtype=int)
                cells1[k] = data[:-1]
                cell_data[meshio_name]["medit:ref"][k] = data[-1]

            # adapt 0-base
            cells[meshio_name] = cells1 - 1
        else:
            assert keyword == "End", "Unknown keyword '{}'.".format(keyword)

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh):
    with open(filename, "wb") as fh:
        version = {numpy.dtype(c_float): 1, numpy.dtype(c_double): 2}[mesh.points.dtype]
        # N. B.: PEP 461 Adding % formatting to bytes and bytearray
        fh.write(b"MeshVersionFormatted %d\n" % version)

        n, d = mesh.points.shape

        fh.write(b"Dimension %d\n" % d)

        # vertices
        fh.write(b"\nVertices\n")
        fh.write("{}\n".format(n).encode("utf-8"))
        try:
            labels = mesh.point_data["medit:ref"]
        except KeyError:
            labels = numpy.ones(n, dtype=int)
        data = numpy.c_[mesh.points, labels]
        fmt = " ".join(["%r"] * d) + " %d"
        numpy.savetxt(fh, data, fmt)

        medit_from_meshio = {
            "line": ("Edges", 2),
            "triangle": ("Triangles", 3),
            "quad": ("Quadrilaterals", 4),
            "tetra": ("Tetrahedra", 4),
            "hexahedron": ("Hexahedra", 8),
        }

        for key, data in mesh.cells.items():
            try:
                medit_name, num = medit_from_meshio[key]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    key
                )
                logging.warning(msg)
                continue
            fh.write(b"\n")
            fh.write("{}\n".format(medit_name).encode("utf-8"))
            fh.write("{}\n".format(len(data)).encode("utf-8"))
            try:
                labels = mesh.cell_data[key]["medit:ref"]
            except KeyError:
                labels = numpy.ones(len(data), dtype=int)
            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = " ".join(["%d"] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b"\nEnd\n")

    return
