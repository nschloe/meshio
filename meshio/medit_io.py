# -*- coding: utf-8 -*-
#
"""
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
for something like a specification.
"""
import re
import logging
import numpy

from .mesh import Mesh


def read(filename):
    with open(filename) as f:
        points, cells = read_buffer(f)

    return Mesh(points, cells)


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

    reader = _ItemReader(file)

    while True:
        try:
            keyword = reader.next_item()
        except StopIteration:
            break

        assert keyword.isalpha()

        meshio_from_medit = {
            "Edges": ("line", 2),
            "Triangles": ("triangle", 3),
            "Quadrilaterals": ("quad", 4),
            "Tetrahedra": ("tetra", 4),
            "Hexahedra": ("hexahedra", 8),
        }

        if keyword == "MeshVersionFormatted":
            assert reader.next_item() == "1"
        elif keyword == "Dimension":
            dim = int(reader.next_item())
        elif keyword == "Vertices":
            assert dim > 0
            # The first value is the number of nodes
            num_verts = int(reader.next_item())
            points = numpy.empty((num_verts, dim), dtype=float)
            for k in range(num_verts):
                # Throw away the label immediately
                points[k] = numpy.array(reader.next_items(dim + 1), dtype=float)[:-1]
        elif keyword in meshio_from_medit:
            meshio_name, num = meshio_from_medit[keyword]
            # The first value is the number of elements
            num_cells = int(reader.next_item())
            cell_data = numpy.empty((num_cells, num), dtype=int)
            for k in range(num_cells):
                data = numpy.array(reader.next_items(num + 1), dtype=int)
                # Throw away the label
                cell_data[k] = data[:-1]

            # adapt 0-base
            cells[meshio_name] = cell_data - 1
        else:
            assert keyword == "End", "Unknown keyword '{}'.".format(keyword)

    return points, cells


def write(filename, mesh):
    with open(filename, "wb") as fh:
        fh.write(b"MeshVersionFormatted 1\n")
        fh.write(b"# Created by meshio\n")

        n, d = mesh.points.shape

        # Dimension info
        dim = "\nDimension {}\n".format(d)
        fh.write(dim.encode("utf-8"))

        # vertices
        fh.write(b"\nVertices\n")
        fh.write("{}\n".format(n).encode("utf-8"))
        labels = numpy.ones(n, dtype=int)
        data = numpy.c_[mesh.points, labels]
        fmt = " ".join(["%r"] * d) + " %d"
        numpy.savetxt(fh, data, fmt)

        medit_from_meshio = {
            "line": ("Edges", 2),
            "triangle": ("Triangles", 3),
            "quad": ("Quadrilaterals", 4),
            "tetra": ("Tetrahedra", 4),
            "hexahedra": ("Hexahedra", 8),
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
            labels = numpy.ones(len(data), dtype=int)
            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = " ".join(["%d"] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b"\nEnd\n")

    return
