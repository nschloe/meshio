"""
I/O for the TetGen file format, c.f.
<https://wias-berlin.de/software/tetgen/fformats.node.html>
"""
import os
from itertools import islice

import numpy

from ._mesh import Mesh


def read(filename):
    base, ext = os.path.splitext(filename)
    if ext == ".node":
        node_filename = filename
        ele_filename = base + ".ele"
    else:
        assert ext == ".ele"
        node_filename = base + ".node"
        ele_filename = filename

    # read nodes
    with open(node_filename) as f:
        line = next(islice(f, 1))
        num_points, dim, num_attrs, num_bmarkers = [
            int(item) for item in line.strip().split(" ") if item != ""
        ]
        assert dim == 3
        flt = numpy.vectorize(float)
        points = numpy.empty((num_points, 3), dtype=float)
        for k in range(num_points):
            line = next(islice(f, 1)).strip()
            out = [item for item in line.split(" ") if item != ""]
            assert len(out) == 4 + num_attrs + num_bmarkers
            points[k] = flt(out[1:4])
            if k == 0:
                node_index_base = int(out[0])
            # make sure the nodes a numbered consecutively
            assert int(out[0]) == node_index_base + k
            # discard the attributes and boundary markers

    # read elements
    with open(ele_filename) as f:
        line = next(islice(f, 1))
        num_tets, num_points_per_tet, num_attrs = [
            int(item) for item in line.strip().split(" ") if item != ""
        ]
        assert num_points_per_tet == 4
        cells = numpy.empty((num_tets, 4), dtype=int)
        for k in range(num_tets):
            line = next(islice(f, 1)).strip()
            out = [item for item in line.split(" ") if item != ""]
            assert len(out) == 5 + num_attrs
            cells[k] = [int(item) for item in out[1:5]]
            # discard the attributes

        cells -= node_index_base

    return Mesh(points, {"tetra": cells})
