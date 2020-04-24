"""
I/O SU2 mesh format
<https://su2code.github.io/docs_v7/Mesh-File>
"""
import logging
from itertools import chain, islice

import numpy

from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import CellBlock, Mesh

# follows VTK conventions
su2_type_to_numnodes = {
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quad
    10: 4,  # tetra
    12: 8,  # hexahedron
    13: 6,  # wedge
    14: 5,  # pyramid
}
su2_to_meshio_type = {
    3: "line",
    5: "triangle",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}


def read(filename):

    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = []
    cell_data = {"su2:tag": numpy.empty(0)}

    itype = "i8"
    ftype = "f8"
    dim = 0

    next_tag_id = 0
    expected_nmarkers = 0
    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == "%":
            continue

        name, nitems = line.split("=")

        if name == "NDIME":
            dim = int(nitems)
            if dim != 2 and dim != 3:
                raise ReadError("Invalid dimension value {}".format(line))

        elif name == "NPOIN":
            num_verts = int(nitems)
            points = numpy.fromfile(
                f, count=num_verts * (dim + 1), dtype=ftype, sep=" "
            ).reshape(num_verts, dim + 1)[:, :-1]

        elif name == "NELEM" or name == "MARKER_ELEMS":
            # we cannot? read at onece using numpy becasue we do not know the
            # total size. Read, instead next num_elems as is and re-use the
            # translate_cells function from vtk reader

            num_elems = int(nitems)
            gen = islice(f, num_elems)

            # some files has an extra int column while other not
            # We do not need it so make sure we will skip it
            first_line_str = next(gen)
            first_line = first_line_str.split()
            nnodes = su2_type_to_numnodes[int(first_line[0])]
            has_extra_column = False
            if nnodes + 1 == len(first_line):
                has_extra_column = False
            elif nnodes + 2 == len(first_line):
                has_extra_column = True
            else:
                raise ReadError("Invalid number of columns for {} field".format(name))

            # reset generator
            gen = chain([first_line_str], gen)

            cell_array = " ".join([line.rstrip("\n") for line in gen])
            cell_array = numpy.fromiter(cell_array.split(), dtype=itype)

            cells_, cell_data_ = _translate_cells(cell_array, has_extra_column)

            for eltype, data in cells_.items():
                cells.append(CellBlock(eltype, data))
            if name == "NELEM=":
                cell_data["su2:tag"] = numpy.hstack(
                    (cell_data["su2:tag"], numpy.full(num_elems, 0))
                )
            else:
                tags = numpy.full(num_elems, next_tag_id)
                cell_data["su2:tag"] = numpy.hstack((cell_data["su2:tag"], tags))
        elif name == "NMARK":
            expected_nmarkers = int(nitems)
        elif name == "MARKER_TAG":
            next_tag = nitems
            try:
                next_tag_id = int(next_tag)
            except ValueError:
                next_tag_id += 1
                logging.warning(
                    "meshio does not support tags of string type.\n"
                    "    Surface tag {} will be replaced by {}".format(
                        name, next_tag_id
                    )
                )
            expected_nmarkers -= 1

    if expected_nmarkers != 0:
        raise ReadError("NMARK does not match the number of markers found")

    return Mesh(points, cells, cell_data=cell_data)


def _translate_cells(data, has_extra_column=False):
    # adapted from _vtk.py
    # Translate input array  into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (vtk cell type, p0, p1, ... ,pk, vtk cell type, p10, p11, ..., p1k, ...

    entry_offset = 1
    if has_extra_column:
        entry_offset += 1

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    types = []
    i = 0
    while i < len(data):
        types.append(data[i])
        i += su2_type_to_numnodes[data[i]] + entry_offset

    types = numpy.array(types)
    bins = {u: numpy.where(types == u)[0] for u in numpy.unique(types)}

    # Deduct offsets from the cell types. This is much faster than manually
    # going through the data array. Slight disadvantage: This doesn't work for
    # cells with a custom number of points.
    numnodes = numpy.empty(len(types), dtype=int)
    for tpe, idx in bins.items():
        numnodes[idx] = su2_type_to_numnodes[tpe]
    offsets = numpy.cumsum(numnodes + entry_offset) - (numnodes + entry_offset)

    cells = {}
    cell_data = {}
    for tpe, b in bins.items():
        meshio_type = su2_to_meshio_type[tpe]
        nnodes = su2_type_to_numnodes[tpe]
        indices = numpy.add.outer(offsets[b], numpy.arange(1, nnodes + 1))
        cells[meshio_type] = data[indices]

    return cells, cell_data


def write(filename, mesh):
    return


register("su2", [".su2"], read, {"su2": write})
