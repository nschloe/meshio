"""
I/O for Ansys's msh format, cf.
<http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf>.
"""
import logging
import re

import numpy

from ..__about__ import __version__
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


def _skip_to(f, char):
    c = None
    while c != char:
        c = f.read(1).decode("utf-8")
    return


def _skip_close(f, num_open_brackets):
    while num_open_brackets > 0:
        char = f.read(1).decode("utf-8")
        if char == "(":
            num_open_brackets += 1
        elif char == ")":
            num_open_brackets -= 1
    return


def _read_points(f, line, first_point_index_overall, last_point_index):
    # If the line is self-contained, it is merely a declaration
    # of the total number of points.
    if line.count("(") == line.count(")"):
        return None, None, None

    # (3010 (zone-id first-index last-index type ND)
    out = re.match("\\s*\\(\\s*(|20|30)10\\s*\\(([^\\)]*)\\).*", line)
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()

    first_point_index = a[1]
    # store the very first point index
    if first_point_index_overall is None:
        first_point_index_overall = first_point_index
    # make sure that point arrays are subsequent
    if last_point_index is not None:
        if last_point_index + 1 != first_point_index:
            raise ReadError()
    last_point_index = a[2]
    num_points = last_point_index - first_point_index + 1
    dim = a[4]

    # Skip ahead to the byte that opens the data block (might
    # be the current line already).
    last_char = line.strip()[-1]
    while last_char != "(":
        last_char = f.read(1).decode("utf-8")

    if out.group(1) == "":
        # ASCII data
        pts = numpy.empty((num_points, dim))
        for k in range(num_points):
            # skip ahead to the first line with data
            line = ""
            while line.strip() == "":
                line = f.readline().decode("utf-8")
            dat = line.split()
            if len(dat) != dim:
                raise ReadError()
            for d in range(dim):
                pts[k][d] = float(dat[d])
    else:
        # binary data
        if out.group(1) == "20":
            dtype = numpy.float32
        else:
            if out.group(1) != "30":
                ReadError("Expected keys '20' or '30', got {}.".format(out.group(1)))
            dtype = numpy.float64
        # read point data
        pts = numpy.fromfile(f, count=dim * num_points, dtype=dtype).reshape(
            (num_points, dim)
        )

    # make sure that the data set is properly closed
    _skip_close(f, 2)
    return pts, first_point_index_overall, last_point_index


def _read_cells(f, line):
    # If the line is self-contained, it is merely a declaration of the total number of
    # points.
    if line.count("(") == line.count(")"):
        return None, None

    out = re.match("\\s*\\(\\s*(|20|30)12\\s*\\(([^\\)]+)\\).*", line)
    a = [int(num, 16) for num in out.group(2).split()]
    if len(a) <= 4:
        raise ReadError()
    first_index = a[1]
    last_index = a[2]
    num_cells = last_index - first_index + 1
    zone_type = a[3]
    element_type = a[4]

    if zone_type == 0:
        # dead zone
        return None, None

    key, num_nodes_per_cell = {
        0: ("mixed", None),
        1: ("triangle", 3),
        2: ("tetra", 4),
        3: ("quad", 4),
        4: ("hexahedron", 8),
        5: ("pyra", 5),
        6: ("wedge", 6),
    }[element_type]

    # Skip to the opening `(` and make sure that there's no non-whitespace character
    # between the last closing bracket and the `(`.
    if line.strip()[-1] != "(":
        c = None
        while True:
            c = f.read(1).decode("utf-8")
            if c == "(":
                break
            if not re.match("\\s", c):
                # Found a non-whitespace character before `(`.
                # Assume this is just a declaration line then and
                # skip to the closing bracket.
                _skip_to(f, ")")
                return None, None

    if key == "mixed":
        # From <http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf>:
        #
        # > If a zone is of mixed type (element-type=0), it will have a body that
        # > lists the element type of each cell.
        #
        # No idea where the information other than the element types is stored
        # though. Skip for now.
        data = None
    else:
        # read cell data
        if out.group(1) == "":
            # ASCII cells
            data = numpy.empty((num_cells, num_nodes_per_cell), dtype=int)
            for k in range(num_cells):
                line = f.readline().decode("utf-8")
                dat = line.split()
                if len(dat) != num_nodes_per_cell:
                    raise ReadError()
                data[k] = [int(d, 16) for d in dat]
        else:
            if key == "mixed":
                raise ReadError("Cannot read mixed cells in binary mode yet")
            # binary cells
            if out.group(1) == "20":
                dtype = numpy.int32
            else:
                if out.group(1) != "30":
                    ReadError(
                        "Expected keys '20' or '30', got {}.".format(out.group(1))
                    )
                dtype = numpy.int64
            shape = (num_cells, num_nodes_per_cell)
            count = shape[0] * shape[1]
            data = numpy.fromfile(f, count=count, dtype=dtype).reshape(shape)

    # make sure that the data set is properly closed
    _skip_close(f, 2)
    return key, data


def _read_faces(f, line):
    # faces
    # (13 (zone-id first-index last-index type element-type))

    # If the line is self-contained, it is merely a declaration of
    # the total number of points.
    if line.count("(") == line.count(")"):
        return {}

    out = re.match("\\s*\\(\\s*(|20|30)13\\s*\\(([^\\)]+)\\).*", line)
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()
    first_index = a[1]
    last_index = a[2]
    num_cells = last_index - first_index + 1
    element_type = a[4]

    element_type_to_key_num_nodes = {
        0: ("mixed", None),
        2: ("line", 2),
        3: ("triangle", 3),
        4: ("quad", 4),
    }

    key, num_nodes_per_cell = element_type_to_key_num_nodes[element_type]

    # Skip ahead to the line that opens the data block (might be
    # the current line already).
    if line.strip()[-1] != "(":
        _skip_to(f, "(")

    data = {}
    if out.group(1) == "":
        # ASCII
        if key == "mixed":
            # From <http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf>:
            #
            # > If the face zone is of mixed type (element-type = > 0), the body of the
            # > section will include the face type and will appear as follows
            # >
            # > type v0 v1 v2 c0 c1
            # >
            for k in range(num_cells):
                line = ""
                while line.strip() == "":
                    line = f.readline().decode("utf-8")
                dat = line.split()
                type_index = int(dat[0], 16)
                if type_index == 0:
                    raise ReadError()
                type_string, num_nodes_per_cell = element_type_to_key_num_nodes[
                    type_index
                ]
                if len(dat) != num_nodes_per_cell + 3:
                    raise ReadError()

                if type_string not in data:
                    data[type_string] = []

                data[type_string].append(
                    [int(d, 16) for d in dat[1 : num_nodes_per_cell + 1]]
                )

            data = {key: numpy.array(data[key]) for key in data}

        else:
            # read cell data
            data = numpy.empty((num_cells, num_nodes_per_cell), dtype=int)
            for k in range(num_cells):
                line = f.readline().decode("utf-8")
                dat = line.split()
                # The body of a regular face section contains the grid connectivity, and
                # each line appears as follows:
                #   n0 n1 n2 cr cl
                # where n* are the defining nodes (vertices) of the face, and c* are the
                # adjacent cells.
                if len(dat) != num_nodes_per_cell + 2:
                    raise ReadError()
                data[k] = [int(d, 16) for d in dat[:num_nodes_per_cell]]
            data = {key: data}
    else:
        # binary
        if out.group(1) == "20":
            dtype = numpy.int32
        else:
            if out.group(1) != "30":
                ReadError("Expected keys '20' or '30', got {}.".format(out.group(1)))
            dtype = numpy.int64

        if key == "mixed":
            raise ReadError("Mixed element type for binary faces not supported yet")

        # Read cell data.
        # The body of a regular face section contains the grid
        # connectivity, and each line appears as follows:
        #   n0 n1 n2 cr cl
        # where n* are the defining nodes (vertices) of the face,
        # and c* are the adjacent cells.
        shape = (num_cells, num_nodes_per_cell + 2)
        count = shape[0] * shape[1]
        data = numpy.fromfile(f, count=count, dtype=dtype).reshape(shape)
        # Cut off the adjacent cell data.
        data = data[:, :num_nodes_per_cell]
        data = {key: data}

    # make sure that the data set is properly closed
    _skip_close(f, 2)

    return data


def read(filename):  # noqa: C901
    # Initialize the data optional data fields
    field_data = {}
    cell_data = {}
    point_data = {}

    points = []
    cells = []

    first_point_index_overall = None
    last_point_index = None

    # read file in binary mode since some data might be binary
    with open_file(filename, "rb") as f:
        while True:
            line = f.readline().decode("utf-8")
            if not line:
                break

            if line.strip() == "":
                continue

            # expect the line to have the form
            #  (<index> [...]
            out = re.match("\\s*\\(\\s*([0-9]+).*", line)
            if not out:
                raise ReadError()
            index = out.group(1)

            if index == "0":
                # Comment.
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "1":
                # header
                # (1 "<text>")
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "2":
                # dimensionality
                # (2 3)
                _skip_close(f, line.count("(") - line.count(")"))
            elif re.match("(|20|30)10", index):
                # points
                pts, first_point_index_overall, last_point_index = _read_points(
                    f, line, first_point_index_overall, last_point_index
                )

                if pts is not None:
                    points.append(pts)

            elif re.match("(|20|30)12", index):
                # cells
                # (2012 (zone-id first-index last-index type element-type))
                key, data = _read_cells(f, line)
                if data is not None:
                    cells.append((key, data))

            elif re.match("(|20|30)13", index):
                data = _read_faces(f, line)

                for key in data:
                    cells.append((key, data[key]))

            elif index == "39":
                logging.warning("Zone specification not supported yet. Skipping.")
                _skip_close(f, line.count("(") - line.count(")"))

            elif index == "45":
                # (45 (2 fluid solid)())
                obj = re.match("\\(45 \\([0-9]+ ([\\S]+) ([\\S]+)\\)\\(\\)\\)", line)
                if obj:
                    logging.warning(
                        "Zone specification not supported yet (%r, %r). " "Skipping.",
                        obj.group(1),
                        obj.group(2),
                    )
                else:
                    logging.warning("Zone specification not supported yet.")

            else:
                logging.warning("Unknown index %r. Skipping.", index)
                # Skipping ahead to the next line with two closing brackets.
                _skip_close(f, line.count("(") - line.count(")"))

    points = numpy.concatenate(points)

    # Gauge the cells with the first point_index.
    for k, c in enumerate(cells):
        cells[k] = (c[0], c[1] - first_point_index_overall)

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def write(filename, mesh, binary=True):
    with open_file(filename, "wb") as fh:
        # header
        fh.write('(1 "meshio {}")\n'.format(__version__).encode("utf8"))

        # dimension
        dim = mesh.points.shape[1]
        if dim not in [2, 3]:
            raise WriteError("Can only write dimension 2, 3, got {}.".format(dim))
        fh.write(("(2 {})\n".format(dim)).encode("utf8"))

        # total number of nodes
        first_node_index = 1
        fh.write(
            (
                "(10 (0 {:x} {:x} 0))\n".format(first_node_index, len(mesh.points))
            ).encode("utf8")
        )

        # total number of cells
        total_num_cells = sum([len(c) for c in mesh.cells])
        fh.write(("(12 (0 1 {:x} 0))\n".format(total_num_cells)).encode("utf8"))

        # Write nodes
        key = "3010" if binary else "10"
        fh.write(
            (
                "({} (1 {:x} {:x} 1 {:x})(\n".format(
                    key, first_node_index, mesh.points.shape[0], mesh.points.shape[1]
                )
            ).encode("utf8")
        )
        if binary:
            mesh.points.tofile(fh)
            fh.write(b"\n)")
            fh.write(b"End of Binary Section 3010)\n")
        else:
            numpy.savetxt(fh, mesh.points, fmt="%.16e")
            fh.write(b"))\n")

        # Write cells
        meshio_to_ansys_type = {
            "triangle": 1,
            "tetra": 2,
            "quad": 3,
            "hexahedron": 4,
            "pyra": 5,
            "wedge": 6,
        }
        first_index = 0
        binary_dtypes = {
            # numpy.int16 is not allowed
            numpy.dtype("int32"): "2012",
            numpy.dtype("int64"): "3012",
        }
        for cell_type, values in mesh.cells:
            key = binary_dtypes[values.dtype] if binary else "12"
            last_index = first_index + len(values) - 1
            fh.write(
                (
                    "({} (1 {:x} {:x} 1 {})(\n".format(
                        key, first_index, last_index, meshio_to_ansys_type[cell_type]
                    )
                ).encode("utf8")
            )
            if binary:
                (values + first_node_index).tofile(fh)
                fh.write(b"\n)")
                fh.write(("End of Binary Section {})\n".format(key)).encode("utf8"))
            else:
                numpy.savetxt(fh, values + first_node_index, fmt="%x")
                fh.write(b"))\n")
            first_index = last_index + 1


register("ansys", [], read, {"ansys": write})
