"""
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
<https://www.math.u-bordeaux.fr/~dobrzyns/logiciels/RT-422/node58.html>
for something like a specification.
Latest official up-to-date documentation and reference implementation at
<https://github.com/LoicMarechal/libMeshb>
"""
import logging
from ctypes import c_double, c_float
from inspect import currentframe

import numpy

from .._common import _pick_first_int_data
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh
from ._medit_internal import medit_codes

# map medit element types to meshio
meshio_from_medit = {
    "GmfVertices": ("point", None),
    "GmfEdges": ("line", 2),
    "GmfTriangles": ("triangle", 3),
    "GmfQuadrilaterals": ("quad", 4),
    "GmfTetrahedra": ("tetra", 4),
    "GmfPrisms": ("wedge", 6),
    "GmfHexahedra": ("hexahedron", 8),
}


def read(filename):

    with open_file(filename) as f:
        if filename[-1] == "b":
            mesh = read_binary_buffer(f)
        else:
            mesh = read_buffer(f)
    return mesh


def _produce_dtype(string_type, dim, itype, ftype):
    """
    convert a medit_code to a dtype appropriate for building a numpy array
    """
    res = ""
    c = 0
    while c < len(string_type):
        s = string_type[c]
        if s == "i":
            res += itype
        elif s == "r":
            res += ftype
        elif s == "d":
            res += str(dim)
            c += 1
            continue
        else:
            ReadError("Invalid string type")
        c += 1
        if c != len(string_type):
            res += ","
    return res


def debug(variable):
    print(variable, "--->", repr(eval(variable)))


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def read_binary_buffer(f):
    dim = 0
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}
    itype = ""
    ftype = ""
    postype = ""

    code = numpy.fromfile(f, count=1, dtype="i4").item()

    if code != 1 and code != 16777216:
        raise ReadError("Invalid code")

    print("% : % ", get_linenumber(), code)

    if code == 16777216:
        # swap endianess
        itype += "S"
        ftype += "S"
        postype += "S"

    version = numpy.fromfile(f, count=1, dtype="i4")

    if version < 1 or version > 4:
        raise ReadError("Invalid version")

    if version == 1:
        itype += "i4"
        ftype += "f4"
    elif version == 2 or version == 3:
        itype += "i4"
        ftype += "f8"
    else:
        itype += "i8"
        ftype += "f8"

    postype += "i4"

    points = None
    cells = []
    point_data = dict()
    celldata = None

    field = numpy.fromfile(f, count=1, dtype=postype).item()

    if field != 3:  # =  GmfDimension
        raise ReadError("Invalid dimension code")

    pos = numpy.fromfile(f, count=1, dtype=postype)
    print("%d : %d " % (get_linenumber(), pos))

    dim = numpy.fromfile(f, count=1, dtype=postype).item()
    print("%d : %d " % (get_linenumber(), dim))

    if dim != 2 and dim != 3:
        raise ReadError("Invalid mesh dimension : " + str(dim))

    while True:
        field = numpy.fromfile(f, count=1, dtype=postype)

        if field.size == 0:
            msg = "End-of-file reached before GmfEnd keyword"
            logging.warning(msg)
            break

        field = field.item()
        if field not in medit_codes.keys():
            raise ReadError("Unsupported field")

        field_code = medit_codes[field]
        print(field_code)

        if field_code[0] == "GmfEnd":
            break

        if field_code[0] == "GmfReserved":
            continue

        pos = numpy.fromfile(f, count=1, dtype=postype)

        nitems = 1
        if field_code[1] == "i":
            nitems = numpy.fromfile(f, count=1, dtype=itype).item()

        field_template = field_code[2]
        dtype = numpy.dtype(_produce_dtype(field_template, dim, itype, ftype))
        out = numpy.asarray(numpy.fromfile(f, count=nitems, dtype=dtype))
        if field_code[0] not in meshio_from_medit.keys():
            msg = ("meshio doesn't know {} type. Skipping.").format(field_code[0])
            logging.warning(msg)
            continue

        elif field_code[0] == "GmfVertices":
            points = out["f0"]
            point_data["medit:ref"] = out["f1"]
        else:
            meshio_type, ncols = meshio_from_medit[field_code[0]]
            # transform the structured array to integer array which suffices
            # for the cell connectivity
            out_view = out.view(itype).reshape(nitems, ncols + 1)
            cells.append((meshio_type, out_view[:, :ncols] - 1))
            cell_data["medit:ref"].append(out_view[:,-1])

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def read_buffer(f):
    dim = 0
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}

    meshio_from_medit = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Hexahedra": ("hexahedron", 8),  # Frey
        "Hexaedra": ("hexahedron", 8),  # Dobrzynski
    }

    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue

        items = line.split()

        if not items[0].isalpha():
            raise ReadError()

        if items[0] == "MeshVersionFormatted":
            version = items[1]
            dtype = {"1": c_float, "2": c_double}[version]
        elif items[0] == "Dimension":
            dim = int(items[1])
        elif items[0] == "Vertices":
            if dim <= 0:
                raise ReadError()
            num_verts = int(f.readline())
            out = numpy.fromfile(
                f, count=num_verts * (dim + 1), dtype=dtype, sep=" "
            ).reshape(num_verts, dim + 1)
            points = out[:, :dim]
            point_data["medit:ref"] = out[:, dim].astype(int)
        elif items[0] in meshio_from_medit:
            meshio_type, points_per_cell = meshio_from_medit[items[0]]
            # The first value is the number of elements
            num_cells = int(f.readline())

            out = numpy.fromfile(
                f, count=num_cells * (points_per_cell + 1), dtype=int, sep=" "
            ).reshape(num_cells, points_per_cell + 1)

            # adapt for 0-base
            cells.append((meshio_type, out[:, :points_per_cell] - 1))
            cell_data["medit:ref"].append(out[:, -1])
        elif items[0] == "Normals":
            # those are just discarded
            num_normals = int(f.readline())
            numpy.fromfile(f, count=num_normals * dim, dtype=dtype, sep=" ").reshape(
                num_normals, dim
            )
        elif items[0] == "NormalAtVertices":
            # those are just discarded
            num_normal_at_vertices = int(f.readline())
            numpy.fromfile(
                f, count=num_normal_at_vertices * 2, dtype=int, sep=" "
            ).reshape(num_normal_at_vertices, 2)
        else:
            if items[0] != "End":
                raise ReadError("Unknown keyword '{}'.".format(items[0]))

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh, float_fmt=".15e"):
    with open_file(filename, "wb") as fh:
        version = {numpy.dtype(c_float): 1, numpy.dtype(c_double): 2}[mesh.points.dtype]
        # N. B.: PEP 461 Adding % formatting to bytes and bytearray
        fh.write(b"MeshVersionFormatted %d\n" % version)

        n, d = mesh.points.shape

        fh.write(b"Dimension %d\n" % d)

        # vertices
        fh.write(b"\nVertices\n")
        fh.write("{}\n".format(n).encode("utf-8"))

        # pick out point data
        labels_key, other = _pick_first_int_data(mesh.point_data)
        if labels_key and other:
            logging.warning(
                "Medit can only write one point data array. "
                "Picking {}, skipping {}.".format(labels_key, ", ".join(other))
            )
        labels = mesh.point_data[labels_key] if labels_key else numpy.ones(n, dtype=int)

        fmt = " ".join(["{:" + float_fmt + "}"] * d) + " {:d}\n"
        for x, label in zip(mesh.points, labels):
            fh.write(fmt.format(*x, label).encode("utf-8"))

        medit_from_meshio = {
            "line": ("Edges", 2),
            "triangle": ("Triangles", 3),
            "quad": ("Quadrilaterals", 4),
            "tetra": ("Tetrahedra", 4),
            "hexahedron": ("Hexahedra", 8),
        }

        # pick out cell_data
        labels_key, other = _pick_first_int_data(mesh.cell_data)
        if labels_key and other:
            logging.warning(
                "Medit can only write one cell data array. "
                "Picking {}, skipping {}.".format(labels_key, ", ".join(other))
            )

        for k, (cell_type, data) in enumerate(mesh.cells):
            try:
                medit_name, num = medit_from_meshio[cell_type]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    cell_type
                )
                logging.warning(msg)
                continue
            fh.write(b"\n")
            fh.write("{}\n".format(medit_name).encode("utf-8"))
            fh.write("{}\n".format(len(data)).encode("utf-8"))

            # pick out cell data
            labels = (
                mesh.cell_data[labels_key][k]
                if labels_key
                else numpy.ones(len(data), dtype=data.dtype)
            )

            fmt = " ".join(["{:d}"] * (num + 1)) + "\n"
            # adapt 1-base
            for d, label in zip(data + 1, labels):
                fh.write(fmt.format(*d, label).encode("utf-8"))

        fh.write(b"\nEnd\n")


register("medit", [".mesh", ".meshb"], read, {"medit": write})
