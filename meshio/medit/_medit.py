"""
I/O for Medit's format/Gamma Mesh Format,
Latest official up-to-date documentation and a reference C implementation at
<https://github.com/LoicMarechal/libMeshb>
"""
import logging
import struct
from ctypes import c_double, c_float

import numpy

from .._common import _pick_first_int_data
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh
from ._medit_internal import medit_codes


def read(filename):

    with open_file(filename) as f:
        if filename[-1] == "b":
            mesh = read_binary_buffer(f)
        else:
            mesh = read_ascii_buffer(f)
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
            raise ReadError("Invalid string type")
        c += 1
        if c != len(string_type):
            res += ","
    return res


def read_binary_buffer(f):

    meshio_from_medit = {
        "GmfVertices": ("point", None),
        "GmfEdges": ("line", 2),
        "GmfTriangles": ("triangle", 3),
        "GmfQuadrilaterals": ("quad", 4),
        "GmfTetrahedra": ("tetra", 4),
        "GmfPrisms": ("wedge", 6),
        "GmfPyramids": ("pyramid", 5),
        "GmfHexahedra": ("hexahedron", 8),
    }

    dim = 0
    points = None
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}
    itype = ""
    ftype = ""
    postype = ""
    # the file version
    keytype = "i4"

    code = numpy.fromfile(f, count=1, dtype=keytype).item()

    if code != 1 and code != 16777216:
        raise ReadError("Invalid code")

    if code == 16777216:
        # swap endianess
        swapped = ">" if struct.unpack("=l", struct.pack("<l", 1))[0] == 1 else "<"
        itype += swapped
        ftype += swapped
        postype += swapped
        keytype = swapped + keytype

    version = numpy.fromfile(f, count=1, dtype=keytype).item()

    if version < 1 or version > 4:
        raise ReadError("Invalid version")

    if version == 1:
        itype += "i4"
        ftype += "f4"
        postype += "i4"
    elif version == 2:
        itype += "i4"
        ftype += "f8"
        postype += "i4"
    elif version == 3:
        itype += "i4"
        ftype += "f8"
        postype += "i8"
    else:
        itype += "i8"
        ftype += "f8"
        postype += "i8"

    field = numpy.fromfile(f, count=1, dtype=keytype).item()

    if field != 3:  # =  GmfDimension
        raise ReadError("Invalid dimension code : " + str(field) + " it should be 3")

    numpy.fromfile(f, count=1, dtype=postype)

    dim = numpy.fromfile(f, count=1, dtype=keytype).item()

    if dim != 2 and dim != 3:
        raise ReadError("Invalid mesh dimension : " + str(dim))

    while True:
        field = numpy.fromfile(f, count=1, dtype=keytype)

        if field.size == 0:
            msg = "End-of-file reached before GmfEnd keyword"
            logging.warning(msg)
            break

        field = field.item()
        if field not in medit_codes.keys():
            raise ReadError("Unsupported field")

        field_code = medit_codes[field]

        if field_code[0] == "GmfEnd":
            break

        if field_code[0] == "GmfReserved":
            continue

        numpy.fromfile(f, count=1, dtype=postype)

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
            cell_data["medit:ref"].append(out_view[:, -1])

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def read_ascii_buffer(f):
    dim = 0
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}

    meshio_from_medit = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Prisms": ("wedge", 6),
        "Pyramids": ("pyramid", 5),
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
            if len(items) >= 2:
                dim = int(items[1])
            else:
                dim = int(
                    int(f.readline())
                )  # e.g. Dimension\n3, where the number of dimensions is on the next line
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


def write(filename, mesh, float_fmt=".16e"):
    if filename[-1] == "b":
        write_binary_file(filename, mesh)
    else:
        write_ascii_file(filename, mesh, float_fmt)


def write_ascii_file(filename, mesh, float_fmt=".16e"):
    with open_file(filename, "wb") as fh:
        version = {numpy.dtype(c_float): 1, numpy.dtype(c_double): 2}[mesh.points.dtype]
        # N. B.: PEP 461 Adding % formatting to bytes and bytearray
        fh.write(b"MeshVersionFormatted %d\n" % version)

        n, d = mesh.points.shape

        fh.write(b"Dimension %d\n" % d)

        # vertices
        fh.write(b"\nVertices\n")
        fh.write(f"{n}\n".encode("utf-8"))

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
            "wedge": ("Prisms", 6),
            "pyramid": ("Pyramids", 5),
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
            fh.write(f"{medit_name}\n".encode("utf-8"))
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


def write_binary_file(f, mesh):
    with open_file(f, "wb") as fh:

        version = 3
        itype = "i4"
        postype = "i8"
        ftype = "f8"
        # according to manual keywords are always written as i4 independently of
        # the file version
        keytype = "i4"

        # if we store internally 64bit integers upgrade file version
        has_big_ints = False
        for _, data in mesh.cells:
            if data.dtype.itemsize == 8:
                has_big_ints = True
                break

        if has_big_ints:
            itype = "i8"
            version = 4

        itype_size = numpy.dtype(itype).itemsize
        ftype_size = numpy.dtype(ftype).itemsize
        postype_size = numpy.dtype(postype).itemsize
        keyword_size = numpy.dtype(keytype).itemsize

        code = 1
        field = 3  # GmfDimension
        pos = 4 * keyword_size + postype_size

        num_verts, dim = mesh.points.shape

        header_type = numpy.dtype(
            ",".join([keytype, keytype, keytype, postype, keytype])
        )
        tmp_array = numpy.empty(1, dtype=header_type)
        tmp_array["f0"] = code
        tmp_array["f1"] = version
        tmp_array["f2"] = field
        tmp_array["f3"] = pos
        tmp_array["f4"] = dim
        tmp_array.tofile(fh)

        # write points
        field = 4  # GmfVertices
        field_code = medit_codes[field]

        pos += num_verts * dim * ftype_size
        pos += num_verts * itype_size
        pos += keyword_size + postype_size + itype_size
        header_type = numpy.dtype(",".join([keytype, postype, itype]))
        tmp_array = numpy.empty(1, dtype=header_type)
        tmp_array["f0"] = field
        tmp_array["f1"] = pos
        tmp_array["f2"] = num_verts
        tmp_array.tofile(fh)

        field_template = field_code[2]
        dtype = numpy.dtype(_produce_dtype(field_template, dim, itype, ftype))

        labels_key, other = _pick_first_int_data(mesh.point_data)
        if labels_key and other:
            logging.warning(
                "Medit can only write one point data array. "
                "Picking {}, skipping {}.".format(labels_key, ", ".join(other))
            )
        labels = (
            mesh.point_data[labels_key]
            if labels_key
            else numpy.ones(num_verts, dtype=itype)
        )

        tmp_array = numpy.empty(num_verts, dtype=dtype)
        tmp_array["f0"] = mesh.points
        tmp_array["f1"] = labels
        tmp_array.tofile(fh)

        labels_key, other = _pick_first_int_data(mesh.cell_data)
        if labels_key and other:
            logging.warning(
                "Medit can only write one cell data array. "
                "Picking {}, skipping {}.".format(labels_key, ", ".join(other))
            )

        # first component is medit keyword id see _medit_internal.py
        medit_from_meshio = {
            "line": 5,
            "triangle": 6,
            "quad": 7,
            "tetra": 8,
            "wedge": 9,
            "pyramid": 49,
            "hexahedron": 10,
        }
        for k, (cell_type, data) in enumerate(mesh.cells):
            try:
                medit_key = medit_from_meshio[cell_type]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    cell_type
                )
                logging.warning(msg)
                continue

            num_cells, num_verts = data.shape

            pos += num_cells * (num_verts + 1) * itype_size
            pos += keyword_size + postype_size + itype_size

            header_type = numpy.dtype(",".join([keytype, postype, itype]))
            tmp_array = numpy.empty(1, dtype=header_type)
            tmp_array["f0"] = medit_key
            tmp_array["f1"] = pos
            tmp_array["f2"] = num_cells
            tmp_array.tofile(fh)

            # pick out cell data
            labels = (
                mesh.cell_data[labels_key][k]
                if labels_key
                else numpy.ones(len(data), dtype=data.dtype)
            )
            field_template = medit_codes[medit_key][2]
            dtype = numpy.dtype(_produce_dtype(field_template, dim, itype, ftype))

            tmp_array = numpy.empty(num_cells, dtype=dtype)
            i = 0
            for col_type in dtype.names[:-1]:
                tmp_array[col_type] = data[:, i] + 1
                i += 1

            tmp_array[dtype.names[-1]] = labels
            tmp_array.tofile(fh)

        pos = 0
        field = 54  # GmfEnd
        header_type = numpy.dtype(",".join([keytype, postype]))
        tmp_array = numpy.empty(1, dtype=header_type)
        tmp_array["f0"] = field
        tmp_array["f1"] = pos
        tmp_array.tofile(fh)


register("medit", [".mesh", ".meshb"], read, {"medit": write})
