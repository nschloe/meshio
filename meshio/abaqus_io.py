# -*- coding: utf-8 -*-
#
"""
I/O for Abaqus inp files.
"""
import numpy

from .__about__ import __version__
from .mesh import Mesh


abaqus_to_meshio_type = {
    # trusss
    "T2D2": "line",
    "T2D2H": "line",
    "T2D3": "line3",
    "T2D3H": "line3",
    "T3D2": "line",
    "T3D2H": "line",
    "T3D3": "line3",
    "T3D3H": "line3",
    # beams
    "B21": "line",
    "B21H": "line",
    "B22": "line3",
    "B22H": "line3",
    "B31": "line",
    "B31H": "line",
    "B32": "line3",
    "B32H": "line3",
    "B33": "line3",
    "B33H": "line3",
    # surfaces
    "S4": "quad",
    "S4R": "quad",
    "S4RS": "quad",
    "S4RSW": "quad",
    "S4R5": "quad",
    "S8R": "quad8",
    "S8R5": "quad8",
    "S9R5": "quad9",
    # "QUAD": "quad",
    # "QUAD4": "quad",
    # "QUAD5": "quad5",
    # "QUAD8": "quad8",
    # "QUAD9": "quad9",
    #
    "STRI3": "triangle",
    "S3": "triangle",
    "S3R": "triangle",
    "S3RS": "triangle",
    # "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    "STRI65": "triangle6",
    # 'TRISHELL6': 'triangle6',
    # volumes
    "C3D8": "hexahedron",
    "C3D8H": "hexahedron",
    "C3D8I": "hexahedron",
    "C3D8IH": "hexahedron",
    "C3D8R": "hexahedron",
    "C3D8RH": "hexahedron",
    # "HEX9": "hexahedron9",
    "C3D20": "hexahedron20",
    "C3D20H": "hexahedron20",
    "C3D20R": "hexahedron20",
    "C3D20RH": "hexahedron20",
    # "HEX27": "hexahedron27",
    #
    "C3D4": "tetra",
    "C3D4H": "tetra4",
    # "TETRA8": "tetra8",
    "C3D10": "tetra10",
    "C3D10H": "tetra10",
    "C3D10I": "tetra10",
    "C3D10M": "tetra10",
    "C3D10MH": "tetra10",
    # "TETRA14": "tetra14",
    #
    # "PYRAMID": "pyramid",
    "C3D6": "wedge",
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}


def read(filename):
    """Reads a Abaqus inp file.
    """
    with open(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Initialize the optional data fields
    cells = {}
    nsets = {}
    elsets = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        if line.startswith("*"):
            word = line.strip("*").upper()
            if word == "HEADING":
                pass
            elif word.startswith("PREPRINT"):
                pass
            elif word.startswith("NODE"):
                points, point_gids = _read_nodes(f)
            elif word.startswith("ELEMENT"):
                key, idx = _read_cells(f, word)
                cells[key] = idx
            elif word.startswith("NSET"):
                params_map = get_param_map(word, required_keys=["NSET"])
                setids = read_set(f, params_map)
                name = params_map["NSET"]
                if name not in nsets:
                    nsets[name] = []
                nsets[name].append(setids)
            elif word.startswith("ELSET"):
                params_map = get_param_map(word, required_keys=["ELSET"])
                setids = read_set(f, params_map)
                name = params_map["ELSET"]
                if name not in elsets:
                    elsets[name] = []
                elsets[name].append(setids)
            else:
                pass

    cells = _scan_cells(point_gids, cells)

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def _read_nodes(f):
    points = []
    point_gids = []
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line.startswith("*"):
            break
        gid, *x = line.strip().split(",")
        point_gids.append(int(gid))
        points.append([float(xx) for xx in x])

    f.seek(last_pos)
    return numpy.array(points, dtype=float), numpy.array(point_gids, dtype=int)


def _read_cells(f, line0):
    sline = line0.split(",")[1:]
    etype_sline = sline[0]
    assert "TYPE" in etype_sline, etype_sline
    etype = etype_sline.split("=")[1].strip()
    assert etype in abaqus_to_meshio_type, "Element type not available: {}".format(
        etype
    )
    cell_type = abaqus_to_meshio_type[etype]

    cells = []
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line.startswith("*"):
            break
        _, *idx = [int(k) for k in filter(None, line.split(","))]
        cells.append(idx)

    f.seek(last_pos)
    return cell_type, numpy.array(cells)


def _scan_cells(point_gids, cells):
    for arr in cells.values():
        for value in numpy.nditer(arr, op_flags=["readwrite"]):
            value[...] = numpy.flatnonzero(point_gids == value)[0]
    return cells


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> iline = 0
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(iline, word, required_keys=['instance'])
    params = {
        'elset' : None,
        'instance' : 'dummy2,
        'generate' : None,
    }
    """
    if required_keys is None:
        required_keys = []
    words = word.split(",")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip()
            value = None
        else:
            sword = wordi.split("=")
            assert len(sword) == 2, sword
            key = sword[0].strip()
            value = sword[1].strip()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += "%r not found in %r\n" % (key, word)
    if msg:
        raise RuntimeError(msg)
    return param_map


def read_set(f, params_map):
    set_ids = []
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line.startswith("*"):
            break
        set_ids += [int(k) for k in line.strip().strip(",").split(",")]
    f.seek(last_pos)

    if "generate" in params_map:
        assert len(set_ids) == 3, set_ids
        set_ids = numpy.arange(set_ids[0], set_ids[1], set_ids[2])
    else:
        try:
            set_ids = numpy.unique(numpy.array(set_ids, dtype="int32"))
        except ValueError:
            print(set_ids)
            raise
    return set_ids


def write(filename, mesh):
    with open(filename, "wt") as f:
        f.write("*Heading\n")
        f.write(" Abaqus DataFile Version 6.14\n")
        f.write("written by meshio v{}\n".format(__version__))
        f.write("*Node\n")
        for k, x in enumerate(mesh.points):
            f.write("{}, {!r}, {!r}, {!r}\n".format(k + 1, x[0], x[1], x[2]))
        eid = 0
        for cell_type, node_idcs in mesh.cells.items():
            f.write("*Element,type=" + meshio_to_abaqus_type[cell_type] + "\n")
            for row in node_idcs:
                eid += 1
                nids_strs = (str(nid + 1) for nid in row.tolist())
                f.write(str(eid) + "," + ",".join(nids_strs) + "\n")
        f.write("*end")
