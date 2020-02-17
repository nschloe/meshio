"""
I/O for Abaqus inp files.
"""
import numpy

from ..__about__ import __version__
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Cells, Mesh

abaqus_to_meshio_type = {
    # trusses
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
    "CPS4": "quad",
    "CPS4R": "quad",
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
    "CPS3": "triangle",
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
    #
    # 4-node bilinear displacement and pore pressure
    "CAX4P": "quad",
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}


def read(filename):
    """Reads a Abaqus inp file.
    """
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Initialize the optional data fields
    cells = []
    cell_ids = []
    point_sets = {}
    cell_sets = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    line = f.readline()
    while True:
        if not line:  # EOF
            break

        # Comments
        if line.startswith("**"):
            line = f.readline()
            continue

        keyword = line.partition(",")[0].strip().replace("*", "").upper()
        if keyword == "NODE":
            points, point_ids, line = _read_nodes(f)
        elif keyword == "ELEMENT":
            cell_type, cells_data, ids, line = _read_cells(f, line, point_ids)
            cells.append(Cells(cell_type, cells_data))
            cell_ids.append(ids)
        elif keyword == "NSET":
            params_map = get_param_map(line, required_keys=["NSET"])
            set_ids, line = _read_set(f, params_map)
            name = params_map["NSET"]
            point_sets[name] = numpy.array(
                [point_ids[point_id] for point_id in set_ids], dtype="int32"
            )
        elif keyword == "ELSET":
            params_map = get_param_map(line, required_keys=["ELSET"])
            set_ids, line = _read_set(f, params_map)
            name = params_map["ELSET"]
            cell_sets[name] = []
            for cell_ids_ in cell_ids:
                cell_sets_ = numpy.array(
                    [cell_ids_[set_id] for set_id in set_ids if set_id in cell_ids_],
                    dtype="int32",
                )
                cell_sets[name].append(cell_sets_)
        else:
            # There are just too many Abaqus keywords to explicitly skip them.
            line = f.readline()

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )


def _read_nodes(f):
    points = []
    point_ids = {}
    counter = 0
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip().split(",")
        point_id, coords = line[0], line[1:]
        point_ids[int(point_id)] = counter
        points.append([float(x) for x in coords])
        counter += 1

    return numpy.array(points, dtype=float), point_ids, line


def _read_cells(f, line0, point_ids):
    sline = line0.split(",")[1:]

    etype_sline = sline[0].upper()
    if "TYPE" not in etype_sline:
        raise ReadError(etype_sline)

    etype = etype_sline.split("=")[1].strip()
    if etype not in abaqus_to_meshio_type:
        raise ReadError("Element type not available: {}".format(etype))

    cell_type = abaqus_to_meshio_type[etype]

    cells, idx = [], []
    cell_ids = {}
    counter = 0
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip()
        idx += [int(k) for k in filter(None, line.split(","))]
        if not line.endswith(","):
            cell_ids[idx[0]] = counter
            cells.append([point_ids[k] for k in idx[1:]])
            idx = []
            counter += 1
    return cell_type, numpy.array(cells), cell_ids, line


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(word, required_keys=['instance'])
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
            key = wordi.strip().upper()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip().upper()
            value = sword[1].strip()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += "{} not found in {}\n".format(key, word)
    if msg:
        raise RuntimeError(msg)
    return param_map


def _read_set(f, params_map):
    set_ids = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        set_ids += [int(k) for k in line.strip().strip(",").split(",")]

    set_ids = numpy.array(set_ids, dtype="int32")
    if "GENERATE" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = numpy.arange(set_ids[0], set_ids[1] + 1, set_ids[2], dtype="int32")
    return set_ids, line


def write(filename, mesh, float_fmt=".15e", translate_cell_names=True):
    with open_file(filename, "wt") as f:
        f.write("*Heading\n")
        f.write("Abaqus DataFile Version 6.14\n")
        f.write("written by meshio v{}\n".format(__version__))
        f.write("*Node\n")
        fmt = ", ".join(["{}"] + ["{:" + float_fmt + "}"] * mesh.points.shape[1]) + "\n"
        for k, x in enumerate(mesh.points):
            f.write(fmt.format(k + 1, *x))
        eid = 0
        for cell_type, node_idcs in mesh.cells:
            name = (
                meshio_to_abaqus_type[cell_type] if translate_cell_names else cell_type
            )
            f.write("*Element,type=" + name + "\n")
            for row in node_idcs:
                eid += 1
                nids_strs = (str(nid + 1) for nid in row.tolist())
                f.write(str(eid) + "," + ",".join(nids_strs) + "\n")

        nnl = 8
        for ic in range(len(mesh.cells)):
            for k, v in mesh.cell_sets.items():
                els = [str(i + 1) for i in v[ic]]
                f.write("*ELSET, ELSET=%s\n" % k)
                f.write(
                    ",\n".join(
                        ",".join(els[i : i + nnl]) for i in range(0, len(els), nnl)
                    )
                    + "\n"
                )

        for k, v in mesh.point_sets.items():
            nds = [str(i + 1) for i in v]
            f.write("*NSET, NSET=%s\n" % k)
            f.write(
                ",\n".join(",".join(nds[i : i + nnl]) for i in range(0, len(nds), nnl))
                + "\n"
            )

        f.write("*end")


register("abaqus", [".inp"], read, {"abaqus": write})
