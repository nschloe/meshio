"""
I/O for Abaqus inp files.
"""
import numpy

from ..__about__ import __version__
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh, Cells

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

        keyword = line.strip("*")
        if keyword.upper().startswith("NODE"):
            points, point_ids, line = _read_nodes(f)
        elif keyword.upper().startswith("ELEMENT"):
            key, idx, line = _read_cells(f, keyword, point_ids)
            cells.append(Cells(key, idx))
        elif keyword.upper().startswith("NSET"):
            params_map = get_param_map(keyword, required_keys=["NSET"])
            set_ids, line = _read_set(f, params_map)
            name = params_map["NSET"]
            point_sets[name] = numpy.array(
                [point_ids[point_id] for point_id in set_ids], dtype="int32"
            )
        elif keyword.upper().startswith("ELSET"):
            params_map = get_param_map(keyword, required_keys=["ELSET"])
            setids, line = _read_set(f, params_map)
            name = params_map["ELSET"]
            if name not in cell_sets:
                cell_sets[name] = []
            cell_sets[name].append(setids)
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
    )


def _read_nodes(f):
    points = []
    point_ids = {}
    index = 0
    while True:
        line = f.readline()
        if line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip().split(",")
        point_id, coords = line[0], line[1:]
        point_ids[int(point_id)] = index
        points.append([float(x) for x in coords])
        index += 1

    return numpy.array(points, dtype=float), point_ids, line


def _read_cells(f, line0, point_ids):
    sline = line0.split(",")[1:]

    etype_sline = sline[0].upper()
    if "TYPE" not in etype_sline:
        raise ReadError(etype_sline)

    etype = etype_sline.split("=")[1].strip()
    if etype not in abaqus_to_meshio_type:
        raise ReadError(f"Element type not available: {etype}")

    cell_type = abaqus_to_meshio_type[etype]

    cells, idx = [], []
    while True:
        line = f.readline()
        if line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip()
        idx += [int(k) for k in filter(None, line.split(","))]
        if not line.endswith(","):
            # the first item is just a running index
            cells.append([point_ids[k] for k in idx[1:]])
            idx = []
    return cell_type, numpy.array(cells), line


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
            key = wordi.strip()
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
            msg += f"{key!r} not found in {word!r}\n"
    if msg:
        raise RuntimeError(msg)
    return param_map


def _read_set(f, params_map):
    set_ids = []
    while True:
        line = f.readline()
        if line.startswith("*"):
            break
        if line.strip() == "":
            continue

        set_ids += [int(k) for k in line.strip().strip(",").split(",")]

    if "generate" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = numpy.arange(set_ids[0], set_ids[1], set_ids[2])
    else:
        try:
            set_ids = numpy.unique(numpy.array(set_ids, dtype="int32"))
        except ValueError:
            raise ReadError(set_ids)
    return set_ids, line


def write(filename, mesh, translate_cell_names=True):
    with open_file(filename, "wt") as f:
        f.write("*Heading\n")
        f.write("Abaqus DataFile Version 6.14\n")
        f.write(f"written by meshio v{__version__}\n")
        f.write("*Node\n")
        fmt = ", ".join(["{}"] + ["{!r}"] * mesh.points.shape[1]) + "\n"
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
        f.write("*end")


register("abaqus", [".inp"], read, {"abaqus": write})
