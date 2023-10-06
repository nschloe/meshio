from __future__ import annotations

import shlex

import numpy as np
from numpy.typing import ArrayLike

from .._common import warn
from .._exceptions import ReadError, WriteError

c_int = np.dtype("int32")
c_double = np.dtype("float64")


def _fast_forward_to_end_block(f, block):
    """fast-forward to end of block"""
    # See also https://github.com/nschloe/pygalmesh/issues/34

    for line in f:
        try:
            line = line.decode()
        except UnicodeDecodeError:
            pass
        if line.strip() == f"$End{block}":
            break
    else:
        warn(f"${block} not closed by $End{block}.")


def _fast_forward_over_blank_lines(f):
    is_eof = False
    while True:
        line = f.readline().decode()
        if not line:
            is_eof = True
            break
        elif len(line.strip()) > 0:
            break
    return line, is_eof


def _read_physical_names(f, field_data):
    line = f.readline().decode()
    num_phys_names = int(line)
    for _ in range(num_phys_names):
        line = shlex.split(f.readline().decode())
        key = line[2]
        value = np.array(line[1::-1], dtype=int)
        field_data[key] = value
    _fast_forward_to_end_block(f, "PhysicalNames")


def _read_data(f, tag, data_dict, data_size, is_ascii):
    # Read string tags
    num_string_tags = int(f.readline().decode())
    string_tags = [
        f.readline().decode().strip().replace('"', "") for _ in range(num_string_tags)
    ]
    # The real tags typically only contain one value, the time.
    # Discard it.
    num_real_tags = int(f.readline().decode())
    for _ in range(num_real_tags):
        f.readline()
    num_integer_tags = int(f.readline().decode())
    integer_tags = [int(f.readline().decode()) for _ in range(num_integer_tags)]
    num_components = integer_tags[1]
    num_items = integer_tags[2]
    if is_ascii:
        data = np.fromfile(f, count=num_items * (1 + num_components), sep=" ").reshape(
            (num_items, 1 + num_components)
        )
        # The first entry is the node number
        data = data[:, 1:]
    else:
        # binary
        dtype = [("index", c_int), ("values", c_double, (num_components,))]
        data = np.fromfile(f, count=num_items, dtype=dtype)
        if not (data["index"] == range(1, num_items + 1)).all():
            raise ReadError()
        data = np.ascontiguousarray(data["values"])

    _fast_forward_to_end_block(f, tag)

    # The gmsh format cannot distinguish between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data


# Translate meshio types to gmsh codes
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-version-2
_gmsh_to_meshio_type = {
    1: "line",
    2: "triangle",
    3: "quad",
    4: "tetra",
    5: "hexahedron",
    6: "wedge",
    7: "pyramid",
    8: "line3",
    9: "triangle6",
    10: "quad9",
    11: "tetra10",
    12: "hexahedron27",
    13: "wedge18",
    14: "pyramid14",
    15: "vertex",
    16: "quad8",
    17: "hexahedron20",
    18: "wedge15",
    19: "pyramid13",
    21: "triangle10",
    23: "triangle15",
    25: "triangle21",
    26: "line4",
    27: "line5",
    28: "line6",
    29: "tetra20",
    30: "tetra35",
    31: "tetra56",
    36: "quad16",
    37: "quad25",
    38: "quad36",
    42: "triangle28",
    43: "triangle36",
    44: "triangle45",
    45: "triangle55",
    46: "triangle66",
    47: "quad49",
    48: "quad64",
    49: "quad81",
    50: "quad100",
    51: "quad121",
    62: "line7",
    63: "line8",
    64: "line9",
    65: "line10",
    66: "line11",
    71: "tetra84",
    72: "tetra120",
    73: "tetra165",
    74: "tetra220",
    75: "tetra286",
    90: "wedge40",
    91: "wedge75",
    92: "hexahedron64",
    93: "hexahedron125",
    94: "hexahedron216",
    95: "hexahedron343",
    96: "hexahedron512",
    97: "hexahedron729",
    98: "hexahedron1000",
    106: "wedge126",
    107: "wedge196",
    108: "wedge288",
    109: "wedge405",
    110: "wedge550",
}
_meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}


def _gmsh_to_meshio_order(cell_type: str, idx: ArrayLike) -> np.ndarray:
    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    meshio_ordering = {
        # fmt: off
        "tetra10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        "hexahedron20": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13,
            9, 16, 18, 19, 17, 10, 12, 14, 15,
        ],  # https://vtk.org/doc/release/4.2/html/classvtkQuadraticHexahedron.html and https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
        "hexahedron27": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13,
            9, 16, 18, 19, 17, 10, 12, 14, 15,
            22, 23, 21, 24, 20, 25, 26,
        ],
        "wedge15": [
            0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11
        ],  # http://davis.lbl.gov/Manuals/VTK-4.5/classvtkQuadraticWedge.html and https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
        "pyramid13": [0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12],
        # fmt: on
    }
    idx = np.asarray(idx)
    if cell_type not in meshio_ordering:
        return idx
    return idx[:, meshio_ordering[cell_type]]


def _meshio_to_gmsh_order(cell_type: str, idx: ArrayLike) -> np.ndarray:
    # Gmsh cells are mostly ordered like VTK, with a few exceptions:
    gmsh_ordering = {
        # fmt: off
        "tetra10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        "hexahedron20": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16,
            9, 17, 10, 18, 19, 12, 15, 13, 14,
        ],
        "hexahedron27": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16,
            9, 17, 10, 18, 19, 12, 15, 13, 14,
            24, 22, 20, 21, 23, 25, 26,
        ],
        "wedge15": [
            0, 1, 2, 3, 4, 5, 6, 8, 12, 7, 13, 14, 9, 11, 10,
        ],
        "pyramid13": [0, 1, 2, 3, 4, 5, 8, 9, 6, 10, 7, 11, 12],
        # fmt: on
    }
    idx = np.asarray(idx)
    if cell_type not in gmsh_ordering:
        return idx
    return idx[:, gmsh_ordering[cell_type]]


def _write_physical_names(fh, field_data):
    # Write physical names
    entries = []
    for phys_name in field_data:
        try:
            phys_num, phys_dim = field_data[phys_name]
            phys_num, phys_dim = int(phys_num), int(phys_dim)
            entries.append((phys_dim, phys_num, phys_name))
        except (ValueError, TypeError):
            warn("Field data contains entry that cannot be processed.")
    entries.sort()
    if entries:
        fh.write(b"$PhysicalNames\n")
        fh.write(f"{len(entries)}\n".encode())
        for entry in entries:
            fh.write('{} {} "{}"\n'.format(*entry).encode())
        fh.write(b"$EndPhysicalNames\n")


def _write_data(fh, tag, name, data, binary):
    fh.write(f"${tag}\n".encode())
    # <http://gmsh.info/doc/texinfo/gmsh.html>:
    # > Number of string tags.
    # > gives the number of string tags that follow. By default the first
    # > string-tag is interpreted as the name of the post-processing view and
    # > the second as the name of the interpolation scheme. The interpolation
    # > scheme is provided in the $InterpolationScheme section (see below).
    fh.write(f"{1}\n".encode())
    fh.write(f'"{name}"\n'.encode())
    fh.write(f"{1}\n".encode())
    fh.write(f"{0.0}\n".encode())
    # three integer tags:
    fh.write(f"{3}\n".encode())
    # time step
    fh.write(f"{0}\n".encode())
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1
    if num_components not in [1, 3, 9]:
        raise WriteError("Gmsh only permits 1, 3, or 9 components per data field.")

    # Cut off the last dimension in case it's 1. This avoids problems with
    # writing the data.
    if len(data.shape) > 1 and data.shape[1] == 1:
        data = data[:, 0]

    fh.write(f"{num_components}\n".encode())
    # num data items
    fh.write(f"{data.shape[0]}\n".encode())
    # actually write the data
    if binary:
        if num_components == 1:
            dtype = [("index", c_int), ("data", c_double)]
        else:
            dtype = [("index", c_int), ("data", c_double, num_components)]
        tmp = np.empty(len(data), dtype=dtype)
        tmp["index"] = 1 + np.arange(len(data))
        tmp["data"] = data
        tmp.tofile(fh)
        fh.write(b"\n")
    else:
        fmt = " ".join(["{}"] + ["{!r}"] * num_components) + "\n"
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, x).encode())
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, *x).encode())

    fh.write(f"$End{tag}\n".encode())
