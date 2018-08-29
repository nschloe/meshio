# -*- coding: utf-8 -*-
#
from ctypes import c_int, c_double
import logging
import shlex

import numpy


def _read_physical_names(f, field_data):
    line = f.readline().decode("utf-8")
    num_phys_names = int(line)
    for _ in range(num_phys_names):
        line = shlex.split(f.readline().decode("utf-8"))
        key = line[2]
        value = numpy.array(line[1::-1], dtype=int)
        field_data[key] = value
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPhysicalNames"
    return


def _read_periodic(f):
    periodic = []
    num_periodic = int(f.readline().decode("utf-8"))
    for _ in range(num_periodic):
        line = f.readline().decode("utf-8")
        edim, stag, mtag = [int(s) for s in line.split()]
        line = f.readline().decode("utf-8").strip()
        if line.startswith("Affine"):
            transform = line
            num_nodes = int(f.readline().decode("utf-8"))
        else:
            transform = None
            num_nodes = int(line)
        slave_master = []
        for _ in range(num_nodes):
            line = f.readline().decode("utf-8")
            snode, mnode = [int(s) for s in line.split()]
            slave_master.append([snode, mnode])
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master -= 1  # Subtract one, Python is 0-based
        periodic.append([edim, (stag, mtag), transform, slave_master])
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPeriodic"
    return periodic


def _read_data(f, tag, data_dict, int_size, data_size, is_ascii):
    # Read string tags
    num_string_tags = int(f.readline().decode("utf-8"))
    string_tags = [
        f.readline().decode("utf-8").strip().replace('"', "")
        for _ in range(num_string_tags)
    ]
    # The real tags typically only contain one value, the time.
    # Discard it.
    num_real_tags = int(f.readline().decode("utf-8"))
    for _ in range(num_real_tags):
        f.readline()
    num_integer_tags = int(f.readline().decode("utf-8"))
    integer_tags = [int(f.readline().decode("utf-8")) for _ in range(num_integer_tags)]
    num_components = integer_tags[1]
    num_items = integer_tags[2]
    if is_ascii:
        data = numpy.fromfile(
            f, count=num_items * (1 + num_components), sep=" "
        ).reshape((num_items, 1 + num_components))
        # The first entry is the node number
        data = data[:, 1:]
    else:
        # binary
        dtype = [("index", c_int), ("values", c_double, (num_components,))]
        data = numpy.fromfile(f, count=num_items, dtype=dtype)
        assert (data["index"] == range(1, num_items + 1)).all()
        data = numpy.ascontiguousarray(data["values"])
        line = f.readline().decode("utf-8")
        assert line == "\n"

    line = f.readline().decode("utf-8")
    assert line.strip() == "$End{}".format(tag)

    # The gmsh format cannot distingiush between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data
    return


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


def _write_physical_names(fh, field_data):
    # Write physical names
    entries = []
    for phys_name in field_data:
        try:
            phys_num, phys_dim = field_data[phys_name]
            phys_num, phys_dim = int(phys_num), int(phys_dim)
            entries.append((phys_dim, phys_num, phys_name))
        except (ValueError, TypeError):
            logging.warning("Field data contains entry that cannot be processed.")
    entries.sort()
    if entries:
        fh.write("$PhysicalNames\n".encode("utf-8"))
        fh.write("{}\n".format(len(entries)).encode("utf-8"))
        for entry in entries:
            fh.write('{} {} "{}"\n'.format(*entry).encode("utf-8"))
        fh.write("$EndPhysicalNames\n".encode("utf-8"))
    return


def _write_periodic(fh, periodic):
    fh.write("$Periodic\n".encode("utf-8"))
    fh.write("{}\n".format(len(periodic)).encode("utf-8"))
    for dim, (stag, mtag), transform, slave_master in periodic:
        fh.write("{} {} {}\n".format(dim, stag, mtag).encode("utf-8"))
        if transform is not None:
            fh.write("{}\n".format(transform).encode("utf-8"))
        slave_master = numpy.array(slave_master, dtype=int).reshape(-1, 2)
        slave_master = slave_master + 1  # Add one, Gmsh is 0-based
        fh.write("{}\n".format(len(slave_master)).encode("utf-8"))
        for snode, mnode in slave_master:
            fh.write("{} {}\n".format(snode, mnode).encode("utf-8"))
    fh.write("$EndPeriodic\n".encode("utf-8"))


def _write_data(fh, tag, name, data, write_binary):
    fh.write("${}\n".format(tag).encode("utf-8"))
    # <http://gmsh.info/doc/texinfo/gmsh.html>:
    # > Number of string tags.
    # > gives the number of string tags that follow. By default the first
    # > string-tag is interpreted as the name of the post-processing view and
    # > the second as the name of the interpolation scheme. The interpolation
    # > scheme is provided in the $InterpolationScheme section (see below).
    fh.write("{}\n".format(1).encode("utf-8"))
    fh.write('"{}"\n'.format(name).encode("utf-8"))
    fh.write("{}\n".format(1).encode("utf-8"))
    fh.write("{}\n".format(0.0).encode("utf-8"))
    # three integer tags:
    fh.write("{}\n".format(3).encode("utf-8"))
    # time step
    fh.write("{}\n".format(0).encode("utf-8"))
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1
    assert num_components in [
        1,
        3,
        9,
    ], "Gmsh only permits 1, 3, or 9 components per data field."

    # Cut off the last dimension in case it's 1. This avoids problems with
    # writing the data.
    if len(data.shape) > 1 and data.shape[1] == 1:
        data = data[:, 0]

    fh.write("{}\n".format(num_components).encode("utf-8"))
    # num data items
    fh.write("{}\n".format(data.shape[0]).encode("utf-8"))
    # actually write the data
    if write_binary:
        dtype = [("index", c_int), ("data", c_double, num_components)]
        tmp = numpy.empty(len(data), dtype=dtype)
        tmp["index"] = 1 + numpy.arange(len(data))
        tmp["data"] = data
        fh.write(tmp.tostring())
        fh.write("\n".encode("utf-8"))
    else:
        fmt = " ".join(["{}"] + ["{!r}"] * num_components) + "\n"
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, x).encode("utf-8"))
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k + 1, *x).encode("utf-8"))

    fh.write("$End{}\n".format(tag).encode("utf-8"))
    return
