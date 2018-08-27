# -*- coding: utf-8 -*-
#
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
