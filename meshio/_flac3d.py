"""
I/O for FLAC3D format.
"""
import logging
import time

import numpy as np

from .__about__ import __version__ as version
from ._mesh import Mesh

meshio_only = {
    "tetra",
    "pyramid",
    "wedge",
    "hexahedron",
}


meshio_data = {
    "flac3d:zone",
    "gmsh:physical",
    "medit:ref",
}


flac3d_to_meshio_type = {
    "T4": "tetra",
    "P5": "pyramid",
    "W6": "wedge",
    "B7": "hexahedron",
    "B8": "hexahedron",
}
meshio_to_flac3d_type = {
    "tetra": "T4",
    "pyramid": "P5",
    "wedge": "W6",
    "hexahedron": "B8",
}


flac3d_to_meshio_order = {
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 4, 2, 3],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 4, 2, 3, 6, 7, 5],
}


meshio_to_flac3d_order = {
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 3, 4, 2],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 3, 4, 2, 7, 5, 6],
}


meshio_to_flac3d_order_2 = {
    "tetra": [0, 2, 1, 3],
    "pyramid": [0, 3, 1, 4, 2],
    "wedge": [0, 2, 3, 1, 5, 4],
    "hexahedron": [0, 3, 1, 4, 2, 5, 7, 6],
}


def read(filename):
    """
    Read FLAC3D f3grid grid file (only ASCII).
    """
    with open(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    """
    Read ASCII file line by line. Use combination of readline, tell and
    seek since we need to rewind to the previous line when we read last
    data line of a ZGROUP section.
    """
    points = []
    zones = {}
    zgroups = {}
    count = {k: 0 for k in meshio_to_flac3d_type.keys()}

    line = f.readline()
    while line:
        if line.startswith("ZGROUP"):
            _read_zgroup(f, line, zgroups)
        elif line.startswith("G"):
            _read_point(line, points)
        elif line.startswith("Z"):
            _read_cell(line, zones, count)
        line = f.readline()

    cells, cell_data, field_data = _translate_cells(zones, zgroups, count)
    return Mesh(
        points=np.array(points), cells=cells, cell_data=cell_data, field_data=field_data
    )


def _read_point(line, points):
    """
    Read point coordinates.
    """
    line = line.rstrip().split()
    points.append([float(l) for l in line[2:]])


def _read_cell(line, zones, count):
    """
    Read cell corners.
    """
    line = line.rstrip().split()
    meshio_type = flac3d_to_meshio_type[line[1]]
    cell = [int(l) - 1 for l in line[3:]]
    if line[1] == "B7":
        cell.append(cell[-1])
    zones[int(line[2])] = {
        "meshio_id": count[meshio_type],
        "meshio_type": meshio_type,
        "cell": [cell[i] for i in flac3d_to_meshio_order[meshio_type]],
    }
    count[meshio_type] += 1


def _read_zgroup(f, line, zgroups):
    """
    Read cell group.
    """
    group_name = line.rstrip().split()[1].replace('"', "")
    zgroups[group_name] = []

    i = f.tell()
    line = f.readline()
    while True:
        line = line.rstrip().split()
        if line and (line[0] not in ["*", "ZGROUP"]):
            zgroups[group_name].extend([int(l) for l in line])
        else:
            f.seek(i)
            break
        i = f.tell()
        line = f.readline()


def _translate_cells(zones, zgroups, count):
    """
    Convert FLAC3D zones and groups as meshio cells and cell_data.
    """
    meshio_types = [k for k in meshio_to_flac3d_type.keys() if count[k]]
    cells = {k: [] for k in meshio_types}
    cell_data = {k: np.zeros(count[k]) for k in meshio_types}
    field_data = {}

    for v in zones.values():
        cells[v["meshio_type"]].append(v["cell"])

    for zid, (k, v) in enumerate(zgroups.items()):
        field_data[k] = np.array([zid + 1, 3])
        for i in v:
            mid = zones[i]["meshio_id"]
            mt = zones[i]["meshio_type"]
            cell_data[mt][mid] = zid + 1

    cells = {k: np.array(v) for k, v in cells.items()}
    cell_data = {k: {"flac3d:zone": v} for k, v in cell_data.items()}
    return cells, cell_data, field_data


def write(filename, mesh):
    """
    Write FLAC3D f3grid grid file (only ASCII).
    """
    assert any(
        cell_type in meshio_only for cell_type in mesh.cells.keys()
    ), "FLAC3D format only supports 'tetra', 'pyramid', 'wedge' and 'hexahedron'."

    if mesh.points.shape[1] == 2:
        logging.warning(
            "FLAC3D requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = np.hstack([mesh.points, np.zeros((len(mesh.points), 1))])

    with open(filename, "w") as f:
        f.write("* FLAC3D grid produced by meshio v{}\n".format(version))
        f.write("* {}\n".format(time.ctime()))
        f.write("* GRIDPOINTS\n")
        _write_points(f, mesh.points)
        f.write("* ZONES\n")
        _write_cells(f, mesh.points, mesh.cells)
        f.write("* ZONE GROUPS\n")
        _write_cell_data(f, mesh.cells, mesh.cell_data, mesh.field_data)


def _write_points(f, points):
    """
    Write points coordinates.
    """
    for i, point in enumerate(points):
        f.write("G\t{:8}\t{:.14e}\t{:.14e}\t{:.14e}\n".format(i + 1, *point))


def _write_cells(f, points, cells):
    """
    Write zones.
    """
    zones, meshio_types = _translate_zones(points, cells)
    for i, (zone, meshio_type) in enumerate(zip(zones, meshio_types)):
        zone_str = " ".join([str(z + 1) for z in zone])
        f.write(
            "Z {} {} {}\n".format(meshio_to_flac3d_type[meshio_type], i + 1, zone_str)
        )


def _translate_zones(points, cells):
    """
    Reorder meshio cells to FLAC3D zones. Four first points must form a
    right-handed coordinate system (outward normal vectors). Reorder corner
    points according to sign of scalar triple products.
    """
    # Calculate scalar triple products
    meshio_types = [k for k in cells.keys() if k in meshio_only]
    corners = [v for k, v in cells.items() if k in meshio_only]
    tmp = [
        corner[:, meshio_to_flac3d_order[k][:4]]
        for k, corner in zip(meshio_types, corners)
    ]
    p0, p1, p2, p3 = points[np.concatenate(tmp).T]
    dets = (np.cross(p1 - p0, p2 - p0) * (p3 - p0)).sum(axis=1)

    # Reorder corner points
    meshio_types = [
        kk for k, v in cells.items() for kk in [k] * len(v) if k in meshio_only
    ]
    corners = [c for corner in corners for c in corner]
    zones = [
        corner[meshio_to_flac3d_order[k]]
        if det > 0.0
        else corner[meshio_to_flac3d_order_2[k]]
        for corner, k, det in zip(corners, meshio_types, dets)
    ]
    return zones, meshio_types


def _write_cell_data(f, cells, cell_data, field_data):
    """
    Write zone groups.
    """
    zgroups, labels = _translate_zgroups(cells, cell_data, field_data)
    for k, v in zgroups.items():
        f.write("ZGROUP '{}'\n".format(labels[k]))
        _write_zgroup(f, v)


def _translate_zgroups(cells, cell_data, field_data):
    """
    Convert meshio cell_data to FLAC3D zone groups.
    """
    n_cells = sum([len(v) for k, v in cells.items() if k in meshio_only])
    zone_data = np.zeros(n_cells, dtype=np.int32)
    idx = 0
    for k, v in cells.items():
        if k in meshio_only:
            if k in cell_data.keys():
                for kk, vv in cell_data[k].items():
                    if kk in meshio_data:
                        zone_data[idx : idx + len(vv)] = vv
                idx += len(v)

    zgroups = {}
    for zid, i in enumerate(zone_data):
        if i in zgroups.keys():
            zgroups[i].append(zid + 1)
        else:
            zgroups[i] = [zid + 1]

    labels = {k: str(k) for k in zgroups.keys()}
    labels[0] = "None"
    if field_data:
        labels.update({v[0]: k for k, v in field_data.items() if v[1] == 3})
    return zgroups, labels


def _write_zgroup(f, data, ncol=20):
    """
    Write zone group data.
    """
    nrow = len(data) // ncol
    lines = np.reshape(data[: nrow * ncol], (nrow, ncol)).tolist()
    if data[nrow * ncol :]:
        lines.append(data[nrow * ncol :])
    for line in lines:
        f.write(" {}\n".format(" ".join([str(l) for l in line])))
