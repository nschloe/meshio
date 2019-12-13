"""
I/O for FLAC3D format.
"""
import time

import numpy

from ..__about__ import __version__ as version
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh

meshio_only = {
    "tetra": "tetra",
    "tetra10": "tetra",
    "pyramid": "pyramid",
    "pyramid13": "pyramid",
    "wedge": "wedge",
    "wedge12": "wedge",
    "wedge15": "wedge",
    "wedge18": "wedge",
    "hexahedron": "hexahedron",
    "hexahedron20": "hexahedron",
    "hexahedron24": "hexahedron",
    "hexahedron27": "hexahedron",
}


meshio_data = {"flac3d:zone", "gmsh:physical", "medit:ref"}


numnodes_to_meshio_type = {
    4: "tetra",
    5: "pyramid",
    6: "wedge",
    8: "hexahedron",
}
meshio_type_to_numnodes = {v: k for k, v in numnodes_to_meshio_type.items()}


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
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    """
    Read ASCII file line by line. Use combination of readline, tell and
    seek since we need to rewind to the previous line when we read last
    data line of a ZGROUP section.
    """
    points = []
    point_ids = {}
    cells = {}
    mapper = {}
    field_data = {}
    slots = set()

    pidx = 0
    zidx = 0
    count = {k: 0 for k in flac3d_to_meshio_order.keys()}
    line = f.readline()
    while line:
        line = line.rstrip().split()
        if line[0] == "G":
            pid, point = _read_point(line)
            points.append(point)
            point_ids[pid] = pidx
            pidx += 1
        elif line[0] == "Z":
            cid, cell = _read_cell(line, point_ids)
            cell_type = numnodes_to_meshio_type[len(cell)]
            if cell_type in cells:
                cells[cell_type].append(cell)
            else:
                cells[cell_type] = [cell]
            mapper[cid] = [count[cell_type], len(cell)]
            count[cell_type] += 1
        elif line[0] == "ZGROUP":
            name, data, slot = _read_zgroup(f, line)
            zidx += 1
            for cid in data:
                mapper[cid].append(zidx)
            field_data[name] = numpy.array([zidx, 3])
            slots.add(slot)
            if len(slots) > 1:
                raise ReadError("Multiple slots are not supported")
        line = f.readline()

    if zidx:
        cell_data = {k: {"flac3d:zone": numpy.zeros(v)} for k, v in count.items() if v}
        for i, numnodes, zid in mapper.values():
            cell_data[numnodes_to_meshio_type[numnodes]]["flac3d:zone"][i] = zid
    else:
        cell_data = {}

    return Mesh(
        points=numpy.array(points),
        cells={
            k: numpy.array(v)[:, flac3d_to_meshio_order[k]] for k, v in cells.items()
        },
        cell_data=cell_data,
        field_data=field_data,
    )


def _read_point(line):
    """
    Read point coordinates.
    """
    return int(line[1]), [float(l) for l in line[2:]]


def _read_cell(line, point_ids):
    """
    Read cell corners.
    """
    cell = [point_ids[int(l)] for l in line[3:]]
    if line[1] == "B7":
        cell.append(cell[-1])
    return int(line[2]), cell


def _read_zgroup(f, line):
    """
    Read cell group.
    """
    name = line[1].replace('"', "")
    data = []
    slot = "" if "SLOT" not in line else line[-1]

    i = f.tell()
    line = f.readline()
    while True:
        line = line.rstrip().split()
        if line and (line[0] not in {"*", "ZGROUP"}):
            data += [int(l) for l in line]
        else:
            f.seek(i)
            break
        i = f.tell()
        line = f.readline()
    return name, data, slot


def write(filename, mesh):
    """
    Write FLAC3D f3grid grid file (only ASCII).
    """
    if not any(cell_type in meshio_only.keys() for cell_type in mesh.cells.keys()):
        raise WriteError("FLAC3D format only supports 3D cells")

    with open_file(filename, "w") as f:
        f.write("* FLAC3D grid produced by meshio v{}\n".format(version))
        f.write("* {}\n".format(time.ctime()))
        f.write("* GRIDPOINTS\n")
        _write_points(f, mesh.points)
        f.write("* ZONES\n")
        _write_cells(f, mesh.points, mesh.cells)

        if mesh.cell_data:
            if set(kk for v in mesh.cell_data.values() for kk in v.keys()).intersection(
                meshio_data
            ):
                f.write("* ZONE GROUPS\n")
                _write_cell_data(f, mesh.cells, mesh.cell_data, mesh.field_data)


def _write_points(f, points):
    """
    Write points coordinates.
    """
    for i, point in enumerate(points):
        f.write("G\t{:8}\t{:.14e}\t{:.14e}\t{:.14e}\n".format(i + 1, *point))


def _write_cells(f, points, cells):
    """Write zones.
    """
    zones = _translate_zones(points, cells)
    i = 1
    for meshio_type, zone in zones.items():
        meshio_type = meshio_only[meshio_type]
        fmt = "Z {} {} " + " ".join((["{}"] * zone.shape[1])) + "\n"
        for entry in zone + 1:
            f.write(fmt.format(meshio_to_flac3d_type[meshio_type], i, *entry))
            i += 1


def _translate_zones(points, cells):
    """Reorder meshio cells to FLAC3D zones. Four first points must form a right-handed
    coordinate system (outward normal vectors). Reorder corner points according to sign
    of scalar triple products.
    """
    # See <https://stackoverflow.com/a/42386330/353337>
    def slicing_summing(a, b, c):
        c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
        c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2

    zones = {}
    for key, idx in cells.items():
        if key not in meshio_only.keys():
            continue

        # Compute scalar triple products
        key = meshio_only[key]
        tmp = points[idx[:, meshio_to_flac3d_order[key][:4]].T]
        det = slicing_summing(tmp[1] - tmp[0], tmp[2] - tmp[0], tmp[3] - tmp[0])
        # Reorder corner points
        zones[key] = numpy.empty((len(idx), meshio_type_to_numnodes[key]), dtype=int)
        zones[key][det > 0] = idx[:, meshio_to_flac3d_order[key]][det > 0]
        zones[key][det <= 0] = idx[:, meshio_to_flac3d_order_2[key]][det <= 0]

    return zones


def _write_cell_data(f, cells, cell_data, field_data):
    """
    Write zone groups.
    """
    zgroups, labels = _translate_zgroups(cells, cell_data, field_data)
    for k in sorted(zgroups.keys()):
        f.write('ZGROUP "{}"\n'.format(labels[k]))
        _write_zgroup(f, zgroups[k])


def _translate_zgroups(cells, cell_data, field_data):
    """
    Convert meshio cell_data to FLAC3D zone groups.
    """
    n_cells = sum([len(v) for k, v in cells.items() if k in meshio_only.keys()])
    zone_data = numpy.zeros(n_cells, dtype=numpy.int32)
    idx = 0
    for k, v in cells.items():
        if k in meshio_only.keys():
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
    lines = numpy.reshape(data[: nrow * ncol], (nrow, ncol)).tolist()
    if data[nrow * ncol :]:
        lines.append(data[nrow * ncol :])
    for line in lines:
        f.write(" {}\n".format(" ".join([str(l) for l in line])))


register("flac3d", [".f3grid"], read, {"flac3d": write})
