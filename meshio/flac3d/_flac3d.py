"""
I/O for FLAC3D format.
"""
import logging
import struct
import time

import numpy

from ..__about__ import __version__ as version
from .._common import _pick_first_int_data
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
    """Read FLAC3D f3grid grid file."""
    # Read a small block of the file to assess its type
    # See <http://code.activestate.com/recipes/173220/>
    with open_file(filename, "rb") as f:
        block = f.read(8)
        binary = b"\x00" in block

    mode = "rb" if binary else "r"
    with open_file(filename, mode) as f:
        out = read_buffer(f, binary)

    return out


def read_buffer(f, binary):
    """Read binary or ASCII file."""
    points = []
    point_ids = {}
    cells = []
    mapper = {}
    field_data = {}
    slots = set()

    if binary:
        # Not sure what the first bytes represent, the format might be wrong
        # It does not seem to be useful anyway
        _ = struct.unpack("<2I", f.read(8))

        (num_nodes,) = struct.unpack("<I", f.read(4))
        for pidx in range(num_nodes):
            pid, point = _read_point(f, binary)
            points.append(point)
            point_ids[pid] = pidx

        (num_cells,) = struct.unpack("<I", f.read(4))
        for cidx in range(num_cells):
            cid, cell = _read_cell(f, point_ids, binary)
            cells = _update_cells(cells, cell)
            mapper[cid] = [cidx]

        (num_groups,) = struct.unpack("<I", f.read(4))
        for zidx in range(num_groups):
            name, slot, data = _read_zgroup(f, binary)
            field_data, mapper = _update_field_data(
                field_data, mapper, data, name, zidx + 1
            )
            slots = _update_slots(slots, slot)
    else:
        pidx = 0
        zidx = 0
        count = 0

        line = f.readline().rstrip().split()
        while line:
            if line[0] == "G":
                pid, point = _read_point(line, binary)
                points.append(point)
                point_ids[pid] = pidx
                pidx += 1
            elif line[0] == "Z":
                cid, cell = _read_cell(line, point_ids, binary)
                cells = _update_cells(cells, cell)
                mapper[cid] = [count]
                count += 1
            elif line[0] == "ZGROUP":
                name, slot, data = _read_zgroup(f, binary, line)
                field_data, mapper = _update_field_data(
                    field_data, mapper, data, name, zidx + 1
                )
                slots = _update_slots(slots, slot)
                zidx += 1

            line = f.readline().rstrip().split()

    if field_data:
        num_cells = numpy.cumsum([len(c[1]) for c in cells])
        cell_data = numpy.empty(num_cells[-1], dtype=int)
        for cid, zid in mapper.values():
            cell_data[cid] = zid
        cell_data = {"flac3d:zone": numpy.split(cell_data, num_cells[:-1])}
    else:
        cell_data = {}

    return Mesh(
        points=numpy.array(points),
        cells=[(k, numpy.array(v)[:, flac3d_to_meshio_order[k]]) for k, v in cells],
        cell_data=cell_data,
        field_data=field_data,
    )


def _read_point(buf_or_line, binary):
    """Read point coordinates."""
    if binary:
        pid, x, y, z = struct.unpack("<I3d", buf_or_line.read(28))
        point = [x, y, z]
    else:
        pid = int(buf_or_line[1])
        point = [float(l) for l in buf_or_line[2:]]

    return pid, point


def _read_cell(buf_or_line, point_ids, binary):
    """Read cell connectivity."""
    if binary:
        cid, num_verts = struct.unpack("<2I", buf_or_line.read(8))
        cell = struct.unpack("<{}I".format(num_verts), buf_or_line.read(4 * num_verts))
        is_b7 = num_verts == 7
    else:
        cid = int(buf_or_line[2])
        cell = buf_or_line[3:]
        is_b7 = buf_or_line[1] == "B7"

    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])

    return cid, cell


def _read_zgroup(buf_or_line, binary, line=None):
    """Read cell group."""
    if binary:
        # Group name
        (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
        (name,) = struct.unpack("<{}s".format(num_chars), buf_or_line.read(num_chars))
        name = name.decode("utf-8")

        # Slot name
        (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
        (slot,) = struct.unpack("<{}s".format(num_chars), buf_or_line.read(num_chars))
        slot = slot.decode("utf-8")

        # Zones
        (num_zones,) = struct.unpack("<I", buf_or_line.read(4))
        data = struct.unpack("<{}I".format(num_zones), buf_or_line.read(4 * num_zones))
    else:
        name = line[1].replace('"', "")
        data = []
        slot = "" if "SLOT" not in line else line[-1]

        i = buf_or_line.tell()
        line = buf_or_line.readline()
        while True:
            line = line.rstrip().split()
            if line and (line[0] not in {"*", "ZGROUP"}):
                data += [int(l) for l in line]
            else:
                buf_or_line.seek(i)
                break
            i = buf_or_line.tell()
            line = buf_or_line.readline()

    return name, slot, data


def _update_cells(cells, cell):
    """Update cell list."""
    cell_type = numnodes_to_meshio_type[len(cell)]
    if len(cells) > 0 and cell_type == cells[-1][0]:
        cells[-1][1].append(cell)
    else:
        cells.append((cell_type, [cell]))

    return cells


def _update_field_data(field_data, mapper, data, name, zidx):
    """Update field data dict."""
    for cid in data:
        mapper[cid].append(zidx)
    field_data[name] = numpy.array([zidx, 3])

    return field_data, mapper


def _update_slots(slots, slot):
    """Update slot set. Only one slot is supported."""
    slots.add(slot)
    if len(slots) > 1:
        raise ReadError("Multiple slots are not supported")

    return slots


def write(filename, mesh, float_fmt=".16e", binary=False):
    """Write FLAC3D f3grid grid file."""
    if not any(c.type in meshio_only.keys() for c in mesh.cells):
        raise WriteError("FLAC3D format only supports 3D cells")

    mode = "wb" if binary else "w"
    with open_file(filename, mode) as f:
        if binary:
            f.write(
                struct.pack("<2I", 1375135718, 3)
            )  # Don't know what these values represent
        else:
            f.write("* FLAC3D grid produced by meshio v{}\n".format(version))
            f.write("* {}\n".format(time.ctime()))

        _write_points(f, mesh.points, binary, float_fmt)
        _write_cells(f, mesh.points, mesh.cells, binary)
        _write_zgroups(f, mesh.cell_data, mesh.field_data, binary)

        if binary:
            f.write(struct.pack("<2I", 0, 0))  # No face and face group


def _write_points(f, points, binary, float_fmt=None):
    """Write points coordinates."""
    if binary:
        f.write(struct.pack("<I", len(points)))
        for i, point in enumerate(points):
            f.write(struct.pack("<I3d", i + 1, *point))
    else:
        f.write("* GRIDPOINTS\n")
        for i, point in enumerate(points):
            fmt = "G\t{:8}\t" + "\t".join(3 * ["{:" + float_fmt + "}"]) + "\n"
            f.write(fmt.format(i + 1, *point))


def _write_cells(f, points, cells, binary):
    """Write zones."""
    zones = _translate_zones(points, cells)

    count = 0
    if binary:
        f.write(struct.pack("<I", sum(len(c.data) for c in cells)))
        for _, zone in zones:
            num_cells, num_verts = zone.shape
            tmp = numpy.column_stack(
                (
                    numpy.arange(1, num_cells + 1) + count,
                    numpy.full(num_cells, num_verts),
                    zone + 1,
                )
            )
            f.write(
                struct.pack("<{}I".format((num_verts + 2) * num_cells), *tmp.ravel())
            )
            count += num_cells
    else:
        f.write("* ZONES\n")
        for meshio_type, zone in zones:
            fmt = "Z {} {} " + " ".join(["{}"] * zone.shape[1]) + "\n"
            for entry in zone + 1:
                count += 1
                f.write(fmt.format(meshio_to_flac3d_type[meshio_type], count, *entry))


def _write_zgroups(f, cell_data, field_data, binary):
    """Write zone groups."""
    zgroups = None
    if cell_data:
        # Pick out material
        key, other = _pick_first_int_data(cell_data)
        if key:
            material = numpy.concatenate(cell_data[key])
            if other:
                logging.warning(
                    "FLAC3D can only write one cell data array. "
                    "Picking {}, skipping {}.".format(key, ", ".join(other))
                )
            zgroups, labels = _translate_zgroups(material, field_data)

    if zgroups:
        if binary:
            slot = "Default".encode("utf-8")

            f.write(struct.pack("<I", len(zgroups)))
            for k in sorted(zgroups.keys()):
                num_chars, num_zones = len(labels[k]), len(zgroups[k])
                fmt = "<H{}sH7sI{}I".format(num_chars, num_zones)
                tmp = [
                    num_chars,
                    labels[k].encode("utf-8"),
                    7,
                    slot,
                    num_zones,
                    *zgroups[k],
                ]
                f.write(struct.pack(fmt, *tmp))
        else:
            f.write("* ZONE GROUPS\n")
            for k in sorted(zgroups.keys()):
                f.write('ZGROUP "{}"\n'.format(labels[k]))
                _write_table(f, zgroups[k])
    else:
        if binary:
            f.write(struct.pack("<I", 0))


def _translate_zones(points, cells):
    """Reorder meshio cells to FLAC3D zones.

    Four first points must form a right-handed coordinate system (outward normal vectors).
    Reorder corner points according to sign of scalar triple products.
    """
    # See <https://stackoverflow.com/a/42386330/353337>
    def slicing_summing(a, b, c):
        c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
        c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2

    zones = []
    for key, idx in cells:
        if key not in meshio_only.keys():
            continue

        # Compute scalar triple products
        key = meshio_only[key]
        tmp = points[idx[:, meshio_to_flac3d_order[key][:4]].T]
        det = slicing_summing(tmp[1] - tmp[0], tmp[2] - tmp[0], tmp[3] - tmp[0])
        # Reorder corner points
        data = numpy.where(
            (det > 0)[:, None],
            idx[:, meshio_to_flac3d_order[key]],
            idx[:, meshio_to_flac3d_order_2[key]],
        )
        zones.append((key, data))

    return zones


def _translate_zgroups(zone_data, field_data):
    """Convert meshio cell_data to FLAC3D zone groups."""
    zgroups = {k: numpy.nonzero(zone_data == k)[0] + 1 for k in numpy.unique(zone_data)}

    labels = {k: str(k) for k in zgroups.keys()}
    labels[0] = "None"
    if field_data:
        labels.update({v[0]: k for k, v in field_data.items() if v[1] == 3})
    return zgroups, labels


def _write_table(f, data, ncol=20):
    """Write zone group data table."""
    nrow = len(data) // ncol
    lines = numpy.split(data, numpy.full(nrow, ncol).cumsum())
    for line in lines:
        if len(line):
            f.write(" {}\n".format(" ".join([str(l) for l in line])))


register("flac3d", [".f3grid"], read, {"flac3d": write})
