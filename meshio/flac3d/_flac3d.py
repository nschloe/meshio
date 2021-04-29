"""
I/O for FLAC3D format.
"""
import logging
import struct
import time

import numpy as np

from ..__about__ import __version__ as version
from .._common import _pick_first_int_data
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh

meshio_only = {
    "zone": {
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
    },
    "face": {
        "triangle": "triangle",
        "triangle6": "triangle",
        "triangle7": "triangle",
        "quad": "quad",
        "quad8": "quad",
        "quad9": "quad",
    },
}


numnodes_to_meshio_type = {
    "zone": {4: "tetra", 5: "pyramid", 6: "wedge", 8: "hexahedron"},
    "face": {3: "triangle", 4: "quad"},
}


meshio_to_flac3d_type = {
    "triangle": "T3",
    "quad": "Q4",
    "tetra": "T4",
    "pyramid": "P5",
    "wedge": "W6",
    "hexahedron": "B8",
}


flac3d_to_meshio_order = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 4, 2, 3],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 4, 2, 3, 6, 7, 5],
}


meshio_to_flac3d_order = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
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


flag_to_numdim = {
    "zone": 3,
    "face": 2,
}


def read(filename):
    """Read FLAC3D f3grid grid file."""
    # Read a small block of the file to assess its type
    # See <https://code.activestate.com/recipes/173220/>
    with open_file(filename, "rb") as f:
        block = f.read(8)
        binary = b"\x00" in block

    mode = "rb" if binary else "r"
    with open_file(filename, mode) as f:
        out = read_buffer(f, binary)

    return out


def read_buffer(f, binary):
    """Read binary or ASCII file."""
    flags = {
        "Z": "zone",
        "F": "face",
        "ZGROUP": "zone",
        "FGROUP": "face",
    }

    points = []
    point_ids = {}
    cells = []
    field_data = {}

    # Zones and faces do not share the same cell ID pool in FLAC3D
    # i.e. a given cell ID can be assigned to a zone and a face concurrently
    mapper = {"zone": {}, "face": {}}
    slots = {"zone": set(), "face": set()}

    pidx = 0
    cidx = 0
    gidx = 0
    if binary:
        # Not sure what the first bytes represent, the format might be wrong
        # It does not seem to be useful anyway
        _ = struct.unpack("<2I", f.read(8))

        (num_nodes,) = struct.unpack("<I", f.read(4))
        for pidx in range(num_nodes):
            pid, point = _read_point(f, binary)
            points.append(point)
            point_ids[pid] = pidx

        for flag in ["zone", "face"]:
            (num_cells,) = struct.unpack("<I", f.read(4))
            for _ in range(num_cells):
                cid, cell = _read_cell(f, point_ids, binary)
                cells = _update_cells(cells, cell, flag)
                mapper[flag][cid] = [cidx]
                cidx += 1

            (num_groups,) = struct.unpack("<I", f.read(4))
            for _ in range(num_groups):
                name, slot, data = _read_group(f, binary)
                field_data, mapper[flag] = _update_field_data(
                    field_data,
                    mapper[flag],
                    data,
                    name,
                    gidx + 1,
                    flag,
                )
                slots[flag] = _update_slots(slots[flag], slot)
                gidx += 1
    else:
        line = f.readline().rstrip().split()
        while line:
            if line[0] == "G":
                pid, point = _read_point(line, binary)
                points.append(point)
                point_ids[pid] = pidx
                pidx += 1
            elif line[0] in {"Z", "F"}:
                flag = flags[line[0]]
                cid, cell = _read_cell(line, point_ids, binary)
                cells = _update_cells(cells, cell, flag)
                mapper[flag][cid] = [cidx]
                cidx += 1
            elif line[0] in {"ZGROUP", "FGROUP"}:
                flag = flags[line[0]]
                name, slot, data = _read_group(f, binary, line)
                field_data, mapper[flag] = _update_field_data(
                    field_data,
                    mapper[flag],
                    data,
                    name,
                    gidx + 1,
                    flag,
                )
                slots[flag] = _update_slots(slots[flag], slot)
                gidx += 1

            line = f.readline().rstrip().split()

    if field_data:
        num_cells = np.cumsum([len(c[1]) for c in cells])
        cell_data = np.zeros(num_cells[-1], dtype=int)
        for k, v in mapper.items():
            if not slots[k]:
                continue

            for cid, zid in v.values():
                cell_data[cid] = zid
        cell_data = {"flac3d:group": np.split(cell_data, num_cells[:-1])}
    else:
        cell_data = {}

    return Mesh(
        points=np.array(points),
        cells=[(k, np.array(v)[:, flac3d_to_meshio_order[k]]) for k, v in cells],
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
        cell = struct.unpack(f"<{num_verts}I", buf_or_line.read(4 * num_verts))
        is_b7 = num_verts == 7
    else:
        cid = int(buf_or_line[2])
        cell = buf_or_line[3:]
        is_b7 = buf_or_line[1] == "B7"

    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])

    return cid, cell


def _read_group(buf_or_line, binary, line=None):
    """Read cell group."""
    if binary:
        # Group name
        (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
        (name,) = struct.unpack(f"<{num_chars}s", buf_or_line.read(num_chars))
        name = name.decode()

        # Slot name
        (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
        (slot,) = struct.unpack(f"<{num_chars}s", buf_or_line.read(num_chars))
        slot = slot.decode()

        # Zones
        (num_zones,) = struct.unpack("<I", buf_or_line.read(4))
        data = struct.unpack(f"<{num_zones}I", buf_or_line.read(4 * num_zones))
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


def _update_cells(cells, cell, flag):
    """Update cell list."""
    cell_type = numnodes_to_meshio_type[flag][len(cell)]
    if len(cells) > 0 and cell_type == cells[-1][0]:
        cells[-1][1].append(cell)
    else:
        cells.append((cell_type, [cell]))

    return cells


def _update_field_data(field_data, mapper, data, name, gidx, flag):
    """Update field data dict."""
    for cid in data:
        mapper[cid].append(gidx)
    field_data[name] = np.array([gidx, flag_to_numdim[flag]])

    return field_data, mapper


def _update_slots(slots, slot):
    """Update slot set. Only one slot is supported."""
    slots.add(slot)
    if len(slots) > 1:
        raise ReadError("Multiple slots are not supported")

    return slots


def write(filename, mesh, float_fmt=".16e", binary=False):
    """Write FLAC3D f3grid grid file."""
    skip = [c for c in mesh.cells if c.type not in meshio_only["zone"]]
    if skip:
        logging.warning(
            f'FLAC3D format only supports 3D cells. Skipping {", ".join(skip)}.'
        )

    # Pick out material
    material = None
    if mesh.cell_data:
        key, other = _pick_first_int_data(mesh.cell_data)
        if key:
            material = np.concatenate(mesh.cell_data[key])
            if other:
                logging.warning(
                    "FLAC3D can only write one cell data array. "
                    f'Picking {key}, skipping {", ".join(other)}.'
                )

    mode = "wb" if binary else "w"
    with open_file(filename, mode) as f:
        if binary:
            f.write(
                struct.pack("<2I", 1375135718, 3)
            )  # Don't know what these values represent
        else:
            f.write(f"* FLAC3D grid produced by meshio v{version}\n")
            f.write(f"* {time.ctime()}\n")

        _write_points(f, mesh.points, binary, float_fmt)
        for flag in ["zone", "face"]:
            _write_cells(f, mesh.points, mesh.cells, flag, binary)
            _write_groups(f, mesh.cells, material, mesh.field_data, flag, binary)


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


def _write_cells(f, points, cells, flag, binary):
    """Write cells."""
    if flag == "zone":
        count = 0
        cells = _translate_zones(points, cells)
    else:
        count = sum(len(c[1]) for c in cells if c.type in meshio_only["zone"])
        cells = _translate_faces(cells)

    if binary:
        f.write(
            struct.pack(
                "<I", sum(len(c[1]) for c in cells if c[0] in meshio_only[flag])
            )
        )
        for _, cdata in cells:
            num_cells, num_verts = cdata.shape
            tmp = np.column_stack(
                (
                    np.arange(1, num_cells + 1) + count,
                    np.full(num_cells, num_verts),
                    cdata + 1,
                )
            ).astype(int)
            f.write(struct.pack(f"<{(num_verts + 2) * num_cells}I", *tmp.ravel()))
            count += num_cells
    else:
        entity, abbrev = {
            "zone": ("ZONES", "Z"),
            "face": ("FACES", "F"),
        }[flag]

        f.write(f"* {entity}\n")
        for ctype, cdata in cells:
            fmt = f"{abbrev} {{}} {{}} " + " ".join(["{}"] * cdata.shape[1]) + "\n"
            for entry in cdata + 1:
                count += 1
                f.write(fmt.format(meshio_to_flac3d_type[ctype], count, *entry))


def _write_groups(f, cells, cell_data, field_data, flag, binary):
    """Write groups."""
    if cell_data is not None:
        groups, labels = _translate_groups(cells, cell_data, field_data, flag)

        if binary:
            slot = b"Default"

            f.write(struct.pack("<I", len(groups)))
            for k in sorted(groups.keys()):
                num_chars, num_zones = len(labels[k]), len(groups[k])
                fmt = f"<H{num_chars}sH7sI{num_zones}I"
                tmp = [
                    num_chars,
                    labels[k].encode(),
                    7,
                    slot,
                    num_zones,
                    *groups[k],
                ]
                f.write(struct.pack(fmt, *tmp))
        else:
            flag_to_text = {
                "zone": "ZGROUP",
                "face": "FGROUP",
            }

            f.write(f"* {flag.upper()} GROUPS\n")
            for k in sorted(groups.keys()):
                f.write(f'{flag_to_text[flag]} "{labels[k]}"\n')
                _write_table(f, groups[k])
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
        if key not in meshio_only["zone"].keys():
            continue

        # Compute scalar triple products
        key = meshio_only["zone"][key]
        tmp = points[idx[:, meshio_to_flac3d_order[key][:4]].T]
        det = slicing_summing(tmp[1] - tmp[0], tmp[2] - tmp[0], tmp[3] - tmp[0])

        # Reorder corner points
        data = np.where(
            (det > 0)[:, None],
            idx[:, meshio_to_flac3d_order[key]],
            idx[:, meshio_to_flac3d_order_2[key]],
        )
        zones.append((key, data))

    return zones


def _translate_faces(cells):
    """Reorder meshio cells to FLAC3D faces."""
    faces = []
    for key, idx in cells:
        if key not in meshio_only["face"].keys():
            continue

        key = meshio_only["face"][key]
        data = idx[:, meshio_to_flac3d_order[key]]
        faces.append((key, data))

    return faces


def _translate_groups(cells, cell_data, field_data, flag):
    """Convert meshio cell_data to FLAC3D groups."""
    num_dims = np.concatenate(
        [np.full(len(c[1]), 2 if c[0] in meshio_only["face"] else 3) for c in cells]
    )
    groups = {
        k: np.nonzero(np.logical_and(cell_data == k, num_dims == flag_to_numdim[flag]))[
            0
        ]
        + 1
        for k in np.unique(cell_data)
    }
    groups = {k: v for k, v in groups.items() if v.size}

    labels = {k: str(k) for k in groups.keys()}
    labels[0] = "None"
    if field_data:
        labels.update(
            {v[0]: k for k, v in field_data.items() if v[1] == flag_to_numdim[flag]}
        )

    return groups, labels


def _write_table(f, data, ncol=20):
    """Write group data table."""
    nrow = len(data) // ncol
    lines = np.split(data, np.full(nrow, ncol).cumsum())
    for line in lines:
        if len(line):
            f.write(" {}\n".format(" ".join([str(l) for l in line])))


register("flac3d", [".f3grid"], read, {"flac3d": write})
