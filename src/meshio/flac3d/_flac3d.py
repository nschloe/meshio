"""
I/O for FLAC3D format.
"""
from __future__ import annotations

import re
import struct
import time

import numpy as np

from ..__about__ import __version__ as version
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
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


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


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
    points = []
    point_ids = {}
    f_cells = []
    z_cells = []
    f_cell_sets = {}
    z_cell_sets = {}
    f_cell_ids = []
    z_cell_ids = []

    pidx = 0
    if binary:
        # Not sure what the first bytes represent, the format might be wrong
        # It does not seem to be useful anyway
        _ = struct.unpack("<2I", f.read(8))

        (num_nodes,) = struct.unpack("<I", f.read(4))
        for pidx in range(num_nodes):
            pid, point = _read_point_binary(f)
            points.append(point)
            point_ids[pid] = pidx

        for flag in ["zone", "face"]:
            if flag == "zone":
                cell_ids = z_cell_ids
                cells = z_cells
                cell_sets = z_cell_sets
            else:
                cell_ids = f_cell_ids
                cells = f_cells
                cell_sets = f_cell_sets

            (num_cells,) = struct.unpack("<I", f.read(4))
            for _ in range(num_cells):
                cell_id, cell = _read_cell_binary(f, point_ids)
                cell_ids.append(cell_id)
                _update_cells(cells, cell, flag)
                # mapper[flag][cid] = [cidx]
                # cidx += 1

            (num_groups,) = struct.unpack("<I", f.read(4))
            for _ in range(num_groups):
                name, slot, data = _read_cell_group_binary(f)
                cell_sets[f"{flag}:{name}:{slot}"] = np.array(data)
    else:
        while True:
            line = f.readline()

            if not line:
                break

            if line.strip() == "":
                continue

            split = line.rstrip().split()

            if split[0] == "G":
                pid, point = _read_point_ascii(split)
                points.append(point)
                point_ids[pid] = pidx
                pidx += 1

            elif split[0] == "Z":
                cell_id, cell = _read_cell_ascii(split, point_ids)
                z_cell_ids.append(cell_id)
                _update_cells(z_cells, cell, "zone")

            elif split[0] == "F":
                cell_id, cell = _read_cell_ascii(split, point_ids)
                f_cell_ids.append(cell_id)
                _update_cells(f_cells, cell, "face")

            elif split[0] == "ZGROUP":
                # ZGROUP "Region 2" SLOT 1
                name, slot, data = _read_cell_group_ascii(f, line)
                # Watch out! data refers to the global cell_ids, so we need to
                # adapt this later.
                z_cell_sets[f"zone:{name}:{slot}"] = np.asarray(data)

            elif split[0] == "FGROUP":
                name, slot, data = _read_cell_group_ascii(f, line)
                # Watch out! data refers to the global cell_ids, so we need to
                # adapt this later.
                f_cell_sets[f"face:{name}:{slot}"] = np.asarray(data)

    cells = f_cells + z_cells

    # enforce int type, empty numpy arrays have type float64
    f_cell_ids = np.asarray(f_cell_ids, dtype=int)
    z_cell_ids = np.asarray(z_cell_ids, dtype=int)
    z_offset = len(f_cell_ids)

    cell_ids = np.concatenate([f_cell_ids, z_cell_ids + z_offset], dtype=np.int64)

    cell_blocks = [
        (key, np.array(indices)[:, flac3d_to_meshio_order[key]])
        for key, indices in cells
    ]

    # sanity check, but not really necessary
    # _, counts = np.unique(z_ cell_ids, return_counts=True)
    # assert np.all(counts == 1), "Zone cell IDs not unique"
    # _, counts = np.unique(f_ cell_ids, return_counts=True)
    # assert np.all(counts == 1), "Zone cell IDs not unique"

    # FLAC3D contains global cell ids. Create an inverse array that maps the
    # global IDs to the running index (0, 1,..., n) that's used in meshio.
    if len(f_cell_ids) > 0:
        f_inv = np.full(np.max(f_cell_ids) + 1, -1)
        f_inv[f_cell_ids] = np.arange(len(f_cell_ids))
        f_cell_sets = {key: f_inv[value] for key, value in f_cell_sets.items()}
    if len(z_cell_ids) > 0:
        z_inv = np.full(np.max(z_cell_ids) + 1, -1)
        z_inv[z_cell_ids] = np.arange(len(z_cell_ids))
        z_cell_sets = {
            key: z_inv[value] + z_offset for key, value in z_cell_sets.items()
        }

    cell_sets = _merge(f_cell_sets, z_cell_sets)

    # cell_sets contains the indices into the global cell list. Since this is
    # split up into blocks, we need to split the cell_sets, too.
    bins = np.cumsum([len(cb[1]) for cb in cell_blocks])
    for key, data in cell_sets.items():
        d = np.digitize(data, bins)
        cell_sets[key] = [data[d == k] for k in range(len(cell_blocks))]

    # assert len(cell_ids) == sum(len(block) for _, block in cell_blocks)

    # also store the cell_ids
    cell_data = {}
    if len(cell_blocks) > 0:
        cell_data = {
            "cell_ids": np.split(
                cell_ids, np.cumsum([len(block) for _, block in cell_blocks][:-1])
            )
        }

    return Mesh(
        points=np.array(points),
        cells=cell_blocks,
        cell_data=cell_data,
        cell_sets=cell_sets,
    )


def _read_point_ascii(buf_or_line):
    """Read point coordinates."""
    pid = int(buf_or_line[1])
    point = [float(l) for l in buf_or_line[2:]]
    return pid, point


def _read_point_binary(buf_or_line):
    """Read point coordinates."""
    pid, x, y, z = struct.unpack("<I3d", buf_or_line.read(28))
    return pid, [x, y, z]


def _read_cell_ascii(buf_or_line, point_ids):
    """Read cell connectivity."""
    cid = int(buf_or_line[2])
    cell = buf_or_line[3:]
    is_b7 = buf_or_line[1] == "B7"
    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])
    return cid, cell


def _read_cell_binary(buf_or_line, point_ids):
    """Read cell connectivity."""
    cid, num_verts = struct.unpack("<2I", buf_or_line.read(8))
    cell = struct.unpack(f"<{num_verts}I", buf_or_line.read(4 * num_verts))
    is_b7 = num_verts == 7
    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])
    return cid, cell


def _read_cell_group_binary(buf_or_line):
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
    return name, slot, data


def _read_cell_group_ascii(buf_or_line, line: str):
    # a group line read
    # ```
    # ZGROUP 'group five' SLOT 5
    # ```
    m = re.match(r"^([A-Z]+) *[\'\"](.*?)[\'\"] *([A-Z]+) *(.*?) *$", line)
    if m is None:
        raise ReadError(
            'Expected line of the form\n```\nZGROUP "group name" SLOT 5\n```\n '
            + f"but got \n```\n{line}\n```\n"
        )
    assert m.group(1) in {"ZGROUP", "FGROUP"}
    assert m.group(3) == "SLOT"
    name = m.group(2)
    slot = m.group(4)

    i = buf_or_line.tell()
    line = buf_or_line.readline()
    data = []
    while True:
        line = line.rstrip().split()
        if line and (line[0] not in {"*", "ZGROUP", "FGROUP"}):
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


def split_f_z(mesh):
    # FLAC3D makes a difference between ZONES (3D-cells only) and FACES
    # (2D-cells only). Split cells into zcells and fcells, along with the cell
    # sets etc.
    zcells = []
    fcells = []
    for cell_block in mesh.cells:
        if cell_block.type in meshio_only["zone"]:
            zcells.append(cell_block)
        elif cell_block.type in meshio_only["face"]:
            fcells.append(cell_block)

    zsets = {}
    fsets = {}
    for key, cset in mesh.cell_sets.items():
        zsets[key] = []
        fsets[key] = []
        for cell_block, sblock in zip(mesh.cells, cset):
            zsets[key].append(
                sblock if cell_block.type in meshio_only["zone"] else None
            )
            fsets[key].append(
                sblock if cell_block.type in meshio_only["face"] else None
            )

    # remove the data that is only None
    zsets = {
        key: value
        for key, value in zsets.items()
        if not all(item is None for item in value)
    }
    fsets = {
        key: value
        for key, value in fsets.items()
        if not all(item is None for item in value)
    }

    # Right now, the zsets contain indices into the corresponding cell block.
    # FLAC3D expects _global_ indices. Update.
    cell_block_sizes = [len(cb) for cb in zcells]
    for key, data in zsets.items():
        gid = 0
        for n, block in zip(cell_block_sizes, data):
            block += gid
            gid += n

    # TODO not sure if fcells and zcells share a common global index
    cell_block_sizes = [len(cb) for cb in fcells]
    for key, data in fsets.items():
        gid = 0
        for n, block in zip(cell_block_sizes, data):
            block += gid
            gid += n

    for label, values in zsets.items():
        zsets[label] = np.concatenate(values)
    for label, values in fsets.items():
        fsets[label] = np.concatenate(values)

    # flac3d indices start at 1
    for label, values in zsets.items():
        zsets[label] += 1
    for label, values in fsets.items():
        fsets[label] += 1

    return zcells, fcells, zsets, fsets


def write(filename, mesh: Mesh, float_fmt: str = ".16e", binary: bool = False):
    """Write FLAC3D f3grid grid file."""
    skip = [c.type for c in mesh.cells if c.type not in meshio_only["zone"]]
    if skip:
        warn(f'FLAC3D format only supports 3D cells. Skipping {", ".join(skip)}.')

    # split into face/zone data
    zcells, fcells, zsets, fsets = split_f_z(mesh)

    mode = "wb" if binary else "w"
    with open_file(filename, mode) as f:
        if binary:
            # Don't know what these values represent
            f.write(struct.pack("<2I", 1375135718, 3))
        else:
            f.write(f"* FLAC3D grid produced by meshio v{version}\n")
            f.write(f"* {time.ctime()}\n")

        _write_points(f, mesh.points, binary, float_fmt)
        # Make gid an array such that its value can be persitently altered
        # inside the functions.
        gid = np.array(0)
        #
        cells = _translate_zcells(mesh.points, mesh.cells)
        _write_cells(f, cells, "zone", binary, gid)
        _write_groups(f, mesh.cells, zsets, "zone", binary)
        #
        cells = _translate_fcells(fcells)
        _write_cells(f, cells, "face", binary, gid)
        _write_groups(f, mesh.cells, fsets, "face", binary)


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


def _write_cells(f, cells, flag: str, binary: bool, gid):
    """Write cells."""
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
                    np.arange(1, num_cells + 1) + gid,
                    np.full(num_cells, num_verts),
                    cdata + 1,
                )
            ).astype(int)
            f.write(struct.pack(f"<{(num_verts + 2) * num_cells}I", *tmp.ravel()))
            gid += num_cells
    else:
        entity = "ZONES" if flag == "zone" else "FACES"
        abbrev = entity[0]

        f.write(f"* {entity}\n")
        for ctype, cdata in cells:
            fmt = f"{abbrev} {{}} {{}} " + " ".join(["{}"] * cdata.shape[1]) + "\n"
            for entry in cdata + 1:
                gid += 1
                f.write(fmt.format(meshio_to_flac3d_type[ctype], gid, *entry))


def _write_groups(f, cells, materials, flag, binary) -> None:
    """Write groups."""
    if materials is None:
        if binary:
            f.write(struct.pack("<I", 0))
        return

    # TODO filter materials by zones/faces

    if binary:
        f.write(struct.pack("<I", len(materials)))
        for label, group in materials.items():
            num_chars, num_zones = len(label), len(group)
            fmt = f"<H{num_chars}sH7sI{num_zones}I"
            tmp = [
                num_chars,
                label.encode(),
                7,
                b"Default",  # slot
                num_zones,
                *group,
            ]
            f.write(struct.pack(fmt, *tmp))
    else:
        flg = "ZGROUP" if flag == "zone" else "FGROUP"

        f.write(f"* {flag.upper()} GROUPS\n")
        for label, group in materials.items():
            f.write(f'{flg} "{label}" SLOT 1\n')
            _write_table(f, group)


def _translate_zcells(points, cells):
    """Reorder meshio cells to FLAC3D zones.

    Four first points must form a right-handed coordinate system (outward
    normal vectors). Reorder corner points according to sign of scalar triple
    products.
    """
    # See <https://stackoverflow.com/a/42386330/353337>
    def slicing_summing(a, b, c):
        c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
        c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2

    zones = []
    for cell_block in cells:
        assert cell_block.type in meshio_only["zone"]

        # Compute scalar triple products
        key = meshio_only["zone"][cell_block.type]
        tmp = points[cell_block.data[:, meshio_to_flac3d_order[key][:4]].T]
        det = slicing_summing(tmp[1] - tmp[0], tmp[2] - tmp[0], tmp[3] - tmp[0])

        # Reorder corner points
        data = np.where(
            (det > 0)[:, None],
            cell_block.data[:, meshio_to_flac3d_order[key]],
            cell_block.data[:, meshio_to_flac3d_order_2[key]],
        )
        zones.append((key, data))

    return zones


def _translate_fcells(cells):
    """Reorder meshio cells to FLAC3D faces."""
    faces = []
    for cell_block in cells:
        ctype, data = cell_block
        assert ctype in meshio_only["face"]

        key = meshio_only["face"][ctype]
        data = cell_block.data[:, meshio_to_flac3d_order[key]]
        faces.append((key, data))

    return faces


def _translate_groups(cells, cell_data, field_data, flag):
    """Convert meshio cell_data to FLAC3D groups."""
    dim = np.concatenate(
        [np.full(len(c.data), 2 if c.type in meshio_only["face"] else 3) for c in cells]
    )
    numdim = flag_to_numdim[flag]
    groups = {
        k: np.nonzero(np.logical_and(cell_data == k, dim == numdim))[0] + 1
        for k in np.unique(cell_data)
    }
    groups = {k: v for k, v in groups.items() if v.size}

    labels = {k: str(k) for k in groups.keys()}
    labels[0] = "None"
    if field_data:
        labels.update(
            {v[0]: k for k, v in field_data.items() if v[1] == flag_to_numdim[flag]}
        )

    return dict(zip(labels.values(), groups.values()))


def _write_table(f, data, ncol: int = 20):
    """Write group data table."""
    nrow = len(data) // ncol
    lines = np.split(data, np.full(nrow, ncol).cumsum())
    for line in lines:
        if len(line):
            f.write(" {}\n".format(" ".join([str(l) for l in line])))


register_format("flac3d", [".f3grid"], read, {"flac3d": write})
