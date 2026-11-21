"""
I/O for the Wavefront .obj file format, cf.
<https://en.wikipedia.org/wiki/Wavefront_.obj_file>.
"""

import datetime

import numpy as np

from ..__about__ import __version__
from .._exceptions import WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh


def read(filename):
    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    points = []
    vertex_normals = []
    texture_coords = []
    face_groups = []
    face_group_ids = []
    face_group_id = -1
    while True:
        line = f.readline()

        if not line:
            # EOF
            break

        strip = line.strip()

        if len(strip) == 0 or strip[0] == "#":
            continue

        split = strip.split()

        if split[0] == "v":
            points.append([float(item) for item in split[1:]])
        elif split[0] == "vn":
            vertex_normals.append([float(item) for item in split[1:]])
        elif split[0] == "vt":
            texture_coords.append([float(item) for item in split[1:]])
        elif split[0] == "s":
            # "s 1" or "s off" controls smooth shading
            pass
        elif split[0] == "f":
            # A face corner is v, v/vt, v//vn, or v/vt/vn. Keep all three
            # indices: in OBJ, texture coordinates and normals are indexed
            # per corner, not per point, so a single vertex can carry
            # several different vt (or vn) entries, e.g. at UV seams.
            corners = []
            for item in split[1:]:
                parts = item.split("/")
                v_idx = int(parts[0])
                vt_idx = int(parts[1]) if len(parts) > 1 and parts[1] else None
                vn_idx = int(parts[2]) if len(parts) > 2 and parts[2] else None
                corners.append((v_idx, vt_idx, vn_idx))
            if len(face_groups) == 0 or (
                len(face_groups[-1]) > 0 and len(face_groups[-1][-1]) != len(corners)
            ):
                face_groups.append([])
                face_group_ids.append([])

            face_groups[-1].append(corners)
            face_group_ids[-1].append(face_group_id)
        elif split[0] == "g":
            # new group
            face_groups.append([])
            face_group_ids.append([])
            face_group_id += 1
        else:
            # who knows
            pass

    # There may be empty groups, too. <https://github.com/nschloe/meshio/issues/770>
    # Remove them.
    face_groups = [f for f in face_groups if len(f) > 0]
    face_group_ids = [g for g in face_group_ids if len(g) > 0]

    points = np.array(points)
    texture_coords = np.array(texture_coords)
    vertex_normals = np.array(vertex_normals)
    point_data = {}

    def resolve(idx, count):
        # OBJ indices are 1-based; negative indices count back from the end.
        return idx - 1 if idx > 0 else count + idx

    has_vt = any(
        corner[1] is not None for group in face_groups for face in group for corner in face
    )
    has_vn = any(
        corner[2] is not None for group in face_groups for face in group for corner in face
    )

    if has_vt or has_vn:
        # Texture coordinates and normals are per-corner data in OBJ, while
        # meshio point_data is per-point. Reconcile the two by de-indexing
        # (splitting points): every unique (v, vt, vn) corner combination
        # becomes its own point, carrying that corner's vt/vn. Points are
        # only duplicated where a vertex is actually referenced with
        # different vt/vn entries, and no UV/normal information is lost.
        key_to_new_index = {}
        keys = []
        new_face_groups = []
        for group in face_groups:
            new_group = []
            for face in group:
                new_face = []
                for v_idx, vt_idx, vn_idx in face:
                    key = (
                        resolve(v_idx, len(points)),
                        resolve(vt_idx, len(texture_coords))
                        if vt_idx is not None
                        else None,
                        resolve(vn_idx, len(vertex_normals))
                        if vn_idx is not None
                        else None,
                    )
                    if key not in key_to_new_index:
                        key_to_new_index[key] = len(keys)
                        keys.append(key)
                    new_face.append(key_to_new_index[key])
                new_group.append(new_face)
            new_face_groups.append(new_group)

        points = points[[key[0] for key in keys]]
        if has_vt:
            point_data["obj:vt"] = np.array(
                [
                    texture_coords[key[1]]
                    if key[1] is not None
                    else np.zeros(texture_coords.shape[1])
                    for key in keys
                ]
            )
        if has_vn:
            point_data["obj:vn"] = np.array(
                [
                    vertex_normals[key[2]]
                    if key[2] is not None
                    else np.zeros(vertex_normals.shape[1])
                    for key in keys
                ]
            )
        face_groups = new_face_groups
    else:
        # No vt/vn referenced by any face: keep the historical behaviour of
        # attaching the raw vt/vn lists as point data, which is only
        # meaningful (and only passes Mesh's consistency check) when they
        # happen to be per-point already. Unreferenced lists of a different
        # length cannot be associated with points and are dropped.
        if len(texture_coords) == len(points) and len(texture_coords) > 0:
            point_data["obj:vt"] = texture_coords
        if len(vertex_normals) == len(points) and len(vertex_normals) > 0:
            point_data["obj:vn"] = vertex_normals
        face_groups = [
            [
                [resolve(corner[0], len(points)) for corner in face]
                for face in group
            ]
            for group in face_groups
        ]

    # convert to numpy arrays
    face_groups = [np.array(f) for f in face_groups]
    cell_data = {"obj:group_ids": []}
    cells = []
    for f, gid in zip(face_groups, face_group_ids):
        if f.shape[1] == 3:
            cells.append(CellBlock("triangle", f))
        elif f.shape[1] == 4:
            cells.append(CellBlock("quad", f))
        else:
            cells.append(CellBlock("polygon", f))
        cell_data["obj:group_ids"].append(gid)

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh):
    for c in mesh.cells:
        if c.type not in ["triangle", "quad", "polygon"]:
            raise WriteError(
                "Wavefront .obj files can only contain triangle or quad cells."
            )

    with open_file(filename, "w") as f:
        f.write(
            "# Created by meshio v{}, {}\n".format(
                __version__, datetime.datetime.now().isoformat()
            )
        )
        for p in mesh.points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        if "obj:vn" in mesh.point_data:
            dat = mesh.point_data["obj:vn"]
            fmt = "vn " + " ".join(["{}"] * dat.shape[1]) + "\n"
            for vn in dat:
                f.write(fmt.format(*vn))

        if "obj:vt" in mesh.point_data:
            dat = mesh.point_data["obj:vt"]
            fmt = "vt " + " ".join(["{}"] * dat.shape[1]) + "\n"
            for vt in dat:
                f.write(fmt.format(*vt))

        for cell_block in mesh.cells:
            fmt = "f " + " ".join(["{}"] * cell_block.data.shape[1]) + "\n"
            for c in cell_block.data:
                f.write(fmt.format(*(c + 1)))


register_format("obj", [".obj"], read, {"obj": write})
