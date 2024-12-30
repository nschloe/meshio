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
    points, vertex_normals, texture_coords, face_groups, face_group_ids = [], [], [], [], []
    face_group_id = -1

    while True:
        line = f.readline()
        if not line:
            break
        process_line(line.strip(), points, vertex_normals, texture_coords, face_groups, face_group_ids, face_group_id)

    face_groups, face_group_ids = clean_empty_groups(face_groups, face_group_ids)
    points, texture_coords, vertex_normals, point_data = convert_to_numpy(points, texture_coords, vertex_normals)
    cells, cell_data = create_cells(face_groups, face_group_ids)

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def process_line(strip, points, vertex_normals, texture_coords, face_groups, face_group_ids, face_group_id):
    if len(strip) == 0 or strip[0] == "#":
        return

    split = strip.split()
    if split[0] == "v":
        points.append([float(item) for item in split[1:]])
    elif split[0] == "vn":
        vertex_normals.append([float(item) for item in split[1:]])
    elif split[0] == "vt":
        texture_coords.append([float(item) for item in split[1:]])
    elif split[0] == "s":
        pass
    elif split[0] == "f":
        process_face(split, face_groups, face_group_ids, face_group_id)
    elif split[0] == "g":
        face_groups.append([])
        face_group_ids.append([])
        face_group_id += 1


def process_face(split, face_groups, face_group_ids, face_group_id):
    dat = [int(item.split("/")[0]) for item in split[1:]]
    if len(face_groups) == 0 or (len(face_groups[-1]) > 0 and len(face_groups[-1][-1]) != len(dat)):
        face_groups.append([])
        face_group_ids.append([])

    face_groups[-1].append(dat)
    face_group_ids[-1].append(face_group_id)


def clean_empty_groups(face_groups, face_group_ids):
    face_groups = [f for f in face_groups if len(f) > 0]
    face_group_ids = [g for g in face_group_ids if len(g) > 0]
    return face_groups, face_group_ids


def convert_to_numpy(points, texture_coords, vertex_normals):
    points = np.array(points)
    texture_coords = np.array(texture_coords)
    vertex_normals = np.array(vertex_normals)
    point_data = {}
    if len(texture_coords) > 0:
        point_data["obj:vt"] = texture_coords
    if len(vertex_normals) > 0:
        point_data["obj:vn"] = vertex_normals
    return points, texture_coords, vertex_normals, point_data


def create_cells(face_groups, face_group_ids):
    face_groups = [np.array(f) for f in face_groups]
    cell_data = {"obj:group_ids": []}
    cells = []
    for f, gid in zip(face_groups, face_group_ids):
        if f.shape[1] == 3:
            cells.append(CellBlock("triangle", f - 1))
        elif f.shape[1] == 4:
            cells.append(CellBlock("quad", f - 1))
        else:
            cells.append(CellBlock("polygon", f - 1))
        cell_data["obj:group_ids"].append(gid)
    return cells, cell_data

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
