"""
I/O for OpenFOAM polyMesh format
<https://cfd.direct/openfoam/user-guide/v6-mesh-description>
"""
import logging
import os
from collections import OrderedDict as odict
from enum import Enum

import numpy as np

from .._helpers import register
from .._mesh import CellBlock, Mesh


def _foam_props(name: str, props: dict):
    s = f"{name}\n{{"
    for key, prop in props.items():
        s += f"\t{key}\t\t{str(prop)};\n"
    s += "}\n"
    return s


def _foam_header(class_name: str, object_name: str):
    s = """
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  8
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
"""
    props = {
        "version": "2.0",
        "format": "ascii",
        "class": class_name,
        "location": '"constant/polyMesh"',
        "object": object_name,
    }
    s += _foam_props("FoamFile", props)
    s += """
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    return s


def _foam_footer():
    return """
// ************************************************************************* //
"""


# Return a unique id to represent a set of points (ignore order)
def _face_id(points):
    return tuple(sorted(points))


def write(filename, mesh, binary=False):
    poly_mesh_dir = filename
    try:
        os.makedirs(poly_mesh_dir)
    except FileExistsError:
        logging.warning("Directory already exists. Aborting to be safe.")
        return None

    # Write points file
    with open(os.path.join(poly_mesh_dir, "points"), "w") as f:
        f.write(_foam_header("vectorField", "points"))
        f.write(f"{len(mesh.points)}\n(\n")
        for p in mesh.points:
            f.write(f"({p[0]} {p[1]} {p[2]}) \n")
        f.write(")\n")
        f.write(_foam_footer())

    # Faces
    #  - key is an ordered list of vertices forming face (the 'face_id')
    #    The order of vertices is arbitrary here, so that faces sharing the
    #    same points have the same id.
    #  - Value is a list: [[ordered points], owner index, neighbour index, patch_name]
    #    Ordered points face outwards for owner
    #    Neighbour can be None
    faces = odict()
    # Iterate over cell groups, counting cell index i
    i = 0
    for cell_type, cells in mesh.cells:
        # Define a vertex order for each face which follows right-hand rule
        if cell_type == "tetra":
            cell_order = [[0, 2, 1], [1, 2, 3], [0, 1, 3], [0, 3, 2]]
        elif cell_type == "pyramid":
            cell_order = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 4, 3]]
        elif cell_type == "hexahedron":
            cell_order = [
                [0, 3, 2, 1],
                [0, 1, 5, 4],
                [4, 5, 6, 7],
                [2, 3, 7, 6],
                [1, 2, 6, 5],
                [0, 4, 7, 3],
            ]
        # TODO Wedge etc.
        else:
            logging.warning(f"Unknown type {cell_type}")
            continue
        # Iterate over cells in cell type
        for cell in cells:
            # Iterate over faces in cell
            for face_order in cell_order:
                face_ps = [cell[j] for j in face_order]
                face_id = _face_id(face_ps)
                face = faces.get(face_id, None)
                if face is not None:
                    # Face already registered; we are the neighbour
                    face[2] = i
                else:
                    # Face does not exist; we are the owner
                    faces[face_id] = [face_ps, i, None, None]
            i += 1

    # Physical names
    patch_names = []
    for patch_name, tags in mesh.cell_sets.items():
        patch_names.append(patch_name)
        for idx, elem_tags in enumerate(tags):
            if elem_tags is None:
                continue
            for tag in elem_tags:
                elem_points = mesh.cells[idx][1][tag]
                face_id = _face_id(elem_points)
                face = faces.get(face_id, None)
                if face is not None:
                    face[3] = patch_name
                else:
                    logging.warning(f"Physical tag not found for: {patch_name}")

    # Reorder faces
    # Faces seem to need to be in the following order:
    #  - Internal faces
    #  - Boundary faces (grouped by physical labels)
    internal_faces = odict()
    named_patches = [odict() for i in range(len(patch_names))]
    unnamed_boundary_faces = odict()
    for key, value in faces.items():
        if value[2] is not None:
            internal_faces[key] = value
        else:
            if value[3] is None:
                unnamed_boundary_faces[key] = value
            else:
                named_patches[patch_names.index(value[3])][key] = value
    num_internal = len(internal_faces)
    faces = odict()
    faces.update(internal_faces)
    faces.update(unnamed_boundary_faces)
    for patch in named_patches:
        faces.update(patch)

    # Write faces file
    with open(os.path.join(poly_mesh_dir, "faces"), "w") as f:
        f.write(_foam_header("faceList", "faces"))
        f.write(f"{len(faces)}\n(\n")
        for face in faces.values():
            points_string = " ".join([str(p) for p in face[0]])
            f.write(f"{len(face[0])}({points_string})\n")
        f.write(")\n")
        f.write(_foam_footer())

    # Write owner file
    with open(os.path.join(poly_mesh_dir, "owner"), "w") as f:
        f.write(_foam_header("labelList", "owner"))
        f.write(f"{len(faces)}\n(\n")
        for face in faces.values():
            f.write(f"{face[1]}\n")
        f.write(")\n")
        f.write(_foam_footer())

    # Write neighbour file
    neighboured_faces = list(faces.values())[0:num_internal]
    with open(os.path.join(poly_mesh_dir, "neighbour"), "w") as f:
        f.write(_foam_header("labelList", "neighbour"))
        f.write(f"{len(neighboured_faces)}\n(\n")
        for face in neighboured_faces:
            f.write(f"{face[2]}\n")
        f.write(")\n")
        f.write(_foam_footer())

    # Write boundary file
    include_default_patch = len(unnamed_boundary_faces) > 0
    with open(os.path.join(poly_mesh_dir, "boundary"), "w") as f:
        f.write(_foam_header("polyBoundaryMesh", "boundary"))
        f.write(f"{len(patch_names) + int(include_default_patch)}\n(\n")
        patch_dict = {
            "type": "patch",
            "physicalType": "patch",
            "startFace": num_internal,
            "nFaces": len(unnamed_boundary_faces),
        }
        if include_default_patch:
            f.write(_foam_props("defaultPatch", patch_dict))
        patch_dict["startFace"] += len(unnamed_boundary_faces)
        for i, patch_name in enumerate(patch_names):
            patch_dict["nFaces"] = len(named_patches[i])
            f.write(_foam_props(patch_name, patch_dict))
            patch_dict["startFace"] += len(named_patches[i])
        f.write(")\n")
        f.write(_foam_footer())


class ParsedType(Enum):
    NONE = 0
    STRING = 1
    LIST = 2
    DICT = 3


# Parse the next object in a text stream, recursing into children
# Possible outputs:
#  - (NONE,): Nothing parsed (end of object)
#  - (STRING, String, end_char): Simple value (string), along with the closing character
#  - (LIST, List): A list of other objects
#  - (DICT, (dict_name, Dict)): A dictionary of objects, along with a name for it
def _parse_object(f, end_on_blank=False):
    BLANKS = [" ", "\t", "\n"]
    ENDS = [")", "}", ";", ","]
    stack = ""

    c = None
    while True:
        c = f.read(1)  # Current character to parse
        if c == "":
            # Reached EOF
            break
        if c in ENDS:
            # Reached end of object
            if stack:
                # Return a string
                return (ParsedType.STRING, stack, c)
            else:
                break

        if c in BLANKS:
            if stack == "":
                # Nothing found yet; ignore
                continue
            if end_on_blank:
                # Reached end of string
                return (ParsedType.STRING, stack, " ")
            if stack[-1] != " ":
                # Add space if there isn't one already
                stack += " "

        if c == "/":
            # Check next character for comment
            cur = f.tell()  # Store position in case we need to go back
            c = f.read(1)
            if c == "/":
                # Single line comment; skip to end of line
                while True:
                    c = f.read(1)
                    if c == "\n" or c == "":
                        break
                continue
            if c == "*":
                # Multiline comment; skip until '*/'
                while True:
                    c = f.read(2)
                    if c == "*/" or len(c) < 2:
                        break
                continue
            # Not a comment; go back one and move on
            f.seek(cur)
            c = f.read(1)

        if c == "(":
            # Parse a list
            lis = []
            # Recursively parse all array elements
            while True:
                obj = _parse_object(f, end_on_blank=True)
                if obj[0] == ParsedType.NONE:
                    # End of list
                    break
                if obj[0] == ParsedType.DICT and obj[1][0] == "" and lis:
                    # Edge case:
                    # Dict name was read as a lone string element and dict is nameless
                    # Fix by replacing the string element with named dict
                    lis[-1] = (lis[-1], obj[1][1])
                    continue
                lis.append(obj[1])
                if obj[0] == ParsedType.STRING and obj[2] == ")":
                    # End of list
                    break
            return (ParsedType.LIST, lis)
        if c == "{":
            # Parsing a new odict; stack should contain dict name
            dic = odict()
            dic_name = stack.strip()
            while True:
                key = _parse_object(f, end_on_blank=True)
                if key[0] == ParsedType.NONE:
                    # End of dict (empty key)
                    break
                value = _parse_object(f)
                if value[0] == ParsedType.NONE:
                    # Empty value
                    dic[key[1]] = None
                    continue
                dic[key[1]] = value[1]
                if value[0] == ParsedType.STRING and value[2] == "}":
                    # End of dict (ignore missing ';')
                    break
            return (ParsedType.DICT, (dic_name, dic))

        # Nothing came up; stack character and continue
        stack += c

    # Didn't parse anything of interest
    return (ParsedType.NONE,)


def _read_file(filepath: str):
    with open(filepath, "r") as f:
        objs = []
        while True:
            obj = _parse_object(f)
            if obj[0] == ParsedType.NONE:
                break
            else:
                objs.append(obj[1])
    return objs


def read(dirpath: str):
    points_obj = _read_file(os.path.join(dirpath, "points"))
    points = np.array(points_obj[1], dtype=np.float64)

    faces_obj = _read_file(os.path.join(dirpath, "faces"))
    # faces = [np.array(vec, dtype=np.float64) for vec in faces_obj[1]]
    faces = [[int(c) for c in p] for p in faces_obj[1]]

    owner_obj = _read_file(os.path.join(dirpath, "owner"))
    owners = np.array(owner_obj[1], dtype=np.int64)

    neighbour_obj = _read_file(os.path.join(dirpath, "neighbour"))
    neighbours = np.array(neighbour_obj[1], dtype=np.int64)

    boundary_obj = _read_file(os.path.join(dirpath, "boundary"))
    for patch_name, patch in boundary_obj[1]:
        print(patch_name, patch)

    cells_data = {}

    cells_faces = {}
    # Cell-indexed tuples: (face, is_inverted)
    for face, owner in enumerate(owners):
        if owner in cells_faces:
            cells_faces[owner].append((face, False))
        else:
            cells_faces[owner] = [(face, False)]
    # Negative face index means invert point order
    for face, neighbour in enumerate(neighbours):
        if neighbour in cells_faces:
            cells_faces[neighbour].append((face, True))
        else:
            cells_faces[neighbour] = [(face, True)]

    # Identify cell type by number of faces
    face_types = {4: "tetra", 5: "pyramid", 6: "hexahedron"}
    for fs in cells_faces.values():
        if len(fs) not in face_types:
            logging.warning("Unsupported cell type. Skipping.")
            continue
        type_ = face_types[len(fs)]

        # Faces with left-hand-rule pointing outwards
        faces_lh = []
        for f, inverted in fs:
            face = faces[f]
            if not inverted:
                face.reverse()
            faces_lh.append(face)
        ps = []
        if type_ == "tetra":
            for face in faces_lh:
                for p in face:
                    if p not in ps:
                        ps.append(p)
        elif type_ == "hexahedron":
            for face in faces_lh:
                if not np.isin(face, ps).any():
                    if ps:
                        face.reverse()
                    ps += face
        elif type_ == "pyramid":
            for face in faces_lh:
                if len(face) != 4:
                    continue
                ps += face
                break
            assert ps
            for face in faces_lh:
                if len(face) == 4:
                    continue
                for p in face:
                    if p not in ps:
                        ps.append(p)
                        break
                break

        cell = np.array(ps, dtype=np.int64)
        if type_ not in cells_data:
            cells_data[type_] = [cell]
        else:
            cells_data[type_].append(cell)

    # Convert to Numpy
    cells = []
    for cell_type, block in cells_data.items():
        cells.append(CellBlock(cell_type, np.array(block)))

    return Mesh(points, cells)


register("openfoam", [], read, {"openfoam": write})
