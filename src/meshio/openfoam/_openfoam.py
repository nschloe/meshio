"""
I/O for OpenFOAM polyMesh format
<https://cfd.direct/openfoam/user-guide/v6-mesh-description>
"""
import logging
import os
import numpy as np

from collections import OrderedDict as odict

from .._files import open_file
from .._helpers import register


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
        os.mkdir(poly_mesh_dir)
    except OSError:
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
        if cell_type == "tetra":
            cell_order = [
                [0, 2, 1],
                [1, 2, 3],
                [0, 1, 3],
                [0, 3, 2],
            ]
        elif cell_type == "pyramid":
            cell_order = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 4, 3]]
        else:
            print(f"WARNING: Unknown type {cell_type}")
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
    with open(os.path.join(poly_mesh_dir, "boundary"), "w") as f:
        f.write(_foam_header("polyBoundaryMesh", "boundary"))
        f.write(f"{len(patch_names) + 1}\n(\n")
        patch_dict = {
            "type": "patch",
            "physicalType": "patch",
            "startFace": num_internal,
            "nFaces": len(unnamed_boundary_faces),
        }
        f.write(_foam_props("defaultPatch", patch_dict))
        patch_dict["startFace"] += len(unnamed_boundary_faces)
        for i, patch_name in enumerate(patch_names):
            patch_dict["nFaces"] = len(named_patches[i])
            f.write(_foam_props(patch_name, patch_dict))
            patch_dict["startFace"] += len(named_patches[i])
        f.write(_foam_footer())


register("openfoam", [], None, {"openfoam": write})
