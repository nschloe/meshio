"""
I/O for OpenFOAM polyMesh format
<https://cfd.direct/openfoam/user-guide/v6-mesh-description>
"""
import logging
import os
import numpy as np

from collections import OrderedDict

from .._files import open_file
from .._helpers import register


def _foam_header(class_name: str, object_name: str):
    s = """
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  8
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
"""
    props = {
        "version": "2.0",
        "format": "ascii",
        "class": class_name,
        "location": '"constant/polyMesh"',
        "object": object_name,
    }
    for key, prop in props.items():
        s += f"\t{key}\t\t{prop};\n"
    s += """
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    return s


def _foam_footer():
    return """
// ************************************************************************* //
"""


def write(filename, mesh, binary=False):
    poly_mesh_dir = filename
    try:
        os.mkdir(poly_mesh_dir)
    except OSError:
        logging.warning("Directory already exists. Aborting to be safe.")
        return None

    # Points
    # with open_file(os.path.join(poly_mesh_dir, 'points')) as f:
    with open(os.path.join(poly_mesh_dir, "points"), "w") as f:
        f.write(_foam_header("vectorField", "points"))
        f.write(f"{len(mesh.points)}\n(\n")
        for p in mesh.points:
            f.write(f"({p[0]} {p[1]} {p[2]}) \n")
        f.write(")\n")
        f.write(_foam_footer())

    # Faces

    # Faces dictionary:
    #  - key is an ordered list of vertices forming face (the 'face_id')
    #    The order of vertices is arbitrary here, so that faces sharing the
    #    same points have the same id.
    #  - Value is a list: [[ordered points], owner index, neighbour index]
    #    Ordered points face outwards for owner
    #    Neighbour can be None
    faces = OrderedDict()
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
                face_id = tuple(set(face_ps))
                face = faces.get(face_id, None)
                if face is not None:
                    # Face already registered; we are the neighbour
                    face[2] = i
                else:
                    # Face does not exist; we are the owner
                    faces[face_id] = [face_ps, i, None]
            i += 1

    # Reorder faces
    # Faces seem to need to be in the following order:
    #  - Internal faces
    #  - Boundary faces (grouped by physical labels)
    # TODO Physical labels

    faces_old = faces.copy()
    faces = OrderedDict()
    for key, value in faces_old.items():
        if value[2] is not None:
            faces[key] = value
    num_internal = len(faces)
    for key, value in faces_old.items():
        if value[2] is None:
            faces[key] = value

    # Temporary check of face ordering
    fs = list(faces.values())
    for i in range(1, len(faces)):
        if fs[i - 1][2] is None and fs[i][2] is not None:
            print("WARNING: Faces are incorrectly ordered.")
            break

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
    # TODO Divide and name based on physical groups
    with open(os.path.join(poly_mesh_dir, "boundary"), "w") as f:
        f.write(_foam_header("polyBoundaryMesh", "boundary"))
        f.write(
            f"""1
(
    defaultPatch
    {{
        type            patch;
        physicalType    patch;
        nFaces          {len(faces)-num_internal};
        startFace       {num_internal};
    }}
)
"""
        )
        f.write(_foam_footer())


register("openfoam", [], None, {"openfoam": write})
