import numpy as np
import pytest

import meshio

from . import helpers

test_set = [
    helpers.tet_mesh,
    helpers.hex_mesh,
    helpers.pyramid_mesh,
]


# Takes a triangle face as 3 points ('face'), and a fourth point ('point')
# Returns True if the face's right-hand normal points *towards* the point
def point_is_above_face(face, point):
    # Triangle's right-hand-rule normal vector
    n = np.cross(face[1] - face[0], face[2] - face[0])
    # Distance from face to point
    d = np.dot(point - face[0], n)
    return d > 0


def assert_node_order(points, cell):
    # Coordinates of each vertex
    ps = np.array([points[x] for x in cell])
    # Geometric center
    center = np.mean(ps, axis=0)

    n = len(cell)
    if n == 4:
        # Tetra
        assert point_is_above_face(ps[:3], ps[3])
    elif n == 8:
        # Hex
        assert point_is_above_face(ps[:3], center)
        assert point_is_above_face([ps[x] for x in [0, 2, 3]], center)
        assert point_is_above_face([ps[x] for x in [4, 6, 5]], center)
        assert point_is_above_face([ps[x] for x in [4, 7, 6]], center)
    elif n == 5:
        # pyramid
        assert point_is_above_face(ps[:3], ps[4])
        assert point_is_above_face([ps[x] for x in [0, 2, 3]], ps[4])
    else:
        print(f"Unsupported element type (node count is {n})")
        assert False


@pytest.mark.parametrize("mesh", test_set)
def test_write_mesh(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshio.openfoam.write, meshio.openfoam.read, mesh, 1.0e-15
    )
