import copy
import tempfile
from pathlib import Path

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


# Same as the one in 'helpers', but more permissive
def write_read(writer, reader, input_mesh, atol, extension=".dat"):
    """Write and read a file, and make sure the data is the same as before."""
    in_mesh = copy.deepcopy(input_mesh)

    with tempfile.TemporaryDirectory() as temp_dir:
        p = Path(temp_dir) / ("test" + extension)
        writer(p, input_mesh)
        mesh = reader(p)

    # Make sure the output is writeable
    assert mesh.points.flags["WRITEABLE"]
    for cells in input_mesh.cells:
        if isinstance(cells.data, np.ndarray):
            assert cells.data.flags["WRITEABLE"]
        else:
            # This is assumed to be a polyhedron
            for cell in cells.data:
                for face in cell:
                    assert face.flags["WRITEABLE"]

    # assert that the input mesh hasn't changed at all
    assert np.allclose(in_mesh.points, input_mesh.points, atol=atol, rtol=0.0)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.scipy.org/pipermail/numpy-discussion/2015-December/074410.html>.
    # Use allclose.
    if in_mesh.points.shape[0] == 0:
        assert mesh.points.shape[0] == 0
    else:
        n = in_mesh.points.shape[1]
        assert np.allclose(in_mesh.points, mesh.points[:, :n], atol=atol, rtol=0.0)

    # To avoid errors from sorted (below), specify the key as first cell type then index
    # of the first point of the first cell. This may still lead to comparison of what
    # should be different blocks, but chances seem low.
    def cell_sorter(cell):
        if cell.type.startswith("polyhedron"):
            # Polyhedra blocks should be well enough distinguished by their type
            return cell.type
        else:
            return (cell.type, cell.data[0, 0])

    # to make sure we are testing the same type of cells we sort the list
    for cells0, cells1 in zip(
        sorted(input_mesh.cells, key=cell_sorter), sorted(mesh.cells, key=cell_sorter)
    ):
        assert cells0.type == cells1.type, f"{cells0.type} != {cells1.type}"

        if cells0.type.startswith("polyhedron"):
            # Special treatment of polyhedron cells
            # Data is a list (one item per cell) of numpy arrays
            for c_in, c_out in zip(cells0.data, cells1.data):
                for face_in, face_out in zip(c_in, c_out):
                    assert np.allclose(face_in, face_out, atol=atol, rtol=0.0)
        else:
            print("a", cells0.data)
            print("b", cells1.data)
            assert np.array_equal(np.sort(cells0.data), np.sort(cells1.data))
            for cell in cells1.data:
                assert_node_order(mesh.points, cell)

    for key in input_mesh.point_data.keys():
        assert np.allclose(
            input_mesh.point_data[key], mesh.point_data[key], atol=atol, rtol=0.0
        )

    for name, cell_type_data in input_mesh.cell_data.items():
        for d0, d1 in zip(cell_type_data, mesh.cell_data[name]):
            # assert d0.dtype == d1.dtype, (d0.dtype, d1.dtype)
            assert np.allclose(d0, d1, atol=atol, rtol=0.0)

    print()
    print("helpers:")
    print(input_mesh.field_data)
    print()
    print(mesh.field_data)
    for name, data in input_mesh.field_data.items():
        if isinstance(data, list):
            assert data == mesh.field_data[name]
        else:
            assert np.allclose(data, mesh.field_data[name], atol=atol, rtol=0.0)

    # Test of cell sets (assumed to be a list of numpy arrays),
    for name, data in input_mesh.cell_sets.items():
        # Skip the test if the key is not in the read cell set
        if name not in mesh.cell_sets.keys():
            continue
        data2 = mesh.cell_sets[name]
        for var1, var2 in zip(data, data2):
            assert np.allclose(var1, var2, atol=atol, rtol=0.0)


@pytest.mark.parametrize("mesh", test_set)
def test_write_mesh(mesh):
    write_read(meshio.openfoam.write, meshio.openfoam.read, mesh, 1.0e-15)
