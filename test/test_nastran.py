import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh):
    helpers.write_read(meshio._nastran.write, meshio._nastran.read, mesh, 1.0e-15)
    return


def test_reference_file():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "nastran", "cylinder.fem")
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 16.5316866)

    # Cells
    ref_num_cells = {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4}
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells
