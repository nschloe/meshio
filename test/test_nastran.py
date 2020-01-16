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
    helpers.write_read(meshio.nastran.write, meshio.nastran.read, mesh, 1.0e-15)


@pytest.mark.parametrize("filename", ["cylinder.fem", "cylinder_cells_first.fem"])
def test_reference_file(filename):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "nastran", filename)
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 16.5316866)

    # Cells
    ref_num_cells = {
        "line": 241,
        "triangle": 171,
        "quad": 721,
        "pyramid": 1180,
        "tetra": 5309,
    }
    assert {k: v.sum() for k, v in mesh.cells} == ref_num_cells
