import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
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
    def writer(*args, **kwargs):
        return meshio._abaqus.write(*args, **kwargs)

    helpers.write_read(writer, meshio._abaqus.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("abaqus_mesh_ex.inp", -68501.914611293, 3492), ("UUea.inp", 4950.0, 50)],
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file(filename, ref_sum, ref_num_cells, binary):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "abaqus", filename)
    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells["quad"]) == ref_num_cells
    return
