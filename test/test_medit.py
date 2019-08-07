import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
def test_io(mesh):
    helpers.write_read(meshio._medit.write, meshio._medit.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.mesh")
    return
