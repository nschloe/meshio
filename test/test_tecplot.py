import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh", [helpers.tri_mesh, helpers.quad_mesh, helpers.tet_mesh, helpers.hex_mesh],
)
def test(mesh):
    helpers.write_read(meshio.tecplot.write, meshio.tecplot.read, mesh, 1.0e-15)
