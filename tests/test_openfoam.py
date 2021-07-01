import helpers
import pytest

import meshio

test_set = [
    helpers.tet_mesh,
    helpers.hex_mesh,
    helpers.pyramid_mesh,
]


@pytest.mark.parametrize("mesh", test_set)
def test_write_mesh(mesh):
    helpers.write_read(meshio.openfoam.write, meshio.openfoam.read, mesh, 1.0e-15)
