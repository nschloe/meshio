import pytest

import helpers
import meshio

test_set = [helpers.tet_mesh]


@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    helpers.write_read(
        meshio._tetgen.write, meshio._tetgen.read, mesh, 1.0e-15, extension=".node"
    )
    return
