import helpers
import pytest

import meshio


@pytest.mark.parametrize("mesh", [helpers.tet_mesh])
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.cgns.write(*args, **kwargs)

    helpers.write_read(writer, meshio.cgns.read, mesh, 1.0e-15)
