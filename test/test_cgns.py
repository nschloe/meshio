import helpers
import meshio
import pytest


@pytest.mark.parametrize("mesh", [helpers.tet_mesh])
def test(mesh):
    def writer(*args, **kwargs):
        return meshio._cgns.write(*args, **kwargs)

    helpers.write_read(writer, meshio._cgns.read, mesh, 1.0e-15)
