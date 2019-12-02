import helpers
import meshio
import pytest


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_io(mesh):
    helpers.write_read(meshio._off.write, meshio._off.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.off")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.off")
    return
