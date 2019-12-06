import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_io(mesh):
    helpers.write_read(meshio.off.write, meshio.off.read, mesh, 1.0e-15)


def test_generic_io():
    helpers.generic_io("test.off")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.off")
