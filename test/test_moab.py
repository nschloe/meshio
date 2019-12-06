import pytest

import helpers
import meshio

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh", [helpers.line_mesh, helpers.tri_mesh, helpers.tri_mesh_2d, helpers.tet_mesh]
)
def test_io(mesh):
    helpers.write_read(meshio.h5m.write, meshio.h5m.read, mesh, 1.0e-15)


def test_generic_io():
    helpers.generic_io("test.h5m")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.h5m")
