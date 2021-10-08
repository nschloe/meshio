import pytest

import meshio

from . import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tet_mesh,
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.h5m.write, meshio.h5m.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.h5m")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.h5m")
