import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.tri_mesh
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.off.write, meshio.off.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.off")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.off")
