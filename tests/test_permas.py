import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        # helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        # helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.permas.write, meshio.permas.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.post")
    helpers.generic_io(tmp_path / "test.post.gz")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.post")
    helpers.generic_io(tmp_path / "test.0.post.gz")
