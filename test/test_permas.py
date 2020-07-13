import helpers
import pytest

import meshio


@pytest.mark.parametrize(
    "mesh",
    [
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
def test_io(mesh):
    helpers.write_read(meshio.permas.write, meshio.permas.read, mesh, 1.0e-15)


def test_generic_io():
    helpers.generic_io("test.post")
    helpers.generic_io("test.post.gz")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.post")
    helpers.generic_io("test.0.post.gz")
