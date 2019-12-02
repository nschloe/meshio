import helpers
import meshio
import pytest


@pytest.mark.parametrize(
    "mesh",
    [
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
    helpers.write_read(meshio._permas.write, meshio._permas.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.post")
    helpers.generic_io("test.post.gz")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.post")
    helpers.generic_io("test.0.post.gz")
    return
