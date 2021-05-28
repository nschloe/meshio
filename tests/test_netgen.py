import helpers
import pytest

import meshio

test_set = [
    helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh_2d,
    helpers.tri_mesh,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
]


@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    helpers.write_read(
        meshio.netgen.write, meshio.netgen.read, mesh, 1.0e-13, extension=".vol"
    )
    helpers.write_read(
        meshio.netgen.write, meshio.netgen.read, mesh, 1.0e-13, extension=".vol.gz"
    )
