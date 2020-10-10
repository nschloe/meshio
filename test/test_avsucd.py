import helpers
import pytest

import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.add_cell_data(
            helpers.tri_mesh,
            [("avsucd:material", (), int), ("a", (), float), ("b", (3,), float)],
        ),
        helpers.add_point_data(helpers.add_point_data(helpers.tri_mesh, 1), 3),
    ],
)
def test(mesh):
    helpers.write_read(meshio.avsucd.write, meshio.avsucd.read, mesh, 1.0e-13)
