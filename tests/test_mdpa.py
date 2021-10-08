import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        # helpers.add_point_data(helpers.tri_mesh, 1), # NOTE: Data not supported yet
        # helpers.add_point_data(helpers.tri_mesh, 3),
        # helpers.add_point_data(helpers.tri_mesh, 9),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        # helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        # helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        # helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.mdpa.write, meshio.mdpa.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.mesh")
