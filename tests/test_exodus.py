import pytest

import meshio

from . import helpers

netCDF4 = pytest.importorskip("netCDF4")

test_set = [
    helpers.empty_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_point_sets(helpers.tri_mesh),
    helpers.add_point_sets(helpers.tet_mesh),
]


@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.exodus.write, meshio.exodus.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.e")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.e")
