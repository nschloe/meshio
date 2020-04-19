import pytest

import helpers
import meshio

netCDF4 = pytest.importorskip("netCDF4")

test_set = [
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


# The tests on gh suddenly and mysteriously fail with
# ```
#  data[:] = 0.0
# netCDF4/_netCDF4.pyx:4950: in netCDF4._netCDF4.Variable.__setitem__
#     ???
# netCDF4/_netCDF4.pyx:5229: in netCDF4._netCDF4.Variable._put
#     ???
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
# >   ???
# E   RuntimeError: NetCDF: HDF error
#
# netCDF4/_netCDF4.pyx:1887: RuntimeError
# ```
# Skip for now
@pytest.mark.skip("Failing on gh-actions")
@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh):
    helpers.write_read(meshio.exodus.write, meshio.exodus.read, mesh, 1.0e-15)


@pytest.mark.skip("Failing on gh-actions")
def test_generic_io():
    helpers.generic_io("test.e")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.e")
