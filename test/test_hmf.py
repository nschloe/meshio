import helpers
import numpy
import pytest

import meshio

test_set_full = [
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.line_tri_mesh,
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
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), numpy.float64)]),
]


@pytest.mark.parametrize("mesh", test_set_full)
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_xdmf3(mesh, compression):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, compression=compression, **kwargs)

    helpers.write_read(write, meshio.xdmf.read, mesh, 1.0e-14)


def test_generic_io():
    helpers.generic_io("test.hmf")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.hmf")
