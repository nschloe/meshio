import numpy as np
import pytest

import meshio

from . import helpers

test_set_full = [
    helpers.empty_mesh,
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
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
]


@pytest.mark.parametrize("mesh", test_set_full)
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_xdmf3(mesh, compression, tmp_path):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, compression=compression, **kwargs)

    helpers.write_read(tmp_path, write, meshio.xdmf.read, mesh, 1.0e-14)


@pytest.mark.skip
def test_generic_io(tmp_path):
    with pytest.warns(UserWarning):
        helpers.generic_io(tmp_path / "test.hmf")

    with pytest.warns(UserWarning):
        # With additional, insignificant suffix:
        helpers.generic_io(tmp_path / "test.0.hmf")
