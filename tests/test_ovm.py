import pytest

import meshio
from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
     helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        #helpers.polygon_mesh, # we return a 4-polygon as quad
        #helpers.tri_quad_mesh, # we return a combined triangle cell block instead of two
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.pyramid_mesh,
        helpers.wedge_mesh,
#        helpers.add_point_data(helpers.tri_mesh, 3),
        #helpers.add_cell_data(helpers.tri_mesh, [("medit:ref", (), int)]),
    ],
)
def test_ovm(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.ovm.write, meshio.ovm.read, mesh, 1.0e-15)

