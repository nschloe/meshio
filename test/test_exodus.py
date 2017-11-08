# -*- coding: utf-8 -*-
#
import helpers

import pytest

vtk = pytest.importorskip('vtk')


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        ])
def test_io(mesh):
    helpers.write_read('test.e', 'exodus', mesh, 1.0e-15)
    return
