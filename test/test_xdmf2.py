# -*- coding: utf-8 -*-
#
import helpers

import pytest

vtk = pytest.importorskip('vtk')

@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tet_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 1)
        ])
def test_xdmf2(mesh):
    # FIXME data is only stored in single precision
    # <https://gitlab.kitware.com/vtk/vtk/issues/17037>
    helpers.write_read('test.xdmf', 'xdmf2', mesh, 1.0e-6)
    return
