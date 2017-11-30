# -*- coding: utf-8 -*-
#
import helpers

import pytest

vtk = pytest.importorskip('vtk')


@pytest.mark.skipif(not hasattr(vtk, 'vtkXdmf3Writer'), reason='Need XDMF3')
@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tet_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 1)
        ])
def test_xdmf3(mesh):
    helpers.write_read('test.xdmf', 'xdmf', mesh, 1.0e-15)
    return
