# -*- coding: utf-8 -*-
#
import helpers

import pytest

vtk = pytest.importorskip('vtk')


@pytest.mark.parametrize('mesh', [
        # helpers.quad_mesh,
        helpers.tri_mesh,
        # helpers.tet_mesh,
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        ])
def test_io(mesh):
    # Only single precision; see
    # <https://gitlab.kitware.com/vtk/vtk/issues/17161>.
    helpers.write_read('test.e', 'exodus', mesh, 1.0e-6)
    return
