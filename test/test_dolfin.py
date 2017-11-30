# -*- coding: utf-8 -*-
#
import helpers

import pytest


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.tet_mesh,
        helpers.add_cell_data(helpers.tri_mesh, 1),
        ])
def test_io(mesh):
    helpers.write_read('test.xml', 'dolfin-xml', mesh, 1.0e-15)
    return
