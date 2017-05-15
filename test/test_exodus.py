# -*- coding: utf-8 -*-
#
import helpers

import pytest


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        ])
def test_io(mesh):
    # TODO report exodus precision failure
    helpers.write_read('test.e', 'exodus', mesh, 1.0e-8)
    return
