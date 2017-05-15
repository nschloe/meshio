# -*- coding: utf-8 -*-
#
import helpers

import pytest


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh
        ])
def test_io(mesh):
    helpers.write_read('test.off', 'off', mesh, 1.0e-15)
    return
