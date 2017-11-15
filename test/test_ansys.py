# -*- coding: utf-8 -*-
#
import helpers

import pytest


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        ])
def test_ascii(mesh):
    helpers.write_read('test.msh', 'ansys-ascii', mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        ])
def test_binary(mesh):
    helpers.write_read('test.msh', 'ansys-binary', mesh, 1.0e-15)
    return
