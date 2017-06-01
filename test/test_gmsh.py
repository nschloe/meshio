# -*- coding: utf-8 -*-
#
import helpers

import pytest


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        ])
def test_gmsh(mesh):
    helpers.write_read('test.msh', 'gmsh-ascii', mesh, 1.0e-15)
    return


# @pytest.mark.parametrize('mesh', [
#         helpers.tri_mesh,
#         helpers.quad_mesh,
#         helpers.tri_quad_mesh,
#         helpers.tet_mesh,
#         ])
# def test_binary(mesh):
#     helpers.write_read('test.msh', 'gmsh-binary', mesh, 1.0e-15)
#     return
