# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
def test_io(mesh):
    helpers.write_read(meshio.medit_io.write, meshio.medit_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.mesh")
    return
