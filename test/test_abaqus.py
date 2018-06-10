# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.abaqus_io.write(*args, **kwargs)

    helpers.write_read(writer, meshio.abaqus_io.read, mesh, 1.0e-15)
    return
