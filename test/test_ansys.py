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
@pytest.mark.parametrize("write_binary", [False, True])
def test(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.ansys_io.write(*args, write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.ansys_io.read, mesh, 1.0e-15)
    return
