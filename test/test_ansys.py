# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        ])
def test_ascii(mesh):
    def writer(*args, **kwargs):
        return meshio.ansys_io.write(*args, write_binary=False, **kwargs)

    helpers.write_read(writer, meshio.ansys_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        ])
def test_binary(mesh):
    def writer(*args, **kwargs):
        return meshio.ansys_io.write(*args, write_binary=True, **kwargs)

    helpers.write_read(writer, meshio.ansys_io.read, mesh, 1.0e-15)
    return
