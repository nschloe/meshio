# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers

test_set = [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        ]


@pytest.mark.parametrize('mesh', test_set)
def test_gmsh(mesh):
    def writer(*args, **kwargs):
        return meshio.gmsh_io.write(*args, write_binary=False, **kwargs)

    helpers.write_read(writer, meshio.gmsh_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', test_set)
def test_gmsh_binary(mesh):
    def writer(*args, **kwargs):
        return meshio.gmsh_io.write(*args, write_binary=True, **kwargs)

    helpers.write_read(writer, meshio.gmsh_io.read, mesh, 1.0e-15)
    return
