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
        ]


@pytest.mark.parametrize('mesh', test_set)
def test_gmsh(mesh):
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.gmsh_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=False
            )

    helpers.write_read2(writer, meshio.gmsh_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', test_set)
def test_gmsh_binary(mesh):
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.gmsh_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=True
            )

    helpers.write_read2(writer, meshio.gmsh_io.read, mesh, 1.0e-15)
    return
