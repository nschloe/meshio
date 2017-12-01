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
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.ansys_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=False
            )

    helpers.write_read2(writer, meshio.ansys_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        ])
def test_binary(mesh):
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.ansys_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=True
            )

    helpers.write_read2(writer, meshio.ansys_io.read, mesh, 1.0e-15)
    return
