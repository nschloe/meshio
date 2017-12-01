# -*- coding: utf-8 -*-
#
import helpers

import pytest

import meshio

vtk = pytest.importorskip('vtk')


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 2),
        helpers.add_cell_data(helpers.tri_mesh, 3),
        ])
def test_ascii(mesh):
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.vtk_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=False
            )

    # # ASCII files are only meant for debugging, VTK stores only 11 digits
    # # <https://gitlab.kitware.com/vtk/vtk/issues/17038#note_264052>
    # helpers.write_read('test.vtk', 'vtk-ascii', mesh, 1.0e-11)

    helpers.write_read2(writer, meshio.vtk_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 2),
        helpers.add_cell_data(helpers.tri_mesh, 3),
        ])
def test_binary(mesh):
    def writer(filename, points, cells, point_data, cell_data, field_data):
        return meshio.vtk_io.write(
            filename, points, cells, point_data, cell_data, field_data,
            write_binary=True
            )

    helpers.write_read2(writer, meshio.vtk_io.read, mesh, 1.0e-15)
    return


if __name__ == '__main__':
    test_binary(helpers.tri_mesh)
