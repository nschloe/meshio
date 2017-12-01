# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers

vtk = pytest.importorskip('vtk')

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
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, 2),
    helpers.add_cell_data(helpers.tri_mesh, 3),
    ]


@pytest.mark.parametrize('mesh', test_set)
def test_ascii(mesh):
    def writer(*args, **kwargs):
        return meshio.vtu_io.write(*args, write_binary=False, **kwargs)

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/issues/17038#note_264052>
    helpers.write_read(writer, meshio.vtu_io.read, mesh, 1.0e-11)
    return


@pytest.mark.parametrize('mesh', test_set)
def test_binary(mesh):
    def writer(*args, **kwargs):
        return meshio.vtu_io.write(*args, write_binary=True, **kwargs)

    helpers.write_read(writer, meshio.vtu_io.read, mesh, 1.0e-15)
    return


if __name__ == '__main__':
    test_ascii(helpers.tet10_mesh)
