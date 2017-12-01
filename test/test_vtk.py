# -*- coding: utf-8 -*-
#
import helpers

import pytest

import meshio

import legacy_reader

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
        return meshio.vtk_io.write(*args, write_binary=False, **kwargs)

    helpers.write_read2(writer, meshio.vtk_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', test_set)
def test_ascii_legacy1(mesh):
    # test with legacy writer
    def legacy_writer(*args, **kwargs):
        return meshio.legacy_writer.write('vtk-ascii', *args, **kwargs)

    # The legacy writer only writes with low precision.
    helpers.write_read2(legacy_writer, meshio.vtk_io.read, mesh, 1.0e-11)


@pytest.mark.parametrize('mesh', test_set)
def test_ascii_legacy2(mesh):
    def writer(*args, **kwargs):
        return meshio.vtk_io.write(*args, write_binary=False, **kwargs)

    # test with legacy reader
    # The legacy writer only writes with low precision.
    def lr(filename):
        return legacy_reader.read('vtk-ascii', filename)

    helpers.write_read2(writer, lr, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', test_set)
def test_binary(mesh):
    def writer(*args, **kwargs):
        return meshio.vtk_io.write(*args, write_binary=True, **kwargs)

    helpers.write_read2(writer, meshio.vtk_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize('mesh', test_set)
# test with legacy writer
def test_binary_legacy1(mesh):
    def legacy_writer(
            filename, points, cells, point_data, cell_data, field_data
            ):
        return meshio.legacy_writer.write(
            'vtk-binary',
            filename, points, cells, point_data, cell_data, field_data
            )

    # The legacy writer only writes with low precision.
    helpers.write_read2(legacy_writer, meshio.vtk_io.read, mesh, 1.0e-11)
    return


@pytest.mark.parametrize('mesh', test_set)
# test with legacy reader
def test_binary_legacy2(mesh):
    def writer(*args, **kwargs):
        return meshio.vtk_io.write(*args, write_binary=True, **kwargs)

    # The legacy writer only writes with low precision.
    def lr(filename):
        return legacy_reader.read('vtk-binary', filename)

    helpers.write_read2(writer, lr, mesh, 1.0e-15)
    return


if __name__ == '__main__':
    test_binary(helpers.tri_mesh)
