# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers
import legacy_reader
import legacy_writer

vtk = pytest.importorskip("vtk")


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


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [True, False])
def test(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.vtk_io.write(*args, write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.vtk_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [True, False])
def test_legacy_writer(mesh, write_binary):
    # test with legacy writer
    def lw(*args, **kwargs):
        mode = "vtk-binary" if write_binary else "vtk-ascii"
        return legacy_writer.write(mode, *args, **kwargs)

    # The legacy writer only writes with low precision.
    helpers.write_read(lw, meshio.vtk_io.read, mesh, 1.0e-11)
    return


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [True, False])
def test_legacy_reader(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.vtk_io.write(*args, write_binary=write_binary, **kwargs)

    # test with legacy reader
    def lr(filename):
        mode = "vtk-binary" if write_binary else "vtk-ascii"
        return legacy_reader.read(mode, filename)

    helpers.write_read(writer, lr, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.vtk")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.vtk")
    return


if __name__ == "__main__":
    test(helpers.tri_mesh, write_binary=True)
