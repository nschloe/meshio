# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers
import legacy_reader
import legacy_writer

vtk = pytest.importorskip("vtk")
lxml = pytest.importorskip("lxml")

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
@pytest.mark.parametrize("write_binary", [False, True])
def test(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.vtu_io.write(
            *args,
            write_binary=write_binary,
            # don't use pretty xml to increase test coverage
            pretty_xml=False,
            **kwargs
        )

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/issues/17038#note_264052>
    tol = 1.0e-15 if write_binary else 1.0e-11
    helpers.write_read(writer, meshio.vtu_io.read, mesh, tol)
    return


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [False, True])
def test_legacy_writer(mesh, write_binary):
    # test with legacy writer
    def lw(*args, **kwargs):
        mode = "vtu-binary" if write_binary else "vtu-ascii"
        return legacy_writer.write(mode, *args, **kwargs)

    # The legacy writer only writes with low precision.
    tol = 1.0e-15 if write_binary else 1.0e-11
    helpers.write_read(lw, meshio.vtu_io.read, mesh, tol)
    return


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [False, True])
def test_legacy_reader(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.vtu_io.write(*args, write_binary=write_binary, **kwargs)

    # test with legacy reader
    def lr(filename):
        mode = "vtu-binary" if write_binary else "vtu-ascii"
        return legacy_reader.read(mode, filename)

    # the legacy reader only reads at low precision
    tol = 1.0e-15 if write_binary else 1.0e-11
    helpers.write_read(writer, lr, mesh, tol)
    return


def test_generic_io():
    helpers.generic_io("test.vtu")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.vtu")
    return


if __name__ == "__main__":
    test(helpers.tet10_mesh, write_binary=False)
