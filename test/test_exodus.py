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
    # helpers.add_point_data(helpers.tri_mesh, 2),
    # helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_node_sets(helpers.tri_mesh),
    helpers.add_node_sets(helpers.tet_mesh),
]


@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh):
    helpers.write_read(meshio.exodus_io.write, meshio.exodus_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
def test_legacy_writer(mesh):
    # test with legacy writer
    def lw(*args, **kwargs):
        return legacy_writer.write("exodus", *args, **kwargs)

    # The legacy writer only writes with low precision.
    helpers.write_read(lw, meshio.exodus_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize("mesh", [helpers.tri_mesh, helpers.hex_mesh])
def test_legacy_reader(mesh):
    def lr(filename):
        return legacy_reader.read("exodus", filename)

    helpers.write_read(meshio.exodus_io.write, lr, mesh, 1.0e-4)
    return


def test_generic_io():
    helpers.generic_io("test.e")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.e")
    return
