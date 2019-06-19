# -*- coding: utf-8 -*-
#
import pytest

import helpers
import meshio

test_set = [
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
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
    helpers.add_node_sets(helpers.tri_mesh),
    helpers.add_node_sets(helpers.tet_mesh),
]


@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh):
    helpers.write_read(meshio.exodus_io.write, meshio.exodus_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.e")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.e")
    return
