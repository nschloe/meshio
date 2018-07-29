# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        # helpers.add_point_data(helpers.tri_mesh, 1), # NOTE: Data not supported yet
        # helpers.add_point_data(helpers.tri_mesh, 3),
        # helpers.add_point_data(helpers.tri_mesh, 9),
        # helpers.add_cell_data(helpers.tri_mesh, 1),
        # helpers.add_cell_data(helpers.tri_mesh, 3),
        # helpers.add_cell_data(helpers.tri_mesh, 9),
        # helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        # helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        # helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
def test_io(mesh):
    helpers.write_read(meshio.mdpa_io.write, meshio.mdpa_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.mesh")
    return
