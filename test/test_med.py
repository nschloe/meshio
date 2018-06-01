# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.hex_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 2),
        helpers.add_cell_data(helpers.tri_mesh, 3),
    ],
)
def test_io(mesh):
    helpers.write_read(meshio.med_io.write, meshio.med_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.med")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.med")
    return
