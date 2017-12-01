# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers

vtk = pytest.importorskip('vtk')


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        # helpers.triangle6_mesh,
        helpers.quad_mesh,
        # helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        # helpers.tet10_mesh,
        helpers.hex_mesh,
        # helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        ])
def test_io(mesh):
    helpers.write_read(
            meshio.exodus_io.write,
            meshio.exodus_io.read,
            mesh, 1.0e-15
            )
    return
