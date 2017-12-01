# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers

vtk = pytest.importorskip('vtk')


@pytest.mark.skipif(not hasattr(vtk, 'vtkXdmf3Writer'), reason='Need XDMF3')
@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tet_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 1)
        ])
def test_xdmf3(mesh):
    helpers.write_read(
        meshio.xdmf_io.write,
        meshio.xdmf_io.read,
        mesh, 1.0e-15
        )
    return
