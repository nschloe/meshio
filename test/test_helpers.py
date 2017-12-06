# -*- coding: utf-8 -*-
#
import meshio

import helpers


def test_generic_reader():
    meshio.vtk_io.write(
            'test.vtk',
            helpers.tri_mesh['points'],
            helpers.tri_mesh['cells'],
            )

    points, cells, _, _, _ = meshio.helpers.read('test.vtk')
    assert (abs(points - helpers.tri_mesh['points']) < 1.0e-15).all()
    assert (helpers.tri_mesh['cells']['triangle'] == cells['triangle']).all()
    return
