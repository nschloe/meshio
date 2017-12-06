# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers


@pytest.mark.parametrize('filename', [
    'test.e',
    'test.med',
    'test.mesh',
    'test.msh',
    'test.xml',
    'test.post.gz',
    'test.h5m',
    'test.off',
    'test.vtk',
    'test.vtu',
    'test.xmf',
    ])
def test_generic_io(filename):
    meshio.write(
            'test.vtk',
            helpers.tri_mesh['points'],
            helpers.tri_mesh['cells'],
            )

    points, cells, _, _, _ = meshio.helpers.read('test.vtk')
    assert (abs(points - helpers.tri_mesh['points']) < 1.0e-15).all()
    assert (
        helpers.tri_mesh['cells']['triangle'] == cells['triangle']
        ).all()
    return
