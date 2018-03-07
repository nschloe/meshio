# -*- coding: utf-8 -*-
#
import os
import tempfile

import pytest

import meshio

import helpers


@pytest.mark.parametrize('filename', [
    'test.e',
    'test.med',
    'test.mesh',
    'test.msh',
    'test.xml',
    'test.post',
    'test.h5m',
    'test.off',
    'test.vtk',
    'test.vtu',
    'test.xmf',
    ])
def test_generic_io(filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, filename)

        meshio.write(
            filepath,
            helpers.tri_mesh['points'],
            helpers.tri_mesh['cells'],
            )

        points, cells, _, _, _ = meshio.helpers.read(filepath)

        assert (abs(points - helpers.tri_mesh['points']) < 1.0e-15).all()
        assert (
            helpers.tri_mesh['cells']['triangle'] == cells['triangle']
            ).all()
    return
