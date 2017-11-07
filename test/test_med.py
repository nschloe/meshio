# -*- coding: utf-8 -*-
#
import helpers

import pytest

h5py = pytest.importorskip('h5py')


@pytest.mark.parametrize('mesh', [
        helpers.tri_mesh,
        helpers.tet_mesh,
        ])
def test_io(mesh):
    helpers.write_read('test.med', 'med', mesh, 1.0e-15)
    return
