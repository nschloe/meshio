# -*- coding: utf-8 -*-
#
import meshio
import pytest

import helpers

h5py = pytest.importorskip('h5py')


@pytest.mark.parametrize('mesh', [
    helpers.tri_mesh,
    helpers.quad_mesh,
    helpers.tet_mesh,
    helpers.hex_mesh,
    ])
def test_io(mesh):
    helpers.write_read(
        meshio.med_io.write,
        meshio.med_io.read,
        mesh, 1.0e-15
        )
    return
