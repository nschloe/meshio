# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize('mesh', [
    helpers.tri_mesh
    ])
def test_io(mesh):
    helpers.write_read(
        meshio.off_io.write,
        meshio.off_io.read,
        mesh, 1.0e-15
        )
    return
