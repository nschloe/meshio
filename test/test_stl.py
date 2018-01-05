# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize('mesh', [
    helpers.tri_mesh,
    ])
@pytest.mark.parametrize('write_binary', [False, True])
def test_stl(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.stl_io.write(*args, write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.stl_io.read, mesh, 1.0e-15)
    return
