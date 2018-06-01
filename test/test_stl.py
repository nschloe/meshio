# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
@pytest.mark.parametrize(
    "write_binary, tol",
    [
        (False, 1.0e-15),
        # binary STL only operates in single precision
        (True, 1.0e-8),
    ],
)
def test_stl(mesh, write_binary, tol):
    def writer(*args, **kwargs):
        return meshio.stl_io.write(*args, write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.stl_io.read, mesh, tol)
    return
