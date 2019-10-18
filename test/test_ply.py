import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_ply(mesh):
    def writer(*args, **kwargs):
        return meshio._ply.write(*args, **kwargs)

    mesh.cells["triangle"] = mesh.cells["triangle"].astype(numpy.int32)

    helpers.write_read(writer, meshio._ply.read, mesh, 1.0e-12)
    return
