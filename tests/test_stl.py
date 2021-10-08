import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [helpers.empty_mesh, helpers.tri_mesh],
)
@pytest.mark.parametrize(
    "binary, tol",
    [
        (False, 1.0e-15),
        # binary STL only operates in single precision
        (True, 1.0e-8),
    ],
)
def test_stl(mesh, binary, tol, tmp_path):
    def writer(*args, **kwargs):
        return meshio.stl.write(*args, binary=binary, **kwargs)

    helpers.write_read(tmp_path, writer, meshio.stl.read, mesh, tol)
