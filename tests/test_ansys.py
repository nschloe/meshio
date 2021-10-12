import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.pyramid_mesh,
        helpers.wedge_mesh,
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test(mesh, binary, tmp_path):
    def writer(*args, **kwargs):
        return meshio.ansys.write(*args, binary=binary, **kwargs)

    helpers.write_read(tmp_path, writer, meshio.ansys.read, mesh, 1.0e-15)
