import helpers
import pytest

import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test(mesh, binary):
    def writer(*args, **kwargs):
        return meshio.ansys.write(*args, binary=binary, **kwargs)

    helpers.write_read(writer, meshio.ansys.read, mesh, 1.0e-15)
