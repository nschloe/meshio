import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
@pytest.mark.parametrize(
    "accuracy,ext",
    [
        (1.0e-7, ".ugrid"),
        (1.0e-15, ".b8.ugrid"),
        (1.0e-7, ".b4.ugrid"),
        (1.0e-15, ".lb8.ugrid"),
        (1.0e-7, ".lb4.ugrid"),
        (1.0e-15, ".r8.ugrid"),
        (1.0e-7, ".r4.ugrid"),
        (1.0e-15, ".lr8.ugrid"),
        (1.0e-7, ".lr4.ugrid"),
    ],
)
def test_io(mesh, accuracy, ext):
    helpers.write_read(meshio.ugrid.write, meshio.ugrid.read, mesh, accuracy, ext)


# def test_generic_io():
#    helpers.generic_io("test.mesh")
#    # With additional, insignificant suffix:
#    helpers.generic_io("test.0.mesh")
