import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tet_mesh,
        helpers.add_cell_data(
            helpers.tri_mesh, [("a", (), float), ("b", (), numpy.int64)]
        ),
    ],
)
def test_dolfin(mesh):
    helpers.write_read(meshio.dolfin.write, meshio.dolfin.read, mesh, 1.0e-15)


def test_generic_io():
    helpers.generic_io("test.xml")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.xml")
