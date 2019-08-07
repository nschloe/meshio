import numpy
import pytest

import helpers
import meshio

lxml = pytest.importorskip("lxml")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tet_mesh,
        helpers.add_cell_data(helpers.tri_mesh, 1, dtype=float),
        helpers.add_cell_data(helpers.tri_mesh, 1, dtype=numpy.int32),
    ],
)
def test_dolfin(mesh):
    helpers.write_read(meshio._dolfin.write, meshio._dolfin.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.xml")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.xml")
    return
