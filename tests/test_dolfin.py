import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tet_mesh,
        helpers.add_cell_data(
            helpers.tri_mesh, [("a", (), float), ("b", (), np.int64)]
        ),
    ],
)
def test_dolfin(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.dolfin.write, meshio.dolfin.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.xml")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.xml")
