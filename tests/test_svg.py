import pathlib
import tempfile

import pytest

import meshio

from . import helpers

test_set = [
    helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.quad_mesh,
]


@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = pathlib.Path(temp_dir) / "out.svg"
        meshio.write_points_cells(filepath, mesh.points, mesh.cells)
