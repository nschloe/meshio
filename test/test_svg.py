import os
import tempfile

import helpers
import pytest

import meshio

test_set = [helpers.line_mesh, helpers.tri_mesh, helpers.tri_mesh_2d, helpers.quad_mesh]


@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "out.svg")
        meshio.write_points_cells(filepath, mesh.points, mesh.cells)
