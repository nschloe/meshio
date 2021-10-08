import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.polygon_mesh,
    ],
)
def test_obj(mesh, tmp_path):
    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(np.int32))

    helpers.write_read(tmp_path, meshio.obj.write, meshio.obj.read, mesh, 1.0e-12)


@pytest.mark.skip("Fails point data consistency check.")
@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("elephav.obj", 3.678372172450000e05, 1148)]
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "obj" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells
