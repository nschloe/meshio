import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.tri_mesh
    ],
)
def test_wkt(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.wkt.write, meshio.wkt.read, mesh, 1.0e-12)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("simple.wkt", 4, 2), ("whitespaced.wkt", 3.2, 2)],
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "wkt" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells
