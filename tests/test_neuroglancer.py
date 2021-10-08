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
def test_neuroglancer(mesh, tmp_path):
    def writer(*args, **kwargs):
        return meshio.neuroglancer.write(*args, **kwargs)

    # 32bit only
    helpers.write_read(tmp_path, writer, meshio.neuroglancer.read, mesh, 1.0e-8)


@pytest.mark.parametrize("filename, ref_sum, ref_num_cells", [("simple1", 20, 4)])
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "neuroglancer" / filename

    mesh = meshio.read(filename, "neuroglancer")
    tol = 1.0e-5
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells[0].data) == ref_num_cells
