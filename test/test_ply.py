import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_ply(mesh):
    def writer(*args, **kwargs):
        return meshio._ply.write(*args, **kwargs)

    mesh.cells["triangle"] = mesh.cells["triangle"].astype(numpy.int32)

    helpers.write_read(writer, meshio._ply.read, mesh, 1.0e-12)
    return


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("bun_zipper_res4.ply", 3.414583969116211e01, 948)],
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "ply", filename)
    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells["triangle"]) == ref_num_cells
    return
