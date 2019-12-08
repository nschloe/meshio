import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_wkt(mesh):
    def writer(*args, **kwargs):
        return meshio.wkt.write(*args, **kwargs)

    helpers.write_read(writer, meshio.wkt.read, mesh, 1.0e-12)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("simple.wkt", 4, 2), ("whitespaced.wkt", 3.2, 2)],
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "wkt", filename)
    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells["triangle"]) == ref_num_cells
