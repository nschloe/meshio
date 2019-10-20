import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh", [helpers.tri_mesh, helpers.quad_mesh, helpers.tri_quad_mesh]
)
def test_ply(mesh):
    def writer(*args, **kwargs):
        return meshio._obj.write(*args, **kwargs)

    for key in mesh.cells:
        mesh.cells[key] = mesh.cells[key].astype(numpy.int32)

    helpers.write_read(writer, meshio._obj.read, mesh, 1.0e-12)
    return


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("elephav.obj", 3.678372172450000e05, 1148)]
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "obj", filename)
    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells["triangle"]) == ref_num_cells
    return
