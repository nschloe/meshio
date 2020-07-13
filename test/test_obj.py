import pathlib

import helpers
import numpy
import pytest

import meshio


@pytest.mark.parametrize(
    "mesh", [helpers.tri_mesh, helpers.quad_mesh, helpers.tri_quad_mesh]
)
def test_obj(mesh):
    def writer(*args, **kwargs):
        return meshio.obj.write(*args, **kwargs)

    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(numpy.int32))

    helpers.write_read(writer, meshio.obj.read, mesh, 1.0e-12)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("elephav.obj", 3.678372172450000e05, 1148)]
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "obj" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells
