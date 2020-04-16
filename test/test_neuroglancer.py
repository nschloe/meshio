import pathlib

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tri_mesh])
def test_neuroglancer(mesh):
    def writer(*args, **kwargs):
        return meshio.neuroglancer.write(*args, **kwargs)

    # 32bit only
    helpers.write_read(writer, meshio.neuroglancer.read, mesh, 1.0e-8)


@pytest.mark.parametrize("filename, ref_sum, ref_num_cells", [("simple1", 20, 4)])
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "neuroglancer" / filename

    mesh = meshio.read(filename, "neuroglancer")
    tol = 1.0e-5
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells[0].data) == ref_num_cells
