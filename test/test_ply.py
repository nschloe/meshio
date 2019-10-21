import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.tri_mesh,
        # helpers.quad_mesh,
        # helpers.tri_quad_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1, dtype=int),
        helpers.add_point_data(helpers.tri_mesh, 1, dtype=float),
        # helpers.add_cell_data(helpers.tri_mesh, 1),
        # helpers.add_cell_data(helpers.tri_mesh, 3),
        # helpers.add_cell_data(helpers.tri_mesh, 9),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_ply(mesh, binary):
    def writer(*args, **kwargs):
        return meshio._ply.write(*args, binary=binary, **kwargs)

    for key in mesh.cells:
        mesh.cells[key] = mesh.cells[key].astype(numpy.int32)

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
