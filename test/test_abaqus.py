import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.abaqus.write(*args, **kwargs)

    helpers.write_read(writer, meshio.abaqus.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cell_sets",
    [("UUea.inp", 4950.0, 50, 10), ("nle1xf3c.inp", 32.215275528, 12, 2)],
)
def test_reference_file(filename, ref_sum, ref_num_cells, ref_num_cell_sets):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "abaqus", filename)
    mesh = meshio.read(filename)

    assert numpy.isclose(numpy.sum(mesh.points), ref_sum)
    assert sum([len(cells.data) for cells in mesh.cells]) == ref_num_cells
    assert len(mesh.cell_sets) == ref_num_cell_sets
