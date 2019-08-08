import os

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tet_mesh, helpers.hex_mesh])
def test(mesh):
    helpers.write_read(meshio._flac3d.write, meshio._flac3d.read, mesh, 1.0e-15)
    return


def test_reference_file():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "flac3d", "flac3d_mesh_ex.f3grid")
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 306.9999999999999)

    # Cells
    ref_num_cells = {"tetra": 3, "pyramid": 15, "wedge": 15, "hexahedron": 75}
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells

    # Cell data
    ref_sum_cell_data = {"tetra": 9, "pyramid": 36, "wedge": 54, "hexahedron": 171}
    assert {
        k: v["flac3d:zone"].sum() for k, v in mesh.cell_data.items()
    } == ref_sum_cell_data
