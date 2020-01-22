import os
import sys

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize("mesh", [helpers.tet_mesh, helpers.hex_mesh])
def test(mesh):
    helpers.write_read(meshio.flac3d.write, meshio.flac3d.read, mesh, 1.0e-15)
    return


# the failure perhaps has to do with dictionary ordering
@pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails with 3.5")
def test_reference_file():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "flac3d", "flac3d_mesh_ex.f3grid")
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 307.0)

    # Cells
    ref_num_cells = [
        ("hexahedron", 45),
        ("pyramid", 9),
        ("hexahedron", 18),
        ("wedge", 9),
        ("hexahedron", 6),
        ("wedge", 3),
        ("hexahedron", 6),
        ("wedge", 3),
        ("pyramid", 6),
        ("tetra", 3),
    ]
    assert [(k, len(v)) for k, v in mesh.cells] == ref_num_cells
    # Cell data
    ref_sum_cell_data = {"tetra": 9, "pyramid": 36, "wedge": 54, "hexahedron": 171}
    assert {
        k: v["flac3d:zone"].sum() for k, v in mesh.cell_data.items()
    } == ref_sum_cell_data
