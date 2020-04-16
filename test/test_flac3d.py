import pathlib
import sys

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tet_mesh,
        helpers.hex_mesh,
        # helpers.add_cell_data(helpers.tet_mesh, [("a", (), int)]),  # TODO
    ],
)
def test(mesh):
    helpers.write_read(meshio.flac3d.write, meshio.flac3d.read, mesh, 1.0e-15)


# the failure perhaps has to do with dictionary ordering
@pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails with 3.5")
def test_reference_file():
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "flac3d" / "flac3d_mesh_ex.f3grid"

    mesh = meshio.read(filename)

    # points
    assert numpy.isclose(mesh.points.sum(), 307.0)

    # cells
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
    ref_sum_cell_data = [45, 9, 18, 9, 6, 3, 6, 3, 6, 3]
    assert [len(arr) for arr in mesh.cell_data["flac3d:zone"]] == ref_sum_cell_data
