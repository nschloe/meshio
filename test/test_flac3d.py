import copy
import pathlib
import sys

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh, binary, data",
    [
        (helpers.tet_mesh, False, []),
        (helpers.hex_mesh, False, []),
        (helpers.tet_mesh, False, [1, 2]),
        (helpers.tet_mesh, True, []),
        (helpers.hex_mesh, True, []),
        (helpers.tet_mesh, True, [1, 2]),
    ],
)
def test(mesh, binary, data):
    if data:
        mesh = copy.deepcopy(mesh)
        mesh.cell_data["flac3d:zone"] = [numpy.array(data)]
    helpers.write_read(
        lambda f, m: meshio.flac3d.write(f, m, binary=binary),
        meshio.flac3d.read,
        mesh,
        1.0e-15,
    )


# the failure perhaps has to do with dictionary ordering
@pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails with 3.5")
@pytest.mark.parametrize(
    "filename", ["flac3d_mesh_ex.f3grid", "flac3d_mesh_ex_bin.f3grid"],
)
def test_reference_file(filename):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "flac3d" / filename

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
