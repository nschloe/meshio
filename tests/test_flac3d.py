import pathlib
import sys

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.tet_mesh,
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test(mesh, binary, tmp_path):
    helpers.write_read(
        tmp_path,
        lambda f, m: meshio.flac3d.write(f, m, binary=binary),
        meshio.flac3d.read,
        mesh,
        1.0e-15,
    )


# the failure perhaps has to do with dictionary ordering
@pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails with 3.5")
@pytest.mark.parametrize(
    "filename",
    ["flac3d_mesh_ex.f3grid", "flac3d_mesh_ex_bin.f3grid"],
)
def test_reference_file(filename):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "flac3d" / filename

    mesh = meshio.read(filename)

    # points
    assert np.isclose(mesh.points.sum(), 307.0)

    # cells
    ref_num_cells = [
        ("quad", 15),
        ("triangle", 3),
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
    assert [
        (cell_block.type, len(cell_block)) for cell_block in mesh.cells
    ] == ref_num_cells
    # Cell sets
    for arr in mesh.cell_sets.values():
        assert len(arr) == 12
    assert [len(arr) for arr in mesh.cell_sets.values()] == [12, 12, 12, 12, 12]
