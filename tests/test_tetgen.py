import pathlib

import pytest

import meshio

from . import helpers

test_set = [
    # helpers.empty_mesh,
    helpers.tet_mesh
]


@pytest.mark.parametrize("mesh", test_set)
def test(mesh, tmp_path):
    helpers.write_read(
        tmp_path,
        meshio.tetgen.write,
        meshio.tetgen.read,
        mesh,
        1.0e-15,
        extension=".node",
    )


@pytest.mark.parametrize(
    "filename, point_ref_sum, cell_ref_sum", [("mesh.ele", 12, 373)]
)
def test_point_cell_refs(filename, point_ref_sum, cell_ref_sum):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "tetgen" / filename

    mesh = meshio.read(filename)
    assert mesh.point_data["tetgen:ref"].sum() == point_ref_sum
    assert mesh.cell_data["tetgen:ref"][0].sum() == cell_ref_sum
