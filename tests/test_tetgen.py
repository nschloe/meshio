import pathlib

import pytest

import meshio

from . import helpers

test_set = [
    (helpers.tet_mesh, '.node'),
    (helpers.tet_mesh, '.ele'),
    (helpers.tri_mesh, '.face'),
    (helpers.line_mesh, '.edge'),
]


@pytest.mark.parametrize("mesh,extension", test_set)
def test(mesh, extension, tmp_path):
    helpers.write_read(
        tmp_path,
        meshio.tetgen.write,
        meshio.tetgen.read,
        mesh,
        1.0e-15,
        extension=extension,
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
