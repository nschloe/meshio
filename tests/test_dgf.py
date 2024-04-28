import pathlib

import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "filename, ref_cell_type, ref_num_cells, ref_num_points, ref_dim",
    [
        ("line.dgf", "line", 2, 3, 3),
        ("triangle.dgf", "triangle", 6, 7, 2),
        ("cube.dgf", "quad", 3, 7, 2),
    ],
)
def test_dgf(filename, ref_cell_type, ref_num_cells, ref_num_points, ref_dim, tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "dgf" / filename
    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cell_type == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_points
    assert mesh.points.shape[1] == ref_dim
    helpers.write_read(
        tmp_path, meshio.dgf.write, meshio.dgf.read, mesh, 1.0e-15, extension=".dgf"
    )
