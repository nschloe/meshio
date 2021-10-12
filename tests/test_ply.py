import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1, dtype=int),
        helpers.add_point_data(helpers.tri_mesh, 1, dtype=float),
        helpers.line_mesh,
        helpers.polygon_mesh,
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_ply(mesh, binary, tmp_path):
    def writer(*args, **kwargs):
        return meshio.ply.write(*args, binary=binary, **kwargs)

    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(np.int32))

    helpers.write_read(tmp_path, writer, meshio.ply.read, mesh, 1.0e-12)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [
        ("bun_zipper_res4.ply", 3.414583969116211e01, 948),
        ("tet.ply", 6, 4),
    ],
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "ply" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.get_cells_type("triangle")) == ref_num_cells


@pytest.mark.parametrize("binary", [False, True])
def test_no_cells(binary):
    import io

    vertices = np.random.random((30, 3))
    mesh = meshio.Mesh(vertices, [])
    file = io.BytesIO()
    mesh.write(file, "ply", binary=binary)
    mesh2 = meshio.read(io.BytesIO(file.getvalue()), "ply")
    assert np.array_equal(mesh.points, mesh2.points)
    assert len(mesh2.cells) == 0
