import copy
import pathlib
from functools import partial

import numpy as np
import pytest

import meshio

from . import helpers


def gmsh_periodic():
    mesh = copy.deepcopy(helpers.quad_mesh)
    trns = [0] * 16  # just for io testing
    mesh.gmsh_periodic = [
        [0, (3, 1), None, [[2, 0]]],
        [0, (4, 6), None, [[3, 5]]],
        [1, (2, 1), trns, [[5, 0], [4, 1], [4, 2]]],
    ]
    return mesh


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh22(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh40(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="4.0", binary=binary)

    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        # helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.tri_mesh, 9),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41(mesh, binary, tmp_path):
    writer = partial(meshio.gmsh.write, fmt_version="4.1", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.msh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.msh")


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("insulated-2.2.msh", 2.001762136876221, [21, 111])],
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file(filename, ref_sum, ref_num_cells, binary, tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "msh" / filename
    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert [c.type for c in mesh.cells] == ["line", "triangle"]
    assert [len(c.data) for c in mesh.cells] == ref_num_cells
    assert list(map(len, mesh.cell_data["gmsh:geometrical"])) == ref_num_cells
    assert list(map(len, mesh.cell_data["gmsh:physical"])) == ref_num_cells

    writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=binary)
    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cells_in_cell_sets",
    [
        (
            "insulated-4.1.msh",
            2.001762136876221,
            {"line": 21, "triangle": 111},
            {"line": 27, "triangle": 120},
        )
    ],
    # Note that testing on number of cells in
    # cell_sets_dict will count both cells associated with physical tags, and
    # bounding entities.
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file_with_entities(
    filename, ref_sum, ref_num_cells, ref_num_cells_in_cell_sets, binary, tmp_path
):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "msh" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert {k: len(v) for k, v in mesh.cells_dict.items()} == ref_num_cells
    assert {
        k: len(v) for k, v in mesh.cell_data_dict["gmsh:physical"].items()
    } == ref_num_cells

    writer = partial(meshio.gmsh.write, fmt_version="4.1", binary=binary)

    num_cells = {k: 0 for k in ref_num_cells_in_cell_sets}
    for vv in mesh.cell_sets_dict.values():
        for k, v in vv.items():
            num_cells[k] += len(v)
    assert num_cells == ref_num_cells_in_cell_sets

    helpers.write_read(tmp_path, writer, meshio.gmsh.read, mesh, 1.0e-15)
