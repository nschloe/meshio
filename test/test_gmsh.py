import copy
import os
from functools import partial

import pytest

import helpers
import meshio


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
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 9),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh2(mesh, binary):
    writer = partial(meshio._gmsh.write, fmt_version="2", binary=binary)

    helpers.write_read(writer, meshio._gmsh.read, mesh, 1.0e-15)
    return


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
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 9),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh40(mesh, binary):
    writer = partial(meshio._gmsh.write, fmt_version="4.0", binary=binary)

    helpers.write_read(writer, meshio._gmsh.read, mesh, 1.0e-15)
    return


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
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 9),
        helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
        gmsh_periodic(),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test_gmsh41(mesh, binary):
    writer = partial(meshio._gmsh.write, fmt_version="4.1", binary=binary)

    helpers.write_read(writer, meshio._gmsh.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.msh")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.msh")
    return


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("insulated-2.2.msh", 2.001762136876221, {"line": 21, "triangle": 111})],
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file(filename, ref_sum, ref_num_cells, binary):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "msh", filename)
    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells
    assert {
        k: len(v["gmsh:physical"]) for k, v in mesh.cell_data.items()
    } == ref_num_cells

    writer = partial(meshio._gmsh.write, fmt_version="2", binary=binary)
    helpers.write_read(writer, meshio._gmsh.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells",
    [("insulated-4.1.msh", 2.001762136876221, {"line": 21, "triangle": 111})],
)
def test_reference_file_readonly(filename, ref_sum, ref_num_cells):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "msh", filename)

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum) < tol * ref_sum
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells
    assert {
        k: len(v["gmsh:physical"]) for k, v in mesh.cell_data.items()
    } == ref_num_cells
