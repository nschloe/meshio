# -*- coding: utf-8 -*-
#
import copy
import pytest

import meshio

import helpers


def gmsh_periodic():
    mesh = copy.deepcopy(helpers.quad_mesh)
    trns = "Affine{}".format(" 0" * 16)  # just for io testing
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
        helpers.tri_quad_mesh,
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
@pytest.mark.parametrize("write_binary", [False, True])
def test_gmsh2(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.msh_io.write(*args, "2", write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.msh_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
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
@pytest.mark.parametrize("write_binary", [False, True])
def test_gmsh4(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio.msh_io.write(*args, "4", write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio.msh_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.msh")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.msh")
    return
