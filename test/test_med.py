# -*- coding: utf-8 -*-
#
import pytest

import meshio

import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh_2d,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.hex_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, 1),
        helpers.add_cell_data(helpers.tri_mesh, 2),
        helpers.add_cell_data(helpers.tri_mesh, 3),
    ],
)
def test_io(mesh):
    helpers.write_read(meshio.med_io.write, meshio.med_io.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.med")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.med")
    return


@pytest.mark.parametrize(
    "filename, md5, ref_sum_points, ref_num_cells, ref_sum_point_tags, ref_sum_cell_tags",
    [
        (
            "med/cylinder.med",
            "e36b365542c72ef470b83fc21f4dad58",
            16.53169892762988,
            {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4},
            52,
            {"pyramid": -116, "quad": -75, "line": -48, "tetra": -24, "triangle": -30},
        )
    ],
)
def test_reference_file(
    filename, md5, ref_sum_points, ref_num_cells, ref_sum_point_tags, ref_sum_cell_tags
):
    filename = helpers.download(filename, md5)

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = mesh.points.sum()
    assert abs(s - ref_sum_points) < tol * ref_sum_points
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells
    assert mesh.point_data["point_tags"].sum() == ref_sum_point_tags
    assert {
        k: v["cell_tags"].sum() for k, v in mesh.cell_data.items()
    } == ref_sum_cell_tags
    helpers.write_read(meshio.med_io.write, meshio.med_io.read, mesh, 1.0e-15)
