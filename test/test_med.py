# -*- coding: utf-8 -*-
#
import pytest
import meshio
import helpers
import numpy


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


def test_reference_file_with_mixed_cells():
    filename = "med/cylinder.med"
    md5 = "e36b365542c72ef470b83fc21f4dad58"
    filename = helpers.download(filename, md5)
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 16.53169892762988)

    # Cells
    ref_num_cells = {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4}
    assert {k: len(v) for k, v in mesh.cells.items()} == ref_num_cells

    # Point tags
    assert mesh.point_data["point_tags"].sum() == 52
    ref_point_tags_info = {2: ["Side"], 3: ["Side", "Top"], 4: ["Top"]}
    assert mesh.point_tags == ref_point_tags_info

    # Cell tags
    ref_sum_cell_tags = {
        "pyramid": -116,
        "quad": -75,
        "line": -48,
        "tetra": -24,
        "triangle": -30,
    }
    assert {
        k: v["cell_tags"].sum() for k, v in mesh.cell_data.items()
    } == ref_sum_cell_tags
    ref_cell_tags_info = {
        -6: ["Top circle"],
        -7: ["Top", "Top and down"],
        -8: ["Top and down"],
        -9: ["A", "B"],
        -10: ["B"],
        -11: ["B", "C"],
        -12: ["C"],
    }
    assert mesh.cell_tags == ref_cell_tags_info

    helpers.write_read(meshio.med_io.write, meshio.med_io.read, mesh, 1.0e-15)


def test_reference_file_with_point_cell_data():
    filename = "med/box.med"
    md5 = "0867fb11bd14b83ad11ab20e2b1fd57d"
    filename = helpers.download(filename, md5)
    mesh = meshio.read(filename)

    # Points
    assert numpy.isclose(mesh.points.sum(), 12)

    # Cells
    assert {k: len(v) for k, v in mesh.cells.items()} == {"hexahedron": 1}

    # Point data
    data_u = mesh.point_data["resu____DEPL"]
    assert data_u.shape == (8, 3)
    assert numpy.isclose(data_u.sum(), 12)

    # Cell data
    # ELNO (1 data point for every node of each element)
    data_eps = mesh.cell_data["hexahedron"]["resu____EPSI_ELNO"]
    assert data_eps.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_eps_mean = numpy.mean(data_eps, axis=1)[0]
    eps_ref = numpy.array([1, 0, 0, 0.5, 0.5, 0])
    assert numpy.allclose(data_eps_mean, eps_ref)

    data_sig = mesh.cell_data["hexahedron"]["resu____SIEF_ELNO"]
    assert data_sig.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_sig_mean = numpy.mean(data_sig, axis=1)[0]
    sig_ref = numpy.array(
        [7328.44611253, 2645.87030114, 2034.06063679, 1202.6, 569.752, 0]
    )
    assert numpy.allclose(data_sig_mean, sig_ref)

    data_psi = mesh.cell_data["hexahedron"]["resu____ENEL_ELNO"]
    assert data_psi.shape == (1, 8, 1)  # (n_cells, n_nodes_per_element, n_components)

    # ELEM (1 data point for each element)
    data_psi_elem = mesh.cell_data["hexahedron"]["resu____ENEL_ELEM"]
    assert numpy.isclose(numpy.mean(data_psi, axis=1)[0, 0], data_psi_elem[0])

    helpers.write_read(meshio.med_io.write, meshio.med_io.read, mesh, 1.0e-15)
