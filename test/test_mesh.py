import copy

import helpers
import numpy as np
import pytest

import meshio


def test_public_attributes():
    # Just make sure this is here
    meshio.extension_to_filetype


@pytest.mark.parametrize(
    "mesh",
    [helpers.tri_mesh, helpers.empty_mesh],
)
def test_print_prune(mesh):
    mesh = copy.deepcopy(mesh)
    print(mesh)
    mesh.remove_orphaned_nodes()
    mesh.remove_lower_dimensional_cells()
    mesh.prune_z_0()


def test_remove_orphaned():
    points = np.array(
        [
            [3.14, 2.71],  # orphaned
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    cells = np.array([[1, 2, 3]])
    a = {"a": np.array([0.1, 0.2, 0.3, 0.4])}
    mesh = meshio.Mesh(points, {"triangle": cells}, point_data=a)
    mesh.remove_orphaned_nodes()

    assert len(mesh.points) == 3
    assert len(mesh.point_data["a"]) == 3
    # make sure the dict `a` wasn't changed,
    # <https://github.com/nschloe/meshio/pull/994>
    assert len(a["a"]) == 4
    assert np.all(mesh.cells[0].data == [0, 1, 2])


def test_cells_dict():
    mesh = copy.deepcopy(helpers.tri_mesh)
    assert len(mesh.cells_dict) == 1
    assert np.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])

    # two cells groups
    mesh = meshio.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        / 3,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("triangle", np.array([[0, 2, 3]])),
        ],
        cell_data={"a": [[0.5], [1.3]]},
    )
    assert len(mesh.cells_dict) == 1
    assert np.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(mesh.cell_data_dict["a"]["triangle"], [0.5, 1.3])


def test_sets_to_int_data():
    mesh = helpers.add_cell_sets(helpers.tri_mesh)

    mesh.sets_to_int_data()
    assert "grain0-grain1" in mesh.cell_data
    assert np.all(mesh.cell_data["grain0-grain1"][0] == [0, 1])


def test_int_data_to_sets():
    mesh = helpers.tri_mesh
    mesh.cell_data = {"grain0-grain1": [np.array([0, 1])]}

    mesh.int_data_to_sets()
    assert "grain0" in mesh.cell_sets
    assert np.all(mesh.cell_sets["grain0"][0] == [0])
    assert "grain1" in mesh.cell_sets
    assert np.all(mesh.cell_sets["grain1"][0] == [1])


if __name__ == "__main__":
    # test_sets_to_int_data()
    test_int_data_to_sets()
