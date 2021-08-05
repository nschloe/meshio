import copy

import numpy as np
import pytest
from numpy.testing import assert_equal

import meshio

from . import helpers


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
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [("triangle", [[0, 1, 2]]), ("triangle", [[0, 2, 3]])],
        cell_data={"a": [[0.5], [1.3]]},
    )
    assert len(mesh.cells_dict) == 1
    assert_equal(mesh.cells_dict, {"triangle": [[0, 1, 2], [0, 2, 3]]})
    assert_equal(mesh.cell_data_dict, {"a": {"triangle": [0.5, 1.3]}})


def test_sets_to_int_data():
    mesh = helpers.tri_mesh_5
    mesh = helpers.add_point_sets(mesh)
    mesh = helpers.add_cell_sets(mesh)

    mesh.sets_to_int_data()

    assert mesh.cell_sets == {}
    assert_equal(mesh.cell_data, {"grain0-grain1": [[0, 0, 1, 1, 1]]})

    assert mesh.point_sets == {}
    assert_equal(mesh.point_data, {"fixed-loose": [0, 0, 0, 1, 1, 1, 1]})

    # now back to set data
    mesh.int_data_to_sets()

    assert mesh.cell_data == {}
    assert_equal(mesh.cell_sets, {"grain0": [[0, 1]], "grain1": [[2, 3, 4]]})

    assert mesh.point_data == {}
    assert_equal(mesh.point_sets, {"fixed": [0, 1, 2], "loose": [3, 4, 5, 6]})


def test_sets_to_int_data_warning():
    mesh = meshio.Mesh(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        {"triangle": [[0, 1, 2], [1, 2, 3]]},
        cell_sets={"tag": [[0]]},
    )
    with pytest.warns(UserWarning):
        mesh.sets_to_int_data()
    assert np.all(mesh.cell_data["tag"] == np.array([[0, -1]]))

    mesh = meshio.Mesh(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        {"triangle": [[0, 1, 2], [1, 2, 3]]},
        point_sets={"tag": [[0, 1, 3]]},
    )
    with pytest.warns(UserWarning):
        mesh.sets_to_int_data()

    assert np.all(mesh.point_data["tag"] == np.array([[0, 0, -1, 0]]))


def test_int_data_to_sets():
    mesh = helpers.tri_mesh
    mesh.cell_data = {"grain0-grain1": [np.array([0, 1])]}

    mesh.int_data_to_sets()

    assert_equal(mesh.cell_sets, {"grain0": [[0]], "grain1": [[1]]})


def test_gh_1165():
    mesh = meshio.Mesh(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        {
            "triangle": [[0, 1, 2], [1, 2, 3]],
            "line": [[0, 1], [0, 2], [1, 3], [2, 3]],
        },
        cell_sets={
            "test": [[], [1]],
            "sets": [[0, 1], [0, 2, 3]],
        },
    )

    mesh.sets_to_int_data()
    mesh.int_data_to_sets()

    assert_equal(mesh.cell_sets, {"test": [[], [1]], "sets": [[0, 1], [0, 2, 3]]})


if __name__ == "__main__":
    # test_sets_to_int_data()
    test_int_data_to_sets()
