import copy
import sys

import helpers
import numpy
import pytest

import meshio


def test_public_attributes():
    # Just make sure this is here
    meshio.extension_to_filetype


def test_print_prune():
    mesh = copy.deepcopy(helpers.tri_mesh)
    print(mesh)
    mesh.prune()


def test_cells_dict():
    mesh = copy.deepcopy(helpers.tri_mesh)
    assert len(mesh.cells_dict) == 1
    assert numpy.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])

    # two cells groups
    mesh = meshio.Mesh(
        numpy.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        / 3,
        [
            ("triangle", numpy.array([[0, 1, 2]])),
            ("triangle", numpy.array([[0, 2, 3]])),
        ],
        cell_data={"a": [[0.5], [1.3]]},
    )
    assert len(mesh.cells_dict) == 1
    assert numpy.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])
    assert numpy.array_equal(mesh.cell_data_dict["a"]["triangle"], [0.5, 1.3])


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires Python 3.6 or higher")
def test_sets_to_int_data():
    mesh = helpers.add_cell_sets(helpers.tri_mesh)

    mesh.sets_to_int_data()
    assert "grain0-grain1" in mesh.cell_data
    assert numpy.all(mesh.cell_data["grain0-grain1"][0] == [0, 1])


def test_int_data_to_sets():
    mesh = helpers.tri_mesh
    mesh.cell_data = {"grain0-grain1": [numpy.array([0, 1])]}

    mesh.int_data_to_sets()
    assert "grain0" in mesh.cell_sets
    assert numpy.all(mesh.cell_sets["grain0"][0] == [0])
    assert "grain1" in mesh.cell_sets
    assert numpy.all(mesh.cell_sets["grain1"][0] == [1])


if __name__ == "__main__":
    # test_sets_to_int_data()
    test_int_data_to_sets()
