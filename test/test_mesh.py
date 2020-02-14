import copy

import numpy

import helpers
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
