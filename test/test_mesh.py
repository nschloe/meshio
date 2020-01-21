import copy

import numpy

import helpers


def test_print_prune():
    mesh = copy.deepcopy(helpers.tri_mesh)
    print(mesh)
    mesh.prune()


def test_cells_dict():
    mesh = copy.deepcopy(helpers.tri_mesh)
    assert len(mesh.cells_dict) == 1
    assert numpy.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])
