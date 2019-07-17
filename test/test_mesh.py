import copy

import helpers


def test():
    mesh = copy.deepcopy(helpers.tri_mesh)
    print(mesh)
    mesh.prune()
    return
