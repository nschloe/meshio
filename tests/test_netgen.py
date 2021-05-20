import pathlib
from functools import partial

import helpers
import numpy as np
import pytest

import meshio

test_set = [
#     helpers.empty_mesh,
#     helpers.line_mesh,
    helpers.tri_mesh_2d,
    helpers.tri_mesh,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
#     helpers.polygon_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
]

i = 0

@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    global i
    meshio.netgen.write("mesh_{}.vol".format(i), mesh)

    i+=1
