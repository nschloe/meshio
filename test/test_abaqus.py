# -*- coding: utf-8 -*-
#
import numpy
import pytest

import meshio

import helpers


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
    ],
)
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.abaqus_io.write(*args, **kwargs)

    helpers.write_read(writer, meshio.abaqus_io.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize(
    "filename, md5, ref_sum, ref_num_cells",
    [
        (
            "abaqus/abaqus_mesh_ex.inp",
            "e0a9a7a88b25d9fadccdd653c91e33ea",
            -68501.914611293,
            3492,
        ),
        ("abaqus/UUea.inp", "d76e526eeced5f79ba867d496559002a", 4950.0, 50),
    ],
)
@pytest.mark.parametrize("write_binary", [False, True])
def test_reference_file(filename, md5, ref_sum, ref_num_cells, write_binary):
    filename = helpers.download(filename, md5)

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert len(mesh.cells["quad"]) == ref_num_cells
    return
