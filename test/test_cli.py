# -*- coding: utf-8 -*-
#
import tempfile

import numpy

import meshio

import helpers


def test_cli():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh4-ascii")

    outfile = tempfile.NamedTemporaryFile().name

    meshio.cli.main(
        [
            infile,
            outfile,
            "--input-format",
            "gmsh-ascii",
            "--output-format",
            "vtk-binary",
        ]
    )

    mesh = meshio.read(outfile, file_format="vtk-binary")

    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)

    for cell_type, data in input_mesh.cells.items():
        assert numpy.allclose(data, mesh.cells[cell_type])
    return
