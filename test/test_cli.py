import tempfile

import numpy

import helpers
import meshio


def test_cli():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh4-ascii")

    outfile = tempfile.NamedTemporaryFile().name

    meshio._cli.info([infile, "--input-format", "gmsh"])

    meshio._cli.convert(
        [infile, outfile, "--input-format", "gmsh", "--output-format", "vtk-binary"]
    )

    mesh = meshio.read(outfile, file_format="vtk")

    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)

    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)
