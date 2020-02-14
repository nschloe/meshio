import tempfile

import numpy

import helpers
import meshio


def test_info():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh4-ascii")
    meshio._cli.info([infile, "--input-format", "gmsh"])


def test_convert():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh4-ascii")

    outfile = tempfile.NamedTemporaryFile().name

    meshio._cli.convert(
        [infile, outfile, "--input-format", "gmsh", "--output-format", "vtk-binary"]
    )

    mesh = meshio.read(outfile, file_format="vtk")

    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)

    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)


def test_compress():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile(suffix=".vtu").name
    meshio.write(infile, input_mesh)

    meshio._cli.decompress([infile])
    mesh = meshio.read(infile)
    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)
    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)

    meshio._cli.compress([infile])
    mesh = meshio.read(infile)
    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)
    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)


def test_ascii_binary():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile(suffix=".vtk").name
    meshio.write(infile, input_mesh)

    meshio._cli.ascii([infile])
    mesh = meshio.read(infile)
    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)
    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)

    meshio._cli.binary([infile])
    mesh = meshio.read(infile)
    atol = 1.0e-15
    assert numpy.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)
    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert numpy.allclose(cells0.data, cells1.data)
