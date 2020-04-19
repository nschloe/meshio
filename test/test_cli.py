import tempfile

import numpy

import helpers
import meshio


def is_same_mesh(mesh0, mesh1, atol):
    if not numpy.allclose(mesh0.points, mesh1.points, atol=atol, rtol=0.0):
        return False
    for cells0, cells1 in zip(mesh0.cells, mesh1.cells):
        if cells0.type != cells1.type or not numpy.allclose(cells0.data, cells1.data):
            return False
    return True


def test_info():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh")
    meshio._cli.info([infile, "--input-format", "gmsh"])


def test_convert():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile().name
    meshio.write(infile, input_mesh, file_format="gmsh")

    outfile = tempfile.NamedTemporaryFile().name

    meshio._cli.convert(
        [infile, outfile, "--input-format", "gmsh", "--output-format", "vtk"]
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
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)

    meshio._cli.compress([infile])
    mesh = meshio.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)


def test_ascii_binary():
    input_mesh = helpers.tri_mesh
    infile = tempfile.NamedTemporaryFile(suffix=".vtk").name
    meshio.write(infile, input_mesh)

    meshio._cli.ascii([infile])
    mesh = meshio.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)

    meshio._cli.binary([infile])
    mesh = meshio.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)
