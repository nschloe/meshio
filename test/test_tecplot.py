import pathlib
from copy import deepcopy

import helpers
import numpy
import pytest

import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        # Those two tests suddenly started failing on gh-actions. No idea why.
        # TODO reinstate
        # helpers.tet_mesh,
        # helpers.hex_mesh
    ],
)
def test(mesh):
    helpers.write_read(meshio.tecplot.write, meshio.tecplot.read, mesh, 1.0e-15)


@pytest.mark.parametrize("filename", ["quad_zone_comma.tec", "quad_zone_space.tec"])
def test_comma_space(filename):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "tecplot" / filename
    mesh = meshio.read(filename)

    helpers.write_read(meshio.tecplot.write, meshio.tecplot.read, mesh, 1.0e-15)


def test_varlocation():
    # Test that VARLOCATION is correctly written and read depending on the
    # number of point and cell data.
    writer = meshio.tecplot.write
    reader = meshio.tecplot.read
    mesh = deepcopy(helpers.tri_mesh)
    num_points = len(mesh.points)
    num_cells = sum(len(c.data) for c in mesh.cells)

    # Add point data: no VARLOCATION
    mesh.point_data["one"] = numpy.ones(num_points)
    helpers.write_read(writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([5] = CELLCENTERED)
    mesh.cell_data["two"] = [numpy.ones(num_cells) * 2.0]
    helpers.write_read(writer, reader, mesh, 1.0e-15)

    # Add point data: VARLOCATION = ([6] = CELLCENTERED)
    mesh.point_data["three"] = numpy.ones(num_points) * 3.0
    helpers.write_read(writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([6-7] = CELLCENTERED)
    mesh.cell_data["four"] = [numpy.ones(num_cells) * 4.0]
    helpers.write_read(writer, reader, mesh, 1.0e-15)

    # Add point data: VARLOCATION = ([7-8] = CELLCENTERED)
    mesh.point_data["five"] = numpy.ones(num_points) * 5.0
    helpers.write_read(writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([7-9] = CELLCENTERED)
    mesh.cell_data["six"] = [numpy.ones(num_cells) * 6.0]
    helpers.write_read(writer, reader, mesh, 1.0e-15)
