import pathlib
from copy import deepcopy

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        # Those two tests suddenly started failing on gh-actions. No idea why.
        # TODO reinstate
        # helpers.tet_mesh,
        # helpers.hex_mesh
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshio.tecplot.write, meshio.tecplot.read, mesh, 1.0e-15
    )


@pytest.mark.parametrize(
    "filename", ["quad_zone_comma.tec", "quad_zone_space.tec", "quad_zone_multivar.tec"]
)
def test_comma_space(filename, tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "tecplot" / filename
    mesh = meshio.read(filename)

    helpers.write_read(
        tmp_path, meshio.tecplot.write, meshio.tecplot.read, mesh, 1.0e-15
    )


def test_varlocation(tmp_path):
    # Test that VARLOCATION is correctly written and read depending on the
    # number of point and cell data.
    writer = meshio.tecplot.write
    reader = meshio.tecplot.read
    mesh = deepcopy(helpers.tri_mesh)
    num_points = len(mesh.points)
    num_cells = sum(len(c.data) for c in mesh.cells)

    # Add point data: no VARLOCATION
    mesh.point_data["one"] = np.ones(num_points)
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([5] = CELLCENTERED)
    mesh.cell_data["two"] = [np.ones(num_cells) * 2.0]
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)

    # Add point data: VARLOCATION = ([6] = CELLCENTERED)
    mesh.point_data["three"] = np.ones(num_points) * 3.0
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([6-7] = CELLCENTERED)
    mesh.cell_data["four"] = [np.ones(num_cells) * 4.0]
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)

    # Add point data: VARLOCATION = ([7-8] = CELLCENTERED)
    mesh.point_data["five"] = np.ones(num_points) * 5.0
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)

    # Add cell data: VARLOCATION = ([7-9] = CELLCENTERED)
    mesh.cell_data["six"] = [np.ones(num_cells) * 6.0]
    helpers.write_read(tmp_path, writer, reader, mesh, 1.0e-15)
