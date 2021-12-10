import io
import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
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
def test(mesh, tmp_path):
    helpers.write_read(
        tmp_path, meshio.nastran.write, meshio.nastran.read, mesh, 1.0e-13
    )


@pytest.mark.parametrize("filename", ["cylinder.fem", "cylinder_cells_first.fem"])
def test_reference_file(filename):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "nastran" / filename

    mesh = meshio.read(filename)

    # points
    assert np.isclose(mesh.points.sum(), 16.5316866)

    # cells
    ref_num_cells = {
        "line": 241,
        "triangle": 171,
        "quad": 721,
        "pyramid": 1180,
        "tetra": 5309,
    }
    assert {
        cell_block.type: cell_block.data.sum() for cell_block in mesh.cells
    } == ref_num_cells


def test_long_format():
    filename = io.StringIO(
        "BEGIN BULK\n"
        "GRID*    43                             1.50000000000000 0.0\n"
        "*        0.\n"
        "ENDDATA\n"
    )

    mesh = meshio.read(filename, "nastran")

    # points
    assert len(mesh.points) == 1
    assert np.isclose(mesh.points.sum(), 1.5)
