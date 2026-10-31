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


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        7.0,
        1234.56789,
        -0.1234,
        3.1415926535897932,
        # three-digit exponents used to trip an assertion in
        # _float_to_nastran_string since the formatted string, at the
        # default precision, is longer than the 16-character field
        1.234567890123e100,
        -1.234567890123e-100,
        1.500000000001e308,
        -1.500000000001e308,
        5e-324,
    ],
)
def test_float_to_nastran_string(value):
    from meshio.nastran._nastran import _float_to_nastran_string

    out = _float_to_nastran_string(value)
    assert len(out) <= 16
    # nastran requires a decimal point in real numbers
    assert "." in out
    # the string must still represent the value (at reduced precision
    # for the extreme values)
    assert float(out) == pytest.approx(value, rel=1.0e-9)


def test_write_extreme_coordinates(tmp_path):
    # Writing a mesh whose coordinates have three-digit exponents used
    # to fail with an AssertionError from _float_to_nastran_string.
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.500000000001e308, 0.0, 0.0],
                [0.0, -1.234567890123e-100, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]]))],
    )
    path = tmp_path / "out.bdf"
    meshio.nastran.write(path, mesh)
    out = meshio.read(path)
    assert np.allclose(out.points[2], [0.0, 0.0, 1.0])
    assert out.points[0][0] == pytest.approx(1.500000000001e308, rel=1.0e-9)
    assert out.points[1][1] == pytest.approx(-1.234567890123e-100, rel=1.0e-9)


def test_float_to_nastran_string_short_field():
    from meshio.nastran._nastran import _float_to_nastran_string

    out = _float_to_nastran_string(1234.56789, length=10)
    assert len(out) <= 10
    assert float(out) == pytest.approx(1234.56789, rel=1.0e-3)


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
