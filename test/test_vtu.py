import numpy
import pytest

import helpers
import meshio

test_set = [
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), numpy.float64)]),
    helpers.add_cell_data(helpers.tri_quad_mesh, [("a", (), numpy.float64)]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), numpy.float32)]),
    helpers.add_cell_data(helpers.tri_mesh, [("b", (3,), numpy.float64)]),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
def test(mesh, data_type):
    binary, compression = data_type

    def writer(*args, **kwargs):
        return meshio.vtu.write(
            *args, binary=binary, compression=compression, **kwargs,
        )

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/issues/17038#note_264052>
    tol = 1.0e-15 if binary else 1.0e-10
    helpers.write_read(writer, meshio.vtu.read, mesh, tol)


def test_generic_io():
    helpers.generic_io("test.vtu")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.vtu")


if __name__ == "__main__":
    test(helpers.tet10_mesh, binary=False)
