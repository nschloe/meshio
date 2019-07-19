from functools import partial

import numpy
import pytest

import helpers
import meshio

test_set = [
    helpers.tri_mesh,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.polygon_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, 2),
    helpers.add_cell_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.add_point_data(helpers.tri_mesh_2d, 2), 2),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("write_binary", [True, False])
def test(mesh, write_binary):
    def writer(*args, **kwargs):
        return meshio._vtk.write(*args, write_binary=write_binary, **kwargs)

    helpers.write_read(writer, meshio._vtk.read, mesh, 1.0e-15)
    return


def test_generic_io():
    helpers.generic_io("test.vtk")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.vtk")
    return


@pytest.mark.parametrize(
    "filename, md5, ref_sum, ref_num_cells",
    [("vtk/rbc_001.vtk", "19f431dcb07971d5f29de33d6bbed79a", 0.00031280518, 996)],
)
@pytest.mark.parametrize("write_binary", [False, True])
def test_reference_file(filename, md5, ref_sum, ref_num_cells, write_binary):
    filename = helpers.download(filename, md5)

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * ref_sum
    assert len(mesh.cells["triangle"]) == ref_num_cells
    writer = partial(meshio._vtk.write, write_binary=write_binary)
    helpers.write_read(writer, meshio._vtk.read, mesh, 1.0e-15)
    return


@pytest.mark.parametrize(
    "filename, md5, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("vtk/00_image.vtk", "2c800ca9734fe92172d9693f0e58fe8f", "quad", 81, 100),
        ("vtk/01_image.vtk", "3981abae840ef521121bf8829252ae04", "hexahedron", 72, 147),
        (
            "vtk/02_structured.vtk",
            "0a958a46b7629149abb03e4af47b7d29",
            "hexahedron",
            72,
            147,
        ),
        (
            "vtk/03_rectilinear.vtk",
            "7b5c057940e61e88a933f7a63e99aae0",
            "hexahedron",
            72,
            147,
        ),
        ("vtk/04_rectilinear.vtk", "f75ddbebb907fdd73159bfcfc46fdbd5", "quad", 27, 40),
        ("vtk/05_rectilinear.vtk", "9f9bbccb7d76277b457162c0a2d3f9e9", "quad", 27, 40),
    ],
)
def test_structured(filename, md5, ref_cells, ref_num_cells, ref_num_pnt):
    filename = helpers.download(filename, md5)
    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells in mesh.cells
    assert len(mesh.cells[ref_cells]) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


if __name__ == "__main__":
    test(helpers.tri_mesh, write_binary=True)
