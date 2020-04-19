import pathlib
from functools import partial

import numpy
import pytest

import helpers
import meshio

test_set = [
    helpers.line_mesh,
    helpers.tri_mesh_2d,
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
    # VTK files float data is always stored in big endian
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), ">f8")]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), ">f8")]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), ">f8")]),
    helpers.add_cell_data(
        helpers.add_point_data(helpers.tri_mesh_2d, 2), [("a", (2,), ">f8")]
    ),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize("binary", [True, False])
def test(mesh, binary):
    def writer(*args, **kwargs):
        return meshio.vtk.write(*args, binary=binary, **kwargs)

    helpers.write_read(writer, meshio.vtk.read, mesh, 1.0e-15)


def test_generic_io():
    helpers.generic_io("test.vtk")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.vtk")


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("rbc_001.vtk", 0.00031280518, 996)]
)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file(filename, ref_sum, ref_num_cells, binary):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtk" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-2
    s = numpy.sum(mesh.points)
    assert abs(s - ref_sum) < tol * ref_sum
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells
    writer = partial(meshio.vtk.write, binary=binary)
    helpers.write_read(writer, meshio.vtk.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("00_image.vtk", "quad", 81, 100),
        ("01_image.vtk", "hexahedron", 72, 147),
        ("02_structured.vtk", "hexahedron", 72, 147),
        ("03_rectilinear.vtk", "hexahedron", 72, 147),
        ("04_rectilinear.vtk", "quad", 27, 40),
        ("05_rectilinear.vtk", "quad", 27, 40),
    ],
)
def test_structured(filename, ref_cells, ref_num_cells, ref_num_pnt):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtk" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


def test_pathlike():
    this_dir = pathlib.Path(__file__).resolve().parent
    meshio.read(this_dir / "meshes" / "vtk" / "rbc_001.vtk")


if __name__ == "__main__":
    test(helpers.tri_mesh, binary=True)
