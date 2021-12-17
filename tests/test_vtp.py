import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

test_set = [
    helpers.vertex_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.quad_mesh,
    helpers.tri_quad_mesh,
    helpers.polygon_mesh,
    helpers.polygon_mesh_one_cell,
    # helpers.polygon2_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_quad_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float32)]),
    helpers.add_cell_data(helpers.tri_mesh, [("b", (3,), np.float64)]),
    helpers.add_cell_data(helpers.polygon_mesh, [("a", (), np.float32)]),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
def test(mesh, data_type, tmp_path):
    binary, compression = data_type

    def writer(*args, **kwargs):
        return meshio.vtp.write(*args, binary=binary, compression=compression, **kwargs)

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/issues/17038#note_264052>
    tol = 1.0e-15 if binary else 1.0e-10
    helpers.write_read(tmp_path, writer, meshio.vtp.read, mesh, tol)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.vtp")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.vtp")


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("00_raw_binary.vtp", "triangle", 108, 56),
        ("01_raw_binary_int64.vtp", "triangle", 108, 56),
        ("02_raw_compressed.vtp", "triangle", 108, 56),
    ],
)
def test_read_from_file(filename, ref_cells, ref_num_cells, ref_num_pnt):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtp" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("03_cow_strips.vtp", "triangle", 5804, 2903),
        ("04_raw_binary_strips.vtp", "triangle", 108, 56),
    ],
)
def test_read_from_file_triangle_strips(
    filename, ref_cells, ref_num_cells, ref_num_pnt
):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtp" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


@pytest.mark.parametrize(
    "filename, ref_cell_dict, ref_num_pnt, ref_cell_blocks",
    [
        (
            "05_cow_strips_mixed.vtp",
            {"triangle": 734, "quad": 2519, "polygon": 10},
            2903,
            22,
        ),
    ],
)
def test_read_from_file_polydata_mixed(
    filename, ref_cell_dict, ref_num_pnt, ref_cell_blocks
):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtp" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == ref_cell_blocks
    cell_dict = {}
    for ref_cells in mesh.cells:
        if ref_cells.type not in cell_dict.keys():
            cell_dict[ref_cells.type] = len(ref_cells.data)
        else:
            cell_dict[ref_cells.type] += len(ref_cells.data)
    for cell_type, ref_cell_num in ref_cell_dict.items():
        assert cell_dict[cell_type] == ref_cell_num
    # assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt
