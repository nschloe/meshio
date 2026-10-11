import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

test_set = [
    # helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_one_cell,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.polygon_mesh,
    helpers.polygon_mesh_one_cell,
    helpers.polygon2_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
    helpers.polyhedron_mesh,
    helpers.lagrange_high_order_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_quad_mesh, [("a", (), np.float64)]),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float32)]),
    helpers.add_cell_data(helpers.tri_mesh, [("b", (3,), np.float64)]),
    helpers.add_cell_data(helpers.polygon_mesh, [("a", (), np.float32)]),
    helpers.add_cell_data(helpers.polyhedron_mesh, [("a", (2,), np.float32)]),
]


@pytest.mark.parametrize("mesh", test_set)
@pytest.mark.parametrize(
    "data_type", [(False, None), (True, None), (True, "lzma"), (True, "zlib")]
)
def test(mesh, data_type, tmp_path):
    binary, compression = data_type

    def writer(*args, **kwargs):
        return meshio.vtu.write(*args, binary=binary, compression=compression, **kwargs)

    # ASCII files are only meant for debugging, VTK stores only 11 digits
    # <https://gitlab.kitware.com/vtk/vtk/-/issues/17038#note_264052>
    tol = 1.0e-15 if binary else 1.0e-10
    helpers.write_read(tmp_path, writer, meshio.vtu.read, mesh, tol)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.vtu")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.vtu")


@pytest.mark.parametrize(
    "filename, ref_cells, ref_num_cells, ref_num_pnt",
    [
        ("00_raw_binary.vtu", "tetra", 162, 64),
        ("01_raw_binary_int64.vtu", "tetra", 162, 64),
        ("02_raw_compressed.vtu", "tetra", 162, 64),
    ],
)
def test_read_from_file(filename, ref_cells, ref_num_cells, ref_num_pnt):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "vtu" / filename

    mesh = meshio.read(filename)
    assert len(mesh.cells) == 1
    assert ref_cells == mesh.cells[0].type
    assert len(mesh.cells[0].data) == ref_num_cells
    assert len(mesh.points) == ref_num_pnt


def _write_minimal_raw_vtu(path, encoding=b"raw"):
    # A tiny raw-appended VTU where the rewritten base64 offset of the
    # connectivity block (72) equals the original raw offset of the
    # offsets block. See https://github.com/nschloe/meshio/issues/1571.
    def block(arr):
        b = np.asarray(arr).tobytes()
        return np.uint32(len(b)).tobytes() + b

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    connectivity = np.array([0, 1], dtype=np.int64)
    offsets = np.array([2], dtype=np.int64)
    types = np.array([3], dtype=np.uint8)
    # last payload byte is 0x0a, exercising trailing-whitespace handling
    tags = np.array([1, 10], dtype=np.int8)

    payload = (
        b"_"
        + block(points)
        + block(connectivity)
        + block(offsets)
        + block(types)
        + block(tags)
    )
    data = (
        b'<VTKFile type="UnstructuredGrid" version="0.1"'
        b' byte_order="LittleEndian" header_type="UInt32">\n'
        b"  <UnstructuredGrid>"
        b'<Piece NumberOfPoints="2" NumberOfCells="1">\n'
        b"    <PointData>"
        b'<DataArray type="Int8" Name="tags" format="appended" offset="89"/>'
        b"</PointData>\n"
        b"    <Points>"
        b'<DataArray type="Float64" NumberOfComponents="3"'
        b' format="appended" offset="0"/>'
        b"</Points>\n"
        b"    <Cells>\n"
        b'      <DataArray type="Int64" Name="connectivity"'
        b' format="appended" offset="52"/>\n'
        b'      <DataArray type="Int64" Name="offsets"'
        b' format="appended" offset="72"/>\n'
        b'      <DataArray type="UInt8" Name="types"'
        b' format="appended" offset="84"/>\n'
        b"    </Cells>\n"
        b"  </Piece></UnstructuredGrid>\n"
        b'<AppendedData encoding="'
        + encoding
        + b'">\n'
        + payload
        + b"\n</AppendedData>\n</VTKFile>"
    )
    path.write_bytes(data)


def test_raw_binary_offset_rewrite_collision(tmp_path):
    filename = tmp_path / "test.vtu"
    _write_minimal_raw_vtu(filename)
    mesh = meshio.read(filename)
    assert mesh.points.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert mesh.cells[0].data.tolist() == [[0, 1]]
    assert mesh.point_data["tags"].tolist() == [1, 10]


def test_raw_binary_requires_raw_encoding(tmp_path):
    from meshio._exceptions import ReadError
    from meshio.vtu._vtu import _parse_raw_binary

    filename = tmp_path / "test.vtu"
    _write_minimal_raw_vtu(filename, encoding=b"base64")
    with pytest.raises(ReadError):
        _parse_raw_binary(str(filename))
