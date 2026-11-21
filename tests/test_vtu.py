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


def _write_raw_vertex_mesh(path, encoding="raw"):
    # One point and one vertex cell, stored as raw appended data. The
    # block sizes are chosen such that the base64 offset that the
    # connectivity array is rewritten to (40) equals the original offset
    # of the offsets array, the regression case from
    # <https://github.com/nschloe/meshio/issues/1571>.
    import struct

    def block(data):
        return struct.pack("<I", len(data)) + data

    blob = (
        block(np.array([[1.0, 2.0, 3.0]]).astype("<f8").tobytes())
        + block(np.array([0], dtype="<i8").tobytes())
        + block(np.array([1], dtype="<i8").tobytes())
        + block(np.array([1], dtype=np.uint8).tobytes())
    )
    header = f"""<VTKFile type="UnstructuredGrid" version="0.1" \
byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid>
    <Piece NumberOfPoints="1" NumberOfCells="1">
      <Points><DataArray type="Float64" Name="Points" \
NumberOfComponents="3" format="appended" offset="0"/></Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="appended" offset="28"/>
        <DataArray type="Int64" Name="offsets" format="appended" offset="40"/>
        <DataArray type="UInt8" Name="types" format="appended" offset="52"/>
      </Cells>
    </Piece>
  </UnstructuredGrid>
  <AppendedData encoding="{encoding}">
   _"""
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(blob)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")


def test_raw_binary_offset_collision(tmp_path):
    path = tmp_path / "collision.vtu"
    _write_raw_vertex_mesh(path)

    mesh = meshio.vtu.read(path)
    assert np.allclose(mesh.points, [[1.0, 2.0, 3.0]])
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "vertex"
    # Without the fix, connectivity and offsets are swapped and this
    # comes back as [[1]], an index that does not exist in the mesh.
    assert mesh.cells[0].data.tolist() == [[0]]


def test_raw_binary_wrong_encoding(tmp_path):
    # A file whose appended data is not XML-parseable falls back to the
    # raw-binary parser; if it does not actually declare the raw
    # encoding, that parser must refuse it with a ReadError instead of
    # silently treating the data as raw.
    path = tmp_path / "wrong-encoding.vtu"
    _write_raw_vertex_mesh(path, encoding="base64")

    with pytest.raises(meshio.ReadError):
        meshio.vtu.read(path)
