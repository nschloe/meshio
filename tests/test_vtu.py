import pathlib
import xml.etree.ElementTree as ET

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



@pytest.mark.parametrize(
    "binary, compression",
    [(False, None), (True, None), (True, "zlib"), (True, "lzma")],
)
def test_field_data_roundtrip(binary, compression, tmp_path):
    """VTU field data should survive all supported output encodings."""
    field_data = {
        "sample_id": np.array([42], dtype=np.int32),
        "physical_region": np.array([5, 2], dtype=np.int32),
        "material_parameters": np.array([[1.2, 0.3], [2.1, 0.7]]),
    }
    mesh = meshio.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        field_data=field_data,
    )
    filename = tmp_path / "field_data.vtu"
    meshio.vtu.write(filename, mesh, binary=binary, compression=compression)

    # Field metadata must use a VTK-compatible XML representation.
    field_xml = ET.parse(filename).getroot().find("./UnstructuredGrid/FieldData")
    assert field_xml is not None
    for array in field_xml:
        original = field_data[array.attrib["Name"]]
        assert int(array.attrib["NumberOfTuples"]) == original.shape[0]
        assert int(array.attrib.get("NumberOfComponents", 1)) == (
            original.shape[1] if original.ndim == 2 else 1
        )

    restored = meshio.vtu.read(filename)
    assert restored.field_data.keys() == field_data.keys()
    for name, values in field_data.items():
        np.testing.assert_array_equal(restored.field_data[name], values)
    np.testing.assert_array_equal(restored.cells[0].data, mesh.cells[0].data)
    np.testing.assert_array_equal(restored.points, mesh.points)


@pytest.mark.parametrize(
    "binary, compression",
    [(False, None), (True, None), (True, "zlib"), (True, "lzma")],
)
@pytest.mark.parametrize(
    "field_value",
    [
        np.array(42, dtype=np.int32),
        np.array([], dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
    ],
    ids=["scalar", "empty-vector", "empty-matrix"],
)
def test_field_data_edge_cases(field_value, binary, compression, tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = meshio.Mesh(
        points,
        [("triangle", np.array([[0, 1, 2]]))],
        field_data={"test_value": field_value},
    )
    filename = tmp_path / "edge_case.vtu"
    meshio.vtu.write(filename, mesh, binary=binary, compression=compression)
    xml = ET.parse(filename).getroot().find("./UnstructuredGrid/FieldData/DataArray")
    assert xml is not None
    assert int(xml.attrib["NumberOfTuples"]) == len(np.atleast_1d(field_value))
    restored = meshio.vtu.read(filename)
    np.testing.assert_array_equal(
        restored.field_data["test_value"], np.atleast_1d(field_value)
    )
