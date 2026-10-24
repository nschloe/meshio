import pathlib
from io import StringIO

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.polygon_mesh,
    ],
)
def test_obj(mesh, tmp_path):
    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(np.int32))

    helpers.write_read(tmp_path, meshio.obj.write, meshio.obj.read, mesh, 1.0e-12)


@pytest.mark.skip("Fails point data consistency check.")
@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells", [("elephav.obj", 3.678372172450000e05, 1148)]
)
def test_reference_file(filename, ref_sum, ref_num_cells):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "obj" / filename

    mesh = meshio.read(filename)
    tol = 1.0e-5
    s = np.sum(mesh.points)
    assert abs(s - ref_sum) < tol * abs(ref_sum)
    assert mesh.cells[0].type == "triangle"
    assert len(mesh.cells[0].data) == ref_num_cells


@pytest.mark.parametrize(
    "num_nodes, cell_type", [(3, "triangle"), (4, "quad"), (5, "polygon")]
)
@pytest.mark.parametrize("index_style", ["positive", "negative", "mixed"])
@pytest.mark.parametrize("full_references", [False, True])
def test_relative_vertex_indices(
    num_nodes, cell_type, index_style, full_references, tmp_path
):
    def vertex_records(index):
        records = [f"v {index} {index % 2} 0"]
        if full_references:
            records.extend([f"vt {index} 0", "vn 0 0 1"])
        return records

    def obj_text(style):
        records = []
        for index in range(num_nodes):
            records.extend(vertex_records(index))
        for face in range(2):
            if face:
                records.extend(vertex_records(num_nodes))
            indices = list(range(1 + face, num_nodes + 1 + face))
            if style == "negative":
                indices = list(range(-num_nodes, 0))
            elif style == "mixed":
                indices[1:] = range(1 - num_nodes, 0)
            references = [
                f"{index}/{index}/{index}" if full_references else str(index)
                for index in indices
            ]
            records.append("f " + " ".join(references))
        return "\n".join(records) + "\n"

    mesh = meshio.obj.read(StringIO(obj_text(index_style)))
    expected_cells = np.array([np.arange(num_nodes), np.arange(1, num_nodes + 1)])
    assert mesh.cells[0].type == cell_type
    np.testing.assert_array_equal(mesh.cells[0].data, expected_cells)
    expected_points = [[index, index % 2, 0] for index in range(num_nodes + 1)]
    np.testing.assert_array_equal(mesh.points, expected_points)

    positive_mesh = meshio.obj.read(StringIO(obj_text("positive")))
    np.testing.assert_array_equal(mesh.cells[0].data, positive_mesh.cells[0].data)
    for name, values in positive_mesh.point_data.items():
        np.testing.assert_array_equal(mesh.point_data[name], values)

    filename = tmp_path / "positive.obj"
    meshio.obj.write(filename, mesh)
    roundtrip = meshio.obj.read(filename)
    np.testing.assert_array_equal(roundtrip.points, positive_mesh.points)
    np.testing.assert_array_equal(roundtrip.cells[0].data, expected_cells)
    for name, values in positive_mesh.point_data.items():
        np.testing.assert_array_equal(roundtrip.point_data[name], values)
