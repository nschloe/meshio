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
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.polygon_mesh,
    ],
)
def test_obj(mesh, tmp_path):
    for k, c in enumerate(mesh.cells):
        mesh.cells[k] = meshio.CellBlock(c.type, c.data.astype(np.int32))

    helpers.write_read(tmp_path, meshio.obj.write, meshio.obj.read, mesh, 1.0e-12)


def test_multiple_texture_vertices_per_vertex(tmp_path):
    # Regression test for <https://github.com/nschloe/meshio/issues/1487>:
    # vertex 1 is referenced with two different texture indices (a UV seam),
    # so len(vt) > len(v). The reader splits the vertex so that every point
    # carries exactly one texture coordinate.
    obj = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vt 0.5 0.5
f 1/1 2/2 3/3
f 1/5 3/3 4/4
"""
    filename = tmp_path / "multi_vt.obj"
    filename.write_text(obj)
    mesh = meshio.read(filename)

    # the shared vertex is split into two points with identical coordinates
    assert len(mesh.points) == 5
    assert np.all(mesh.points[0] == mesh.points[3])
    assert mesh.cells[0].type == "triangle"
    assert np.all(mesh.cells[0].data == [[0, 1, 2], [3, 2, 4]])
    # both texture coordinates of the seam vertex are preserved
    vt = mesh.point_data["obj:vt"]
    assert len(vt) == len(mesh.points)
    assert np.all(vt[0] == [0.0, 0.0])
    assert np.all(vt[3] == [0.5, 0.5])

    # the split mesh round-trips through the writer without loss
    roundtrip = tmp_path / "roundtrip.obj"
    meshio.obj.write(roundtrip, mesh)
    mesh2 = meshio.obj.read(roundtrip)
    assert np.all(mesh2.points == mesh.points)
    assert np.all(mesh2.cells[0].data == mesh.cells[0].data)
    assert np.all(mesh2.point_data["obj:vt"] == vt)


def test_multiple_normals_per_vertex(tmp_path):
    # Same per-corner indexing issue as #1487, for vertex normals.
    obj = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
f 1//1 2//1 3//1
f 1//2 3//2 2//2
"""
    filename = tmp_path / "multi_vn.obj"
    filename.write_text(obj)
    mesh = meshio.read(filename)

    assert len(mesh.points) == 6
    vn = mesh.point_data["obj:vn"]
    assert len(vn) == len(mesh.points)
    assert np.all(vn[:3] == [0.0, 0.0, 1.0])
    assert np.all(vn[3:] == [0.0, 0.0, -1.0])


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
